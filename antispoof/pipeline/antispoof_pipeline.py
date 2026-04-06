"""
pipeline/antispoof_pipeline.py — AntiSpoofPipeline Orchestrator

Runs all 3 stages in sequence on a single webcam BGR frame:
  Stage 1: EAR eye-blink liveness (no model, dlib landmarks)
  Stage 2: YOLOv8n mobile/screen detector
  Stage 3: MobileNetV3 real vs fake face classifier

Short-circuits: Stage 1 failure skips Stages 2 & 3.

Verdicts:
  "LIVE"    — all 3 stages passed
  "SPOOF"   — any stage failed
  "PENDING" — still within the blink time window
  "NO_FACE" — no face detected in frame
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .ear_gate import EyeBlinkLivenessGate
from .screen_gate import ScreenSpoofGate
from .spoof_gate import SpoofClassifierGate

# We import PipelineConfig locally to avoid circular imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from training.config import PipelineConfig


@dataclass
class PipelineResult:
    """Structured result from one pipeline run.

    Attributes:
        verdict:      "LIVE" | "SPOOF" | "PENDING" | "NO_FACE"
        stage_results: Raw result dict from each stage.
        confidence:   Live probability from Stage 3 (0.0 if not reached).
        latency_ms:   Total pipeline wall-clock time.
        spoof_reason: Which stage triggered SPOOF and why. None if LIVE/PENDING.
    """

    verdict:       str
    stage_results: dict = field(default_factory=dict)
    confidence:    float = 0.0
    latency_ms:    float = 0.0
    spoof_reason:  Optional[str] = None


class AntiSpoofPipeline:
    """Three-stage anti-spoofing pipeline orchestrator.

    Accepts a PipelineConfig and loads all three stage models.
    Stateful for Stage 1 (blink counting) — call reset() between subjects.

    Args:
        cfg: PipelineConfig dataclass with paths and thresholds.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        """Initialise and load all three stage models."""
        self.cfg = cfg

        # Stage 1 — EAR blink gate (uses dlib, no torch)
        self.stage1 = EyeBlinkLivenessGate(
            landmarks_path        = cfg.ear.landmarks_path,
            ear_threshold         = cfg.ear.ear_threshold,
            blink_consec_frames   = cfg.ear.blink_consec_frames,
            required_blinks       = cfg.ear.required_blinks,
            time_window_sec       = cfg.ear.time_window_sec,
        )

        # Stage 2 — Screen/phone YOLO detector
        self.stage2 = ScreenSpoofGate(
            weights_path  = cfg.screen_weights,
            device        = cfg.device,
        )

        # Stage 3 — CNN real vs fake classifier
        self.stage3 = SpoofClassifierGate(
            weights_path   = cfg.antispoof_weights,
            live_threshold = cfg.live_threshold,
            device         = cfg.device,
        )

        print("[Pipeline] Loaded: Stage1(EAR), Stage2(YOLO), Stage3(CNN)")

    def reset(self) -> None:
        """Reset Stage 1 blink counter. Call between subjects or sessions."""
        self.stage1.reset()

    def _detect_face(self, frame: np.ndarray) -> Optional[tuple]:
        """Detect the largest face in frame using OpenCV Haar cascade.

        Returns:
            (x1, y1, x2, y2) tuple or None if no face found.
        """
        import cv2
        # Use lightweight Haar cascade for face detection (no models needed)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces  = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
        if not len(faces):
            return None
        # Pick the largest face by area
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (x, y, x + w, y + h)

    def run(self, frame: np.ndarray) -> PipelineResult:
        """Process one webcam BGR frame through all 3 stages.

        Stages 2 and 3 are skipped if Stage 1 is still pending or has failed.

        Args:
            frame: OpenCV BGR frame (H, W, 3) numpy array.

        Returns:
            PipelineResult with verdict, per-stage results, confidence, latency.
        """
        t0 = time.perf_counter()
        stage_results = {"ear": {}, "screen": {}, "classifier": {}}

        # ── Face detection ────────────────────────────────────────────────
        face_bbox = self._detect_face(frame)
        if face_bbox is None:
            return PipelineResult(
                verdict      = "NO_FACE",
                stage_results= stage_results,
                latency_ms   = (time.perf_counter() - t0) * 1000,
            )

        # ── Stage 1 — EAR blink gate ─────────────────────────────────────
        ear_result = self.stage1.check_liveness(frame)
        stage_results["ear"] = ear_result

        if ear_result["live"] is None:
            # Still accumulating blinks
            return PipelineResult(
                verdict      = "PENDING",
                stage_results= stage_results,
                latency_ms   = (time.perf_counter() - t0) * 1000,
            )

        if ear_result["live"] is False:
            return PipelineResult(
                verdict      = "SPOOF",
                stage_results= stage_results,
                spoof_reason= "no_blink_detected",
                latency_ms   = (time.perf_counter() - t0) * 1000,
            )

        # ── Stage 2 — Screen/phone detector ──────────────────────────────
        screen_result = self.stage2.check_frame(frame, face_bbox=face_bbox)
        stage_results["screen"] = screen_result

        if screen_result["spoof_detected"]:
            return PipelineResult(
                verdict      = "SPOOF",
                stage_results= stage_results,
                spoof_reason= "mobile_screen_in_frame",
                latency_ms   = (time.perf_counter() - t0) * 1000,
            )

        # ── Stage 3 — CNN real vs fake classifier ─────────────────────────
        x1, y1, x2, y2 = face_bbox
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return PipelineResult(
                verdict      = "NO_FACE",
                stage_results= stage_results,
                latency_ms   = (time.perf_counter() - t0) * 1000,
            )

        cnn_result = self.stage3.classify(face_crop)
        stage_results["classifier"] = cnn_result

        if not cnn_result["passed"]:
            return PipelineResult(
                verdict      = "SPOOF",
                stage_results= stage_results,
                confidence   = cnn_result["live_prob"],
                spoof_reason= "synthetic_face_detected",
                latency_ms   = (time.perf_counter() - t0) * 1000,
            )

        # ── All stages passed → LIVE ───────────────────────────────────────
        return PipelineResult(
            verdict      = "LIVE",
            stage_results= stage_results,
            confidence   = cnn_result["live_prob"],
            latency_ms   = (time.perf_counter() - t0) * 1000,
        )

    def run_on_video(self, video_path: str, output_path: str) -> dict:
        """Process a video file frame-by-frame and write an annotated output.

        Args:
            video_path:  Path to input video file.
            output_path: Path for the annotated output video.

        Returns:
            Summary dict with per-frame verdicts and aggregate statistics.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        verdicts = []
        self.reset()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.run(frame)
            verdicts.append(result.verdict)

            # Annotate frame
            color = {
                "LIVE":    (0, 255, 0),
                "SPOOF":   (0, 0, 255),
                "PENDING": (255, 165, 0),
                "NO_FACE": (128, 128, 128),
            }.get(result.verdict, (255, 255, 255))

            label = f"{result.verdict} ({result.confidence:.2f})"
            cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, color, 2, cv2.LINE_AA)
            writer.write(frame)

        cap.release()
        writer.release()

        summary = {
            "total_frames":  len(verdicts),
            "live_frames":   verdicts.count("LIVE"),
            "spoof_frames":  verdicts.count("SPOOF"),
            "pending_frames":verdicts.count("PENDING"),
            "output_path":   output_path,
        }
        print(f"[Pipeline] Video processed → {output_path}")
        print(f"[Pipeline] Summary: {summary}")
        return summary

    def benchmark(self, n_frames: int = 200) -> dict:
        """Measure per-stage latency on synthetic frames.

        Args:
            n_frames: Number of blank frames to process.

        Returns:
            Dict with mean latency (ms) per stage and total pipeline.
        """
        import time

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        stage1_times, stage2_times, stage3_times, total_times = [], [], [], []

        for _ in range(n_frames):
            t = time.perf_counter()
            self.stage1.check_liveness(blank)
            stage1_times.append((time.perf_counter() - t) * 1000)

            t = time.perf_counter()
            self.stage2.check_frame(blank)
            stage2_times.append((time.perf_counter() - t) * 1000)

            t = time.perf_counter()
            self.stage3.classify(blank)
            stage3_times.append((time.perf_counter() - t) * 1000)

            t0 = time.perf_counter()
            self.run(blank)
            total_times.append((time.perf_counter() - t0) * 1000)

            self.reset()

        def _avg(lst): return sum(lst) / len(lst)
        result = {
            "stage1_ear_ms":      _avg(stage1_times),
            "stage2_screen_ms":   _avg(stage2_times),
            "stage3_cnn_ms":      _avg(stage3_times),
            "total_pipeline_ms":  _avg(total_times),
            "n_frames":           n_frames,
        }
        print(f"\n[Pipeline Benchmark — {n_frames} frames]")
        for k, v in result.items():
            if k != "n_frames":
                print(f"  {k:<24}: {v:.2f} ms")
        return result


if __name__ == "__main__":
    from training.config import PipelineConfig, EARConfig
    cfg = PipelineConfig(
        ear             = EARConfig(),
        screen_weights  = "checkpoints/best_screen_detector.pt",
        antispoof_weights = "checkpoints/best_antispoof.onnx",
        live_threshold  = 0.52,
        device          = "cpu",
    )
    pipeline = AntiSpoofPipeline(cfg)
    result   = pipeline.benchmark(n_frames=10)
    print(f"\nBenchmark result: {result}")
