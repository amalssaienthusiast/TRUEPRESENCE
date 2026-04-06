"""
pipeline/screen_gate.py — Stage 2: Mobile/Screen Object Detector Gate

Uses a trained YOLOv8n model to detect mobile phones, tablet screens,
or persons displayed on screens in the camera frame.

Decision logic (per Section 4 spec):
  IF any detection with class in [mobile_phone, tablet_screen, person_on_screen]
  AND confidence > 0.45
  AND detection IoU with face bbox > 0.1 (or face IoU > 0.3)
  THEN verdict = "SPOOF"
"""

from pathlib import Path
from typing import Optional

import numpy as np


def _iou(box_a: list, box_b: list) -> float:
    """Compute Intersection over Union between two [x1,y1,x2,y2] boxes.

    Args:
        box_a: First bounding box [x1, y1, x2, y2].
        box_b: Second bounding box [x1, y1, x2, y2].

    Returns:
        IoU value in [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter_area
    return inter_area / (union + 1e-9)


class ScreenSpoofGate:
    """Stage 2 gate — detects if a mobile phone / screen is present near the face.

    Args:
        weights_path:           Path to trained .pt or .onnx screen detector.
        conf_threshold:         Minimum YOLO confidence (default 0.45).
        iou_threshold:          NMS IoU threshold (default 0.45).
        face_overlap_threshold: Minimum IoU between detection and face bbox
                                to trigger SPOOF (default 0.1).
        face_iou_spoof_thresh:  If face bbox IoU with phone > this → SPOOF (0.3).
        device:                 Inference device ('cpu', 'cuda', 'mps').
    """

    SPOOF_CLASSES = {"mobile_phone", "person_on_screen", "tablet_screen"}

    def __init__(
        self,
        weights_path: str = "checkpoints/best_screen_detector.pt",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        face_overlap_threshold: float = 0.1,
        face_iou_spoof_thresh: float = 0.3,
        device: str = "cpu",
    ) -> None:
        """Load the YOLO screen detector."""
        self.conf_threshold      = conf_threshold
        self.iou_threshold       = iou_threshold
        self.face_overlap_thresh = face_overlap_threshold
        self.face_iou_spoof_thresh = face_iou_spoof_thresh
        self.device              = device

        # Lazy import — avoids hard ultralytics dependency at import time
        self._model = None
        if Path(weights_path).exists():
            self.load_model(weights_path)
        else:
            print(f"[ScreenGate] WARNING: {weights_path} not found. "
                  "Train the screen detector first: python training/train_detector.py")

    def load_model(self, weights_path: str) -> None:
        """Load or reload the YOLOv8 model.

        Args:
            weights_path: Path to .pt or .onnx model file.
        """
        try:
            from ultralytics import YOLO
            self._model = YOLO(weights_path)
            print(f"[ScreenGate] Model loaded: {weights_path}")
        except ImportError:
            print("[ScreenGate] Install ultralytics: pip install ultralytics")
        except Exception as e:
            print(f"[ScreenGate] Load error: {e}")

    def check_frame(self, frame: np.ndarray, face_bbox: Optional[tuple] = None) -> dict:
        """Run Stage 2 detection on one BGR frame.

        Args:
            frame:     OpenCV BGR frame (H, W, 3).
            face_bbox: Optional (x1, y1, x2, y2) tuple of the detected face region.
                       If None, proximity check is skipped and any detection = SPOOF.

        Returns:
            Dict with:
              spoof_detected : bool
              detections     : list[dict] of raw YOLO detections
              verdict        : 'SPOOF' | 'PASS'
              reason         : str | None
        """
        result = {
            "spoof_detected": False,
            "detections":     [],
            "verdict":        "PASS",
            "reason":         None,
        }

        if self._model is None:
            # No model loaded → pass gate (graceful degradation)
            return result

        raw = self._model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )

        from models.screen_detector import ScreenDetectorWrapper
        class_map = ScreenDetectorWrapper.CLASS_NAMES

        for r in raw:
            for box in r.boxes:
                cls_id   = int(box.cls[0])
                cls_name = class_map.get(cls_id, f"cls_{cls_id}")
                conf     = float(box.conf[0])
                bbox     = box.xyxy[0].tolist()

                det = {"class": cls_name, "class_id": cls_id,
                       "conf": conf, "bbox": bbox}
                result["detections"].append(det)

                if cls_name not in self.SPOOF_CLASSES:
                    continue

                # Check IoU with face bbox
                if face_bbox is not None:
                    face_box = list(face_bbox)
                    overlap = _iou(bbox, face_box)
                    # Two thresholds per spec: 0.1 (detection overlaps face)
                    # and 0.3 (face heavily inside device bbox)
                    if overlap > self.face_overlap_thresh or overlap > self.face_iou_spoof_thresh:
                        result["spoof_detected"] = True
                        result["verdict"]        = "SPOOF"
                        result["reason"]         = f"{cls_name} detected near face (IoU={overlap:.2f})"
                else:
                    # No face bbox provided — any spoof class = SPOOF
                    result["spoof_detected"] = True
                    result["verdict"]        = "SPOOF"
                    result["reason"]         = f"{cls_name} detected in frame"

        return result


if __name__ == "__main__":
    gate = ScreenSpoofGate(weights_path="checkpoints/best_screen_detector.pt")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res   = gate.check_frame(blank, face_bbox=(100, 100, 300, 300))
    print(f"Result: {res}")
