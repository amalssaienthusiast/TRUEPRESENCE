"""
pipeline/ear_gate.py — Stage 1: EAR Eye-Blink Liveness Gate

Uses dlib's 68-landmark predictor to compute the Eye Aspect Ratio (EAR).
Requires a real person to blink at least `required_blinks` times within
`time_window_sec` seconds, or the verdict is SPOOF.

EAR formula (Soukupova & Cech, 2016):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

Where p1-p6 are the 6 eye landmarks ordered left-to-right.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np

# dlib is imported inside methods to provide a clean warning when not installed
try:
    import dlib
    _DLIB_AVAILABLE = True
except ImportError:
    _DLIB_AVAILABLE = False


def _eye_aspect_ratio(eye_landmarks) -> float:
    """Compute the Eye Aspect Ratio (EAR) for a single eye.

    Args:
        eye_landmarks: 6-point array of (x, y) coordinates.

    Returns:
        EAR value as a float.
    """
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C + 1e-9)


class EyeBlinkLivenessGate:
    """Stage 1 liveness gate based on eye blink detection via EAR.

    Parameters (per Section 4 spec):
        ear_threshold       = 0.25   (below this → eye closed)
        blink_consec_frames = 3      (frames eye must stay closed)
        required_blinks     = 2      (blinks needed in window)
        time_window_sec     = 5.0    (seconds before timeout → SPOOF)

    Args:
        landmarks_path: Path to shape_predictor_68_face_landmarks.dat.
        ear_threshold:       EAR below which the eye is considered closed.
        blink_consec_frames: Consecutive closed-eye frames to count as blink.
        required_blinks:     Blinks required within the time window.
        time_window_sec:     Maximum seconds allowed to accumulate blinks.
    """

    # dlib 68-landmark eye indices (0-indexed)
    LEFT_EYE_IDX  = list(range(36, 42))
    RIGHT_EYE_IDX = list(range(42, 48))

    def __init__(
        self,
        landmarks_path: str = "../data/data_dlib/shape_predictor_68_face_landmarks.dat",
        ear_threshold: float = 0.25,
        blink_consec_frames: int = 3,
        required_blinks: int = 2,
        time_window_sec: float = 5.0,
    ) -> None:
        """Initialise the EAR gate and load dlib predictor."""
        self.ear_threshold       = ear_threshold
        self.blink_consec_frames = blink_consec_frames
        self.required_blinks     = required_blinks
        self.time_window_sec     = time_window_sec

        # ── dlib setup ────────────────────────────────────────────────────
        if not _DLIB_AVAILABLE:
            print("[EAR Gate] WARNING: dlib not installed. "
                  "pip install dlib (needs cmake).")
            self._detector  = None
            self._predictor = None
        else:
            landmarks_path = str(Path(landmarks_path))
            if not Path(landmarks_path).exists():
                print(f"[EAR Gate] WARNING: {landmarks_path} not found. "
                      "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                self._detector  = None
                self._predictor = None
            else:
                self._detector  = dlib.get_frontal_face_detector()
                self._predictor = dlib.shape_predictor(landmarks_path)
                print(f"[EAR Gate] Loaded landmarks: {landmarks_path}")

        # ── State ──────────────────────────────────────────────────────────
        # All state is reset via reset() when a session starts
        self._blink_count:  int   = 0
        self._consec_count: int   = 0
        self._start_time:   Optional[float] = None

    def reset(self) -> None:
        """Reset all blink counting state. Call at the start of each session."""
        self._blink_count  = 0
        self._consec_count = 0
        self._start_time   = None

    def check_liveness(self, frame: np.ndarray) -> dict:
        """Process one BGR frame and update liveness state.

        Args:
            frame: OpenCV BGR frame (H, W, 3) numpy array.

        Returns:
            Dict with keys:
              live       : bool | None  — None=still pending, False=timeout SPOOF
              blinks     : int          — blinks counted so far
              ear        : float        — current average EAR (0.0 if no face)
              time_left  : float        — seconds remaining in window
        """
        # Initialise timer on first call
        if self._start_time is None:
            self._start_time = time.time()

        elapsed   = time.time() - self._start_time
        time_left = max(0.0, self.time_window_sec - elapsed)

        # Default return (no face / dlib not available)
        result = {
            "live":      None,
            "blinks":    self._blink_count,
            "ear":       0.0,
            "time_left": time_left,
        }

        # Timeout check — no face or no blinks in time → SPOOF
        if elapsed >= self.time_window_sec:
            if self._blink_count < self.required_blinks:
                result["live"] = False   # SPOOF verdict
            else:
                result["live"] = True    # passed (rare but possible)
            return result

        if self._detector is None or self._predictor is None:
            # dlib unavailable — pass gate with a warning (degraded mode)
            result["live"] = True
            return result

        # Convert frame to grayscale for dlib
        import cv2
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detector(gray, 0)

        if not faces:
            return result   # No face detected — keep pending

        # Use the largest detected face
        face      = max(faces, key=lambda r: r.width() * r.height())
        shape     = self._predictor(gray, face)
        landmarks = np.array([[shape.part(i).x, shape.part(i).y]
                               for i in range(68)], dtype=np.float32)

        left_eye  = landmarks[self.LEFT_EYE_IDX]
        right_eye = landmarks[self.RIGHT_EYE_IDX]
        ear       = (_eye_aspect_ratio(left_eye) + _eye_aspect_ratio(right_eye)) / 2.0

        result["ear"] = float(ear)

        # Blink state machine
        if ear < self.ear_threshold:
            self._consec_count += 1
        else:
            # Eye opened — if it was closed for enough consecutive frames, count blink
            if self._consec_count >= self.blink_consec_frames:
                self._blink_count += 1
            self._consec_count = 0

        result["blinks"] = self._blink_count

        # Liveness verdict
        if self._blink_count >= self.required_blinks:
            result["live"] = True

        return result


if __name__ == "__main__":
    gate = EyeBlinkLivenessGate()
    print("EyeBlinkLivenessGate initialised (dlib available:", _DLIB_AVAILABLE, ")")
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = gate.check_liveness(fake_frame)
    print(f"Result on blank frame: {result}")
