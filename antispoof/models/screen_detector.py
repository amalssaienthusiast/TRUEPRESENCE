"""
models/screen_detector.py — YOLOv8n Screen/Phone Detector Wrapper (Stage 2)

Wraps the Ultralytics YOLOv8n model for phone/screen/tablet detection.
Used by pipeline/screen_gate.py at inference time.
Training is handled separately in training/train_detector.py.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class ScreenDetectorWrapper:
    """Thin wrapper around a trained YOLOv8n ONNX or PyTorch model.

    Handles detection of mobile phones, tablet screens, and persons
    appearing on screens. Used by ScreenSpoofGate.

    Args:
        weights_path: Path to .pt or .onnx model checkpoint.
        conf:         Confidence threshold (default 0.45 per spec).
        iou:          IoU threshold for NMS (default 0.45).
        device:       'cpu', 'cuda', or 'mps'.
    """

    # Class IDs aligned with data.yaml from mobile_screen loader
    CLASS_NAMES = {0: "mobile_phone", 1: "person_on_screen", 2: "tablet_screen"}

    def __init__(
        self,
        weights_path: str,
        conf: float = 0.45,
        iou: float = 0.45,
        device: str = "cpu",
    ) -> None:
        """Load the YOLOv8n model."""
        self.weights_path = Path(weights_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self._model = None

        if not self.weights_path.exists():
            print(f"[ScreenDetectorWrapper] WARNING: {weights_path} not found. "
                  "Call load_model() or train before running inference.")
            return

        self._load()

    def _load(self) -> None:
        """Internal model loading — deferred to avoid import errors at import time."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights_path))
            print(f"[ScreenDetectorWrapper] Loaded: {self.weights_path}")
        except ImportError:
            print("[ScreenDetectorWrapper] Install ultralytics: pip install ultralytics")
        except Exception as e:
            print(f"[ScreenDetectorWrapper] Load error: {e}")

    def load_model(self, weights_path: str) -> None:
        """Load or reload a model from a given weight path.

        Args:
            weights_path: Path to .pt or .onnx checkpoint.
        """
        self.weights_path = Path(weights_path)
        self._load()

    def detect(
        self, frame: np.ndarray
    ) -> List[dict]:
        """Run detection on a single BGR frame.

        Args:
            frame: OpenCV BGR frame as a numpy array (H, W, 3).

        Returns:
            List of detection dicts, each containing:
              {'class': str, 'class_id': int, 'conf': float, 'bbox': [x1,y1,x2,y2]}
        """
        if self._model is None:
            return []

        results = self._model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "class":    self.CLASS_NAMES.get(cls_id, f"cls_{cls_id}"),
                    "class_id": cls_id,
                    "conf":     float(box.conf[0]),
                    "bbox":     box.xyxy[0].tolist(),   # [x1, y1, x2, y2]
                })
        return detections

    def export_onnx(self, output_path: Optional[str] = None, imgsz: int = 640) -> str:
        """Export the loaded PyTorch model to ONNX.

        Args:
            output_path: Destination path (default: same dir as .pt with .onnx ext).
            imgsz:       Export image size.

        Returns:
            Path to the exported ONNX file.
        """
        if self._model is None:
            raise RuntimeError("No model loaded — call load_model() first.")

        out = output_path or str(self.weights_path.with_suffix(".onnx"))
        self._model.export(format="onnx", imgsz=imgsz)
        print(f"[ScreenDetectorWrapper] ONNX exported → {out}")
        return out


if __name__ == "__main__":
    # Smoke test — loads default yolov8n (COCO) if no custom weights exist
    wrapper = ScreenDetectorWrapper("checkpoints/best_screen_detector.pt")
    print(f"Model loaded: {wrapper._model is not None}")
    print(f"Classes: {wrapper.CLASS_NAMES}")
