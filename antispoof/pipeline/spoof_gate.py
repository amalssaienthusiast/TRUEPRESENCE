"""
pipeline/spoof_gate.py — Stage 3: CNN Real vs Fake Face Classifier Gate

Runs the trained AntiSpoofNet (MobileNetV3-Small) on cropped face regions.
Supports both ONNX (fast, production) and .pth (PyTorch, training/debugging).
Threshold: live_confidence > 0.52 (slightly above 0.5 per spec to cut FP).
"""

from pathlib import Path
from typing import Optional

import numpy as np


class SpoofClassifierGate:
    """Stage 3 gate — classifies a face crop as live or spoof.

    Supports two inference backends:
      - ONNX  (preferred for production — no PyTorch overhead)
      - PyTorch .pth  (useful during training / debugging)

    Args:
        weights_path:    Path to .onnx or .pth checkpoint.
        live_threshold:  Minimum live class probability to pass (default 0.52).
        device:          'cpu', 'cuda', or 'mps' (only for .pth mode).
    """

    def __init__(
        self,
        weights_path: str = "checkpoints/best_antispoof.onnx",
        live_threshold: float = 0.52,
        device: str = "cpu",
    ) -> None:
        """Load the classifier model."""
        self.live_threshold = live_threshold
        self.device         = device
        self._ort_session   = None
        self._torch_model   = None
        self._is_onnx       = False
        self._input_name    = None   # set in _load_onnx

        wpath = Path(weights_path)
        if not wpath.exists():
            print(f"[SpoofGate] WARNING: {weights_path} not found. "
                  "Train classifier first: python training/train_classifier.py")
            return

        self._load(wpath)

    def _load(self, wpath: Path) -> None:
        """Load model from .onnx or .pth checkpoint."""
        if wpath.suffix == ".onnx":
            self._load_onnx(wpath)
        else:
            self._load_pth(wpath)

    def _load_onnx(self, wpath: Path) -> None:
        """Load ONNX Runtime inference session."""
        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._ort_session = ort.InferenceSession(str(wpath), providers=providers)
            self._is_onnx = True
            self._input_name = self._ort_session.get_inputs()[0].name
            print(f"[SpoofGate] ONNX loaded: {wpath} on {self._ort_session.get_providers()[0]}")
        except ImportError:
            print("[SpoofGate] Install onnxruntime: pip install onnxruntime")
        except Exception as e:
            print(f"[SpoofGate] ONNX load error: {e}")

    def _load_pth(self, wpath: Path) -> None:
        """Load PyTorch .pth checkpoint."""
        try:
            import torch
            from models.antispoof_net import AntiSpoofNet
            ckpt = torch.load(str(wpath), map_location=self.device, weights_only=False)
            model = AntiSpoofNet(pretrained=False)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state)
            model.eval()
            self._torch_model = model.to(self.device)
            print(f"[SpoofGate] PyTorch model loaded: {wpath}")
        except Exception as e:
            print(f"[SpoofGate] .pth load error: {e}")

    def classify(self, face_bgr: np.ndarray) -> dict:
        """Classify a face crop as live or spoof.

        Args:
            face_bgr: Cropped face region as a BGR numpy array (H, W, 3).

        Returns:
            Dict with:
              live_prob  : float  — probability of being a live (real) face
              spoof_prob : float  — probability of being a spoof/fake face
              verdict    : 'LIVE' | 'SPOOF'
              passed     : bool   — True if live_prob > live_threshold
        """
        result = {
            "live_prob":  0.5,
            "spoof_prob": 0.5,
            "verdict":    "SPOOF",
            "passed":     False,
        }

        if self._ort_session is None and self._torch_model is None:
            # No model — pass gate in degraded mode
            result["passed"] = True
            result["verdict"] = "LIVE"
            return result

        # Preprocess: BGR → RGB PIL → tensor
        from PIL import Image
        from torchvision import transforms as T
        _preprocess = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        face_rgb = face_bgr[:, :, ::-1]  # BGR→RGB
        pil_img  = Image.fromarray(face_rgb.astype(np.uint8))
        tensor   = _preprocess(pil_img).unsqueeze(0)  # (1, 3, 224, 224)

        if self._is_onnx:
            logits = self._ort_session.run(
                None, {self._input_name: tensor.numpy()}
            )[0]    # (1, 2)
            import numpy as np_
            # Softmax to turn logits into probabilities
            exp_logits = np_.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        else:
            import torch, torch.nn.functional as F
            with torch.no_grad():
                logits = self._torch_model(tensor.to(self.device))
                probs  = F.softmax(logits, dim=1).cpu().numpy()

        spoof_prob = float(probs[0, 0])  # class 0 = spoof
        live_prob  = float(probs[0, 1])  # class 1 = live

        result["live_prob"]  = live_prob
        result["spoof_prob"] = spoof_prob
        result["passed"]     = live_prob > self.live_threshold
        result["verdict"]    = "LIVE" if result["passed"] else "SPOOF"
        return result


if __name__ == "__main__":
    gate = SpoofClassifierGate(weights_path="checkpoints/best_antispoof.onnx")
    blank_face = np.zeros((224, 224, 3), dtype=np.uint8)
    res = gate.classify(blank_face)
    print(f"Classification result: {res}")
