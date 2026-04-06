"""
training/config.py — Centralised hyperparameter configuration.

All training hyperparameters live here as frozen dataclasses.
Supports YAML override and auto-detects Kaggle / Colab environments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import yaml


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _is_kaggle() -> bool:
    """Detect Kaggle kernel environment."""
    return os.path.exists("/kaggle")


def _is_colab() -> bool:
    """Detect Google Colab environment."""
    return os.path.exists("/content")


def _default_data_root() -> str:
    """Return the most appropriate data root for the current environment."""
    if _is_kaggle():
        return "/kaggle/input"
    if _is_colab():
        return "/content/data"
    return str(Path(__file__).parent.parent / "data")


def _default_workers() -> int:
    """Kaggle T4 caps at 2 useful workers; locally use 4."""
    return 2 if (_is_kaggle() or _is_colab()) else 4


def _default_batch() -> int:
    """Kaggle/Colab T4 can handle 64; local GPUs default to 32."""
    return 64 if (_is_kaggle() or _is_colab()) else 32


# ---------------------------------------------------------------------------
# Stage 3 — Real vs Fake Face Classifier
# ---------------------------------------------------------------------------

@dataclass
class ClassifierConfig:
    """Hyperparameters for Stage 3 MobileNetV3 anti-spoof classifier."""

    # Paths
    data_root: str = field(default_factory=_default_data_root)
    dataset: str = "combined"           # lcc_fasd | celeba_spoof | human_faces | fake_140k | combined
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs/classifier"

    # Architecture
    backbone: str = "mobilenet_v3_small"
    pretrained: bool = True
    num_classes: int = 2

    # Training
    epochs: int = 30
    batch_size: int = field(default_factory=_default_batch)
    num_workers: int = field(default_factory=_default_workers)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True

    # Scheduler
    scheduler: str = "onecycle"         # onecycle | cosine | step
    pct_start: float = 0.1
    div_factor: float = 25.0

    # Loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    focal_weight: float = 0.8
    ce_weight: float = 0.2

    # Augmentations
    input_size: int = 224
    resize_size: int = 256
    hflip_prob: float = 0.5
    color_jitter: List[float] = field(
        default_factory=lambda: [0.3, 0.3, 0.2, 0.05]  # brightness, contrast, saturation, hue
    )
    rotation_deg: int = 15
    grayscale_prob: float = 0.05
    erasing_prob: float = 0.1

    # Early stopping
    patience: int = 7
    monitor_metric: str = "val_auc"

    # Targets
    target_acer: float = 0.05
    target_auc: float = 0.96

    def to_dict(self) -> dict:
        """Serialise config to plain dict."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "ClassifierConfig":
        """Load config from YAML file, overriding defaults."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# ---------------------------------------------------------------------------
# Stage 2 — YOLOv8 Screen/Phone Detector
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """Hyperparameters for Stage 2 YOLOv8n phone/screen detector."""

    # Paths
    data_root: str = field(default_factory=_default_data_root)
    data_yaml: str = "data/mobile_screen.yaml"
    checkpoint_dir: str = "checkpoints"
    project: str = "runs/detector"

    # Model
    model: str = "yolov8n.pt"           # Start from COCO pretrained

    # Training
    epochs: int = 80
    imgsz: int = 640
    batch: int = 32
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    patience: int = 20
    save_period: int = 5

    # Augmentation
    mosaic: float = 1.0
    copy_paste: float = 0.3
    degrees: float = 15.0
    flipud: float = 0.1
    mixup: float = 0.1
    close_mosaic: int = 10

    # Inference
    conf_threshold: float = 0.45
    iou_threshold: float = 0.45
    face_overlap_threshold: float = 0.1  # IoU with face bbox to flag SPOOF

    # Validation gate
    min_map50: float = 0.80

    @classmethod
    def from_yaml(cls, path: str) -> "DetectorConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# ---------------------------------------------------------------------------
# Stage 1 — EAR Blink Gate
# ---------------------------------------------------------------------------

@dataclass
class EARConfig:
    """Parameters for Stage 1 Eye Aspect Ratio blink liveness gate."""

    ear_threshold: float = 0.25        # EAR below this = eye closed
    blink_consec_frames: int = 3       # consecutive frames eye must close
    required_blinks: int = 2           # blinks needed within window
    time_window_sec: float = 5.0       # seconds to collect blinks
    landmarks_path: str = "../data/data_dlib/shape_predictor_68_face_landmarks.dat"


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Full pipeline configuration for AntiSpoofPipeline."""

    ear: EARConfig = field(default_factory=EARConfig)
    screen_weights: str = "checkpoints/best_screen_detector.pt"
    antispoof_weights: str = "checkpoints/best_antispoof.onnx"

    # Stage 3 decision threshold (slightly above 0.5 to cut false positives)
    live_threshold: float = 0.52

    device: str = "cuda"               # cuda | cpu | mps


# ---------------------------------------------------------------------------
# Convenience: auto-detected platform info
# ---------------------------------------------------------------------------

def print_environment() -> None:
    """Print detected environment information to stdout."""
    env = "Kaggle" if _is_kaggle() else ("Colab" if _is_colab() else "Local")
    print(f"[Config] Environment : {env}")
    print(f"[Config] Data root   : {_default_data_root()}")
    print(f"[Config] Workers     : {_default_workers()}")
    print(f"[Config] Batch size  : {_default_batch()}")


if __name__ == "__main__":
    print_environment()
    cfg = ClassifierConfig()
    print("\n[ClassifierConfig defaults]")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
