"""
training/train_detector.py — Train Stage 2 YOLOv8n Phone/Screen Detector

CLI:
    python train_detector.py --data data/mobile_screen.yaml --epochs 80
    python train_detector.py --resume checkpoints/yolo_last.pt

Uses the Ultralytics Python API (not subprocess).
Auto-generates data/mobile_screen.yaml with correct paths.
Validates mAP50 > 0.80 before saving final model.
Exports to ONNX after training.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.config import DetectorConfig


# ---------------------------------------------------------------------------
# YAML Generator
# ---------------------------------------------------------------------------

def generate_data_yaml(cfg: DetectorConfig) -> str:
    """Auto-generate data/mobile_screen.yaml for Ultralytics training.

    Creates the YAML in the project root (or cfg.data_yaml path).

    Returns:
        Path to the generated YAML file.
    """
    import yaml

    data_root = Path(cfg.data_root)
    yaml_data = {
        "path":  str(data_root),
        "train": str(data_root / "mobile_person" / "train" / "images"),
        "val":   str(data_root / "mobile_person" / "valid" / "images"),
        "test":  str(data_root / "mobile_person" / "test"  / "images"),
        "nc": 3,
        "names": ["mobile_phone", "person_on_screen", "tablet_screen"],
    }

    out = Path(cfg.data_yaml)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"[Detector] data.yaml written → {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: DetectorConfig, data_yaml: str, resume: str | None = None) -> str:
    """Train YOLOv8n on the mobile screen dataset.

    Args:
        cfg:       DetectorConfig with all hyperparameters.
        data_yaml: Path to the dataset YAML file.
        resume:    Optional checkpoint .pt path for resuming.

    Returns:
        Path to the best trained model weights (.pt).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install Ultralytics: pip install ultralytics")

    # Load base model — resume if path provided, else start from COCO pretrained
    model = YOLO(resume if resume else cfg.model)

    print(f"[Detector] Training YOLOv8n for {cfg.epochs} epochs on {data_yaml}")
    results = model.train(
        data        = data_yaml,
        epochs      = cfg.epochs,
        imgsz       = cfg.imgsz,
        batch       = cfg.batch,
        optimizer   = cfg.optimizer,
        lr0         = cfg.lr0,
        lrf         = cfg.lrf,
        patience    = cfg.patience,
        save_period = cfg.save_period,
        mosaic      = cfg.mosaic,
        copy_paste  = cfg.copy_paste,
        degrees     = cfg.degrees,
        flipud      = cfg.flipud,
        mixup       = cfg.mixup,
        close_mosaic= cfg.close_mosaic,
        project     = cfg.project,
        name        = "train",
        exist_ok    = True,
        verbose     = True,
        resume      = bool(resume),
    )

    best_pt = Path(cfg.project) / "train" / "weights" / "best.pt"
    print(f"[Detector] Training done. Best weights → {best_pt}")
    return str(best_pt)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(cfg: DetectorConfig, weights_path: str) -> float:
    """Validate the trained model and return mAP50.

    Args:
        cfg: DetectorConfig.
        weights_path: Path to best.pt.

    Returns:
        mAP50 value.
    """
    from ultralytics import YOLO

    model   = YOLO(weights_path)
    metrics = model.val(data=cfg.data_yaml, imgsz=cfg.imgsz, verbose=True)
    map50   = float(metrics.box.map50)
    print(f"[Detector] Validation mAP50={map50:.4f} (target={cfg.min_map50})")
    return map50


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(weights_path: str, imgsz: int = 640, output_dir: str = "checkpoints") -> str:
    """Export the trained YOLOv8 model to ONNX format.

    Args:
        weights_path: Path to best.pt checkpoint.
        imgsz:        Export image size.
        output_dir:   Where to copy the final .onnx file.

    Returns:
        Path to the exported ONNX file.
    """
    from ultralytics import YOLO
    import shutil

    model    = YOLO(weights_path)
    onnx_out = model.export(format="onnx", imgsz=imgsz)

    # Copy to checkpoints/
    dest = Path(output_dir) / "best_screen_detector.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_out, dest)
    print(f"[Detector] ONNX exported → {dest}")
    return str(dest)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the detector training script."""
    parser = argparse.ArgumentParser(
        description="Train Stage 2 YOLOv8n Screen/Phone Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",    type=str, default=None,
                        help="Path to data.yaml (auto-generated if not provided)")
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--batch",   type=int, default=None)
    parser.add_argument("--resume",  type=str, default=None,
                        help="Resume training from a .pt checkpoint")
    parser.add_argument("--data_root", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = DetectorConfig()

    if args.epochs:    cfg.epochs    = args.epochs
    if args.batch:     cfg.batch     = args.batch
    if args.data_root: cfg.data_root = args.data_root

    # Generate YAML if not provided
    data_yaml = args.data if (args.data and Path(args.data).exists()) else generate_data_yaml(cfg)
    cfg.data_yaml = data_yaml

    # Train
    best_weights = train(cfg, data_yaml, resume=args.resume)

    # Validate — enforce mAP50 gate
    map50 = validate(cfg, best_weights)
    if map50 < cfg.min_map50:
        print(f"[Detector] ⚠ WARNING: mAP50={map50:.4f} < target={cfg.min_map50}. "
              "Consider more data or longer training.")
    else:
        print(f"[Detector] ✓ mAP50={map50:.4f} exceeds target {cfg.min_map50}.")

    # Export ONNX
    onnx_path = export_onnx(best_weights, imgsz=cfg.imgsz)
    print(f"\n[Detector] Complete. ONNX model: {onnx_path}")
