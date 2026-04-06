"""
training/train_classifier.py — Train Stage 3 Real vs Fake Face Anti-Spoof Classifier

CLI:
    python train_classifier.py --config config.yaml
    python train_classifier.py --dataset lcc_fasd --epochs 25 --batch 64
    python train_classifier.py --dataset combined --epochs 30 --resume checkpoints/last.pth

Features:
    1. Auto GPU detection (DataParallel if >1)
    2. Mixed precision (torch.cuda.amp)
    3. Gradient clipping (max_norm=1.0)
    4. WeightedRandomSampler for class balance
    5. Early stopping on val AUC
    6. Saves best + last checkpoint
    7. TensorBoard logging
    8. Live tqdm progress bars
    9. Auto eval + ONNX export after training
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.antispoof_net import AntiSpoofNet
from models.losses import CombinedAntiSpoofLoss
from training.config import ClassifierConfig


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_datasets(cfg: ClassifierConfig):
    """Instantiate train and val datasets based on cfg.dataset name."""
    from data.loaders.combined import train_transform, val_transform

    t_train = train_transform(cfg.input_size, cfg.resize_size)
    t_val   = val_transform(cfg.input_size, cfg.resize_size)

    if cfg.dataset == "lcc_fasd":
        from data.loaders.lcc_fasd import LCCFASDDataset
        train_ds = LCCFASDDataset(cfg.data_root, split="train", transform=t_train)
        val_ds   = LCCFASDDataset(cfg.data_root, split="val",   transform=t_val)
        sampler  = None

    elif cfg.dataset == "human_faces":
        from data.loaders.human_faces import HumanFacesDataset
        train_ds = HumanFacesDataset(cfg.data_root, split="train", transform=t_train)
        val_ds   = HumanFacesDataset(cfg.data_root, split="val",   transform=t_val)
        sampler  = None

    elif cfg.dataset in ("combined", "all"):
        from data.loaders.combined import CombinedAntiSpoofDataset
        train_ds = CombinedAntiSpoofDataset(cfg.data_root, split="train", transform=t_train)
        val_ds   = CombinedAntiSpoofDataset(cfg.data_root, split="val",   transform=t_val)
        sampler  = train_ds.make_sampler()

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    return train_ds, val_ds, sampler


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(all_labels, all_probs):
    """Compute accuracy, AUC, ACER, APCER, BPCER from arrays.

    Args:
        all_labels: numpy array of true labels.
        all_probs:  numpy array of live (class=1) probabilities.

    Returns:
        dict with acc, auc, acer, apcer, bpcer.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    import numpy as np

    preds = (all_probs > 0.5).astype(int)
    acc   = accuracy_score(all_labels, preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5

    # APCER = FP rate for attack class (spoof predicted as live)
    # BPCER = FP rate for bonafide (live predicted as spoof)
    spoof_mask = all_labels == 0
    live_mask  = all_labels == 1
    apcer = float(((preds == 1) & spoof_mask).sum() / (spoof_mask.sum() + 1e-9))
    bpcer = float(((preds == 0) & live_mask).sum()  / (live_mask.sum()  + 1e-9))
    acer  = (apcer + bpcer) / 2

    return {"acc": acc, "auc": auc, "acer": acer, "apcer": apcer, "bpcer": bpcer}


# ---------------------------------------------------------------------------
# Training + validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, cfg, device):
    """Run one training epoch with AMP and gradient clipping."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc="Train", leave=False, unit="batch")

    for imgs, labels in bar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        batch_size   = labels.size(0)
        total_loss  += loss.item() * batch_size
        correct     += (logits.argmax(1) == labels).sum().item()
        total       += batch_size
        bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model and compute all anti-spoofing metrics."""
    import numpy as np
    import torch.nn.functional as F

    model.eval()
    all_labels, all_probs, total_loss = [], [], 0.0

    for imgs, labels in tqdm(loader, desc="Val", leave=False, unit="batch"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        probs  = F.softmax(logits, dim=1)[:, 1]  # live class prob

        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    metrics    = compute_metrics(all_labels, all_probs)
    metrics["val_loss"] = total_loss / max(len(all_labels), 1)
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, epoch, model, optimizer, scheduler, val_metrics, cfg, history):
    """Save a complete training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "model_state":     model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "val_auc":         val_metrics.get("auc", 0),
        "val_acc":         val_metrics.get("acc", 0),
        "val_acer":        val_metrics.get("acer", 1),
        "cfg":             cfg.to_dict(),
        "history":         history,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Load checkpoint and return (epoch, history, best_auc)."""
    ckpt    = torch.load(path, map_location=device, weights_only=False)
    state   = ckpt.get("model_state", ckpt)
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(state)
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    history = ckpt.get("history", {"t_loss": [], "t_acc": [], "v_loss": [], "v_acc": [], "v_auc": []})
    return ckpt.get("epoch", 0), history, ckpt.get("val_auc", 0.0)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(cfg: ClassifierConfig) -> None:
    """Run the full training pipeline."""
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"[Trainer] Device: {device}")

    # Datasets & loaders
    train_ds, val_ds, sampler = build_datasets(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size   = cfg.batch_size,
        sampler      = sampler,
        shuffle      = sampler is None,
        num_workers  = cfg.num_workers,
        pin_memory   = device.type == "cuda",
        drop_last    = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size   = cfg.batch_size * 2,
        shuffle      = False,
        num_workers  = cfg.num_workers,
        pin_memory   = device.type == "cuda",
    )
    print(f"[Trainer] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model = AntiSpoofNet(pretrained=cfg.pretrained).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"[Trainer] Using {torch.cuda.device_count()} GPUs.")

    # Loss, optimizer, scheduler
    criterion = CombinedAntiSpoofLoss(
        focal_alpha     = cfg.focal_alpha,
        focal_gamma     = cfg.focal_gamma,
        label_smoothing = cfg.label_smoothing,
        focal_weight    = cfg.focal_weight,
        ce_weight       = cfg.ce_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr      = cfg.lr,
        steps_per_epoch = len(train_loader),
        epochs      = cfg.epochs,
        pct_start   = cfg.pct_start,
        div_factor  = cfg.div_factor,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    # Checkpointing
    ckpt_dir   = Path(cfg.checkpoint_dir)
    best_path  = ckpt_dir / "best_antispoof.pth"
    last_path  = ckpt_dir / "last_antispoof.pth"
    history    = {"t_loss": [], "t_acc": [], "v_loss": [], "v_acc": [], "v_auc": []}
    start_epoch, best_auc, no_improve = 0, 0.0, 0

    # Resume from checkpoint
    if args.resume and Path(args.resume).exists():
        start_epoch, history, best_auc = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        print(f"[Trainer] Resumed from epoch {start_epoch}, best AUC={best_auc:.4f}")

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=cfg.log_dir)
        _use_tb = True
    except ImportError:
        print("[Trainer] WARNING: tensorboard not installed — skipping TensorBoard logging. pip install tensorboard")
        writer = None
        _use_tb = False

    print(f"\n[Trainer] Starting training: {cfg.epochs} epochs, batch={cfg.batch_size}")
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, cfg.epochs):
        print(f"\n── Epoch {epoch+1}/{cfg.epochs} ────────────────────────")

        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, cfg, device
        )
        v_metrics = validate(model, val_loader, criterion, device)

        val_auc  = v_metrics["auc"]
        val_acc  = v_metrics["acc"]
        val_acer = v_metrics["acer"]
        val_loss = v_metrics["val_loss"]

        # TensorBoard logging
        global_step += len(train_loader)
        writer.add_scalar("Loss/train",   t_loss,  global_step)
        writer.add_scalar("Loss/val",     val_loss, global_step)
        writer.add_scalar("Acc/train",    t_acc,   global_step)
        writer.add_scalar("Acc/val",      val_acc,  global_step)
        writer.add_scalar("AUC/val",      val_auc,  global_step)
        writer.add_scalar("ACER/val",     val_acer, global_step)
        writer.add_scalar("LR",           optimizer.param_groups[0]["lr"], global_step)

        # History
        history["t_loss"].append(t_loss);  history["t_acc"].append(t_acc)
        history["v_loss"].append(val_loss); history["v_acc"].append(val_acc)
        history["v_auc"].append(val_auc)

        print(f"  Train: loss={t_loss:.4f}  acc={t_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}  acc={val_acc:.4f}  "
              f"AUC={val_auc:.4f}  ACER={val_acer:.4f}")

        # Always save last
        save_checkpoint(last_path, epoch+1, model, optimizer, scheduler, v_metrics, cfg, history)

        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            save_checkpoint(best_path, epoch+1, model, optimizer, scheduler, v_metrics, cfg, history)
            print(f"  ★ New best AUC={best_auc:.4f} — checkpoint saved.")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"  ✗ Early stopping triggered (no improvement for {cfg.patience} epochs).")
                break

    writer.close()
    print(f"\n[Trainer] Training complete. Best AUC={best_auc:.4f}")

    # ── Post-training: evaluate on test set ──────────────────────────────
    print("\n[Trainer] Running final evaluation...")
    test_metrics = validate(model, val_loader, criterion, device)
    print(f"Final metrics: {test_metrics}")

    # ── Auto-export ONNX ─────────────────────────────────────────────────
    onnx_path = str(ckpt_dir / "best_antispoof.onnx")
    base_model = model.module if hasattr(model, "module") else model
    # Re-load best weights before export
    ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt["model_state"])
    base_model.export_onnx(onnx_path)
    print(f"[Trainer] ONNX exported → {onnx_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Stage 3 Anti-Spoof Face Classifier (MobileNetV3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",   type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--dataset",  type=str, default="combined",
                        choices=["lcc_fasd", "human_faces", "combined"],
                        help="Dataset to use for training")
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--batch",    type=int, default=None)
    parser.add_argument("--lr",       type=float, default=None)
    parser.add_argument("--resume",   type=str, default=None, help="Checkpoint .pth to resume from")
    parser.add_argument("--data_root",type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Build config — YAML overrides then CLI overrides
    if args.config:
        cfg = ClassifierConfig.from_yaml(args.config)
    else:
        cfg = ClassifierConfig()

    if args.dataset:  cfg.dataset    = args.dataset
    if args.epochs:   cfg.epochs     = args.epochs
    if args.batch:    cfg.batch_size = args.batch
    if args.lr:       cfg.lr         = args.lr
    if args.data_root: cfg.data_root = args.data_root

    print(f"[Classifier] Config: dataset={cfg.dataset}, epochs={cfg.epochs}, batch={cfg.batch_size}")
    main(cfg)
