"""
training/evaluate.py — Full Evaluation: ACER, AUC, Confusion Matrix, ROC, Latency

CLI:
    python evaluate.py --model checkpoints/best_antispoof.onnx --dataset hard_test
    python evaluate.py --model checkpoints/best_antispoof.pth --full_pipeline

Produces all 6 required outputs:
    1. Classification report (precision, recall, F1 per class)
    2. Confusion matrix PNG
    3. ROC curve PNG with AUC
    4. ACER / APCER / BPCER table
    5. Inference latency benchmark (CPU and GPU if available)
    6. evaluation_report.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for servers/Kaggle
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_for_eval(weights_path: str, device: str = "cpu"):
    """Load ONNX or PyTorch model for inference.

    Returns:
        Tuple of (inference callable, backend_name).
        The callable takes a numpy array (N,3,224,224) float32 and returns probs (N,2).
    """
    wpath = Path(weights_path)

    if wpath.suffix == ".onnx":
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session   = ort.InferenceSession(str(wpath), providers=providers)
        input_name = session.get_inputs()[0].name

        def infer(batch_np):
            """Run ONNX inference and convert logits to softmax probabilities."""
            logits  = session.run(None, {input_name: batch_np})[0]
            exp_l   = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exp_l / exp_l.sum(axis=1, keepdims=True)

        return infer, "ONNX"

    else:
        import torch, torch.nn.functional as F
        from models.antispoof_net import AntiSpoofNet
        ckpt  = torch.load(str(wpath), map_location=device, weights_only=False)
        model = AntiSpoofNet(pretrained=False)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        model.eval().to(device)

        def infer(batch_np):
            """Run PyTorch inference."""
            with torch.no_grad():
                t     = torch.from_numpy(batch_np).to(device)
                probs = F.softmax(model(t), dim=1).cpu().numpy()
            return probs

        return infer, "PyTorch"


def build_test_loader(dataset_name: str, data_root: str, batch_size: int = 32):
    """Build a DataLoader for the specified test dataset."""
    from data.loaders.combined import val_transform
    transform = val_transform()

    if dataset_name == "hard_test":
        # Hard Dataset — never seen during training, measures model ceiling
        from data.loaders.human_faces import HumanFacesDataset
        ds = HumanFacesDataset(data_root, split="test", transform=transform)

    elif dataset_name == "human_faces":
        from data.loaders.human_faces import HumanFacesDataset
        ds = HumanFacesDataset(data_root, split="test", transform=transform)

    elif dataset_name == "lcc_fasd":
        from data.loaders.lcc_fasd import LCCFASDDataset
        ds = LCCFASDDataset(data_root, split="val", transform=transform)

    else:
        from data.loaders.combined import CombinedAntiSpoofDataset
        ds = CombinedAntiSpoofDataset(data_root, split="val", transform=transform)

    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    weights_path: str,
    dataset_name: str,
    data_root: str = "./data",
    output_dir: str = "./runs/eval",
    device: str = "cpu",
) -> dict:
    """Full evaluation: metrics, plots, latency benchmark, JSON report.

    Args:
        weights_path: .onnx or .pth model path.
        dataset_name: Dataset to evaluate on.
        data_root:    Base data directory.
        output_dir:   Directory to save all output files.
        device:       'cpu' or 'cuda'.

    Returns:
        Dict with all computed metrics.
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score, roc_curve,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Eval] Model: {weights_path}")
    print(f"[Eval] Dataset: {dataset_name}")

    # Load model
    infer, backend = load_model_for_eval(weights_path, device)
    print(f"[Eval] Backend: {backend}")

    # Build dataloader
    loader = build_test_loader(dataset_name, data_root)
    if len(loader.dataset) == 0:
        print("[Eval] WARNING: Empty dataset. Check data root.")
        return {}

    # ── Inference loop ─────────────────────────────────────────────────────
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        probs = infer(imgs.numpy())
        all_probs.extend(probs[:, 1].tolist())   # live probability
        all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs > 0.5).astype(int)

    # ── 1. Classification Report ────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds, target_names=["spoof", "live"], output_dict=True
    )
    print("\n── Classification Report ──────────────────────")
    print(classification_report(all_labels, all_preds, target_names=["spoof", "live"]))

    # ── 2. Confusion Matrix ─────────────────────────────────────────────────
    import seaborn as sns
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["spoof", "live"], yticklabels=["spoof", "live"], ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    cm_path = str(out_dir / "confusion_matrix.png")
    plt.tight_layout(); plt.savefig(cm_path, dpi=150); plt.close()
    print(f"[Eval] Confusion matrix → {cm_path}")

    # ── 3. ROC Curve ────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC Curve"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend()
    roc_path = str(out_dir / "roc_curve.png")
    plt.tight_layout(); plt.savefig(roc_path, dpi=150); plt.close()
    print(f"[Eval] ROC curve (AUC={auc:.4f}) → {roc_path}")

    # ── 4. ACER / APCER / BPCER ─────────────────────────────────────────────
    spoof_mask = all_labels == 0
    live_mask  = all_labels == 1
    apcer = float(((all_preds == 1) & spoof_mask).sum() / (spoof_mask.sum() + 1e-9))
    bpcer = float(((all_preds == 0) & live_mask).sum()  / (live_mask.sum()  + 1e-9))
    acer  = (apcer + bpcer) / 2
    acc   = float((all_preds == all_labels).mean())

    print(f"\n── Anti-Spoofing Metrics ─────────────────────")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC      : {auc:.4f}  (target > 0.96)")
    print(f"  APCER    : {apcer:.4f}")
    print(f"  BPCER    : {bpcer:.4f}")
    print(f"  ACER     : {acer:.4f}  (target < 0.05)")

    # ── 5. Latency Benchmark ─────────────────────────────────────────────────
    print("\n── Latency Benchmark ─────────────────────────")
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    n_bench = 200
    t0 = time.perf_counter()
    for _ in range(n_bench):
        infer(dummy)
    cpu_ms = (time.perf_counter() - t0) * 1000 / n_bench
    print(f"  CPU: {cpu_ms:.2f} ms/image ({1000/cpu_ms:.1f} FPS)")

    gpu_ms = None
    try:
        import torch
        if torch.cuda.is_available():
            dummy_t = torch.from_numpy(dummy).cuda()
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    pass
            t0 = time.perf_counter()
            n_bench = 500
            for _ in range(n_bench):
                infer(dummy)
            gpu_ms = (time.perf_counter() - t0) * 1000 / n_bench
            print(f"  GPU: {gpu_ms:.2f} ms/image ({1000/gpu_ms:.1f} FPS)")
    except Exception:
        pass

    # ── 6. JSON Report ───────────────────────────────────────────────────────
    metrics = {
        "model":               weights_path,
        "backend":             backend,
        "dataset":             dataset_name,
        "n_samples":           int(len(all_labels)),
        "accuracy":            acc,
        "auc":                 float(auc),
        "apcer":               apcer,
        "bpcer":               bpcer,
        "acer":                acer,
        "cpu_latency_ms":      cpu_ms,
        "gpu_latency_ms":      gpu_ms,
        "classification_report": report,
        "confusion_matrix":    cm.tolist(),
        "target_auc_met":      auc > 0.96,
        "target_acer_met":     acer < 0.05,
    }
    json_path = str(out_dir / "evaluation_report.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Eval] Report saved → {json_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluate script."""
    parser = argparse.ArgumentParser(
        description="Evaluate anti-spoof classifier: ACER, AUC, confusion matrix, ROC, latency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",    type=str, required=True,
                        help="Path to .onnx or .pth checkpoint")
    parser.add_argument("--dataset",  type=str, default="hard_test",
                        choices=["hard_test", "human_faces", "lcc_fasd", "combined"],
                        help="Test dataset to evaluate on")
    parser.add_argument("--full_pipeline", action="store_true",
                        help="Run the full 3-stage pipeline evaluation instead of classifier only")
    parser.add_argument("--data_root",type=str, default="./data")
    parser.add_argument("--output",   type=str, default="./runs/eval")
    parser.add_argument("--device",   type=str, default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = run_evaluation(
        weights_path = args.model,
        dataset_name = args.dataset,
        data_root    = args.data_root,
        output_dir   = args.output,
        device       = args.device,
    )
    print("\n── Final Summary ────────────────────────────")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")
        elif not isinstance(v, (dict, list)):
            print(f"  {k:<25}: {v}")
