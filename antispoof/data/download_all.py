"""
data/download_all.py — Master Dataset Download Script

CLI:
    python data/download_all.py --datasets all --output ./data
    python data/download_all.py --datasets lcc_fasd,human_faces --output ./data
    python data/download_all.py --list

Handles Kaggle CLI, Roboflow SDK, gdown, and direct HTTP downloads.
Skips datasets that are already downloaded.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Supported datasets and their metadata
DATASET_INFO = {
    "lcc_fasd": {
        "name":    "LCC FASD",
        "type":    "Anti-Spoofing",
        "access":  "Free (Kaggle)",
        "size":    "~1 GB",
        "method":  "kaggle",
        "check":   "LCC_FASD_development",
    },
    "celeba_spoof": {
        "name":    "CelebA-Spoof",
        "type":    "Anti-Spoofing",
        "access":  "Auth Required (GitHub request)",
        "size":    "~50 GB",
        "method":  "manual",
        "check":   "CelebA_Spoof",
    },
    "human_faces": {
        "name":    "Human Faces Dataset",
        "type":    "Real vs AI-Generated",
        "access":  "Free (Kaggle)",
        "size":    "~2 GB",
        "method":  "kaggle",
        "check":   "human_faces_dataset",
    },
    "fake_140k": {
        "name":    "140k Real and Fake Faces",
        "type":    "Real vs AI-Generated",
        "access":  "Free (Kaggle)",
        "size":    "~10 GB",
        "method":  "kaggle",
        "check":   "real_vs_fake",
    },
    "sfhq": {
        "name":    "SFHQ Synthetic Faces (Part 1)",
        "type":    "Synthetic Faces",
        "access":  "Free (Kaggle)",
        "size":    "~5 GB",
        "method":  "kaggle",
        "check":   "sfhq_part1",
    },
    "mobile_person": {
        "name":    "Mobile Person Dataset",
        "type":    "Object Detection (YOLO)",
        "access":  "Roboflow API (requires ROBOFLOW_API_KEY)",
        "size":    "~500 MB",
        "method":  "roboflow",
        "check":   "mobile_person",
    },
    "mobile_phone": {
        "name":    "Mobile Phone Dataset",
        "type":    "Object Detection (YOLO)",
        "access":  "Roboflow API (requires ROBOFLOW_API_KEY)",
        "size":    "~1 GB",
        "method":  "roboflow",
        "check":   "mobile_phone",
    },
}

# Kaggle CLI commands per dataset
KAGGLE_COMMANDS = {
    "lcc_fasd":    None,   # Search on Kaggle — no unique slug
    "human_faces": "kaustubhdhote/human-faces-dataset",
    "fake_140k":   "xhlulu/140k-real-and-fake-faces",
    "sfhq":        "selfishgene/synthetic-faces-high-quality-sfhq-part-1",
}


def _is_downloaded(output_dir: Path, check_subdir: str) -> bool:
    """Return True if the dataset's expected subfolder already exists."""
    return (output_dir / check_subdir).exists()


def _kaggle_download(slug: str, output_dir: Path) -> bool:
    """Download a Kaggle dataset by slug.

    Args:
        slug:       Kaggle dataset slug (user/dataset-name).
        output_dir: Output directory path.

    Returns:
        True on success, False on failure.
    """
    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", slug, "-p", str(output_dir), "--unzip",
    ]
    print(f"[Download] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[Download] ERROR: Kaggle download failed: {e}")
        print("[Download] Ensure Kaggle CLI is configured: kaggle.json in ~/.kaggle/")
        return False


def _roboflow_download(project_info: dict, output_dir: Path) -> bool:
    """Download from Roboflow using the Python SDK.

    Args:
        project_info: Dict with 'workspace', 'project', 'version', and 'dest'.
        output_dir:   Base output directory.

    Returns:
        True on success, False on failure.
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("[Download] ROBOFLOW_API_KEY not set in .env — skipping Roboflow datasets.")
        return False

    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        proj = (
            rf.workspace(project_info["workspace"])
            .project(project_info["project"])
        )
        proj.version(project_info["version"]).download(
            "yolov8",
            location=str(output_dir / project_info["dest"]),
        )
        return True
    except ImportError:
        print("[Download] Install roboflow: pip install roboflow")
        return False
    except Exception as e:
        print(f"[Download] Roboflow error: {e}")
        return False


def download_all(datasets: list, output_dir: str = "./data") -> dict:
    """Download all specified datasets.

    Args:
        datasets: List of dataset keys to download. Use ['all'] for everything.
        output_dir: Base output directory for all datasets.

    Returns:
        Dict mapping dataset name → 'ok' | 'skipped' | 'error' | 'manual'.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    keys = list(DATASET_INFO.keys()) if "all" in datasets else datasets
    results = {}

    for key in keys:
        if key not in DATASET_INFO:
            print(f"[Download] Unknown dataset: {key}. Run --list to see available.")
            continue

        info = DATASET_INFO[key]
        print(f"\n{'='*60}")
        print(f"[Download] {info['name']} ({info['access']})")

        # Skip if already exists
        if _is_downloaded(out, info["check"]):
            print(f"[Download] SKIP — already downloaded: {out / info['check']}")
            results[key] = "skipped"
            continue

        method = info["method"]

        if method == "kaggle":
            slug = KAGGLE_COMMANDS.get(key)
            if not slug:
                print(f"[Download] No Kaggle slug for '{key}'. See download_instructions.md")
                results[key] = "manual"
                continue
            ok = _kaggle_download(slug, out)
            results[key] = "ok" if ok else "error"

        elif method == "roboflow":
            # Map dataset key → Roboflow project info
            rf_info = {
                "mobile_person": {
                    "workspace": "shubham-vishwakarma-5olb4",
                    "project":   "mobile-person-datset",
                    "version":   1,
                    "dest":      "mobile_person",
                },
                "mobile_phone": {
                    "workspace": "datacluster-labs-agryi",
                    "project":   "mobile-phone-dataset",
                    "version":   1,
                    "dest":      "mobile_phone",
                },
            }.get(key, {})

            if not rf_info:
                results[key] = "manual"
                continue

            ok = _roboflow_download(rf_info, out)
            results[key] = "ok" if ok else "error"

        elif method == "manual":
            print(f"[Download] MANUAL — see download_instructions.md for '{key}'")
            results[key] = "manual"

    return results


def print_status_table(output_dir: str = "./data") -> None:
    """Print a table of all datasets with download status."""
    out = Path(output_dir)
    print(f"\n{'Dataset':<20} {'Type':<25} {'Access':<35} {'Status'}")
    print("-" * 100)
    for key, info in DATASET_INFO.items():
        downloaded = _is_downloaded(out, info["check"])
        status = "✓ Downloaded" if downloaded else "✗ Missing"
        print(f"{key:<20} {info['type']:<25} {info['access']:<35} {status}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Master dataset download script for the antispoof module",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets", type=str, default="all",
        help="Comma-separated dataset keys, or 'all'. "
             "Run --list to see all options.",
    )
    parser.add_argument("--output", type=str, default="./data")
    parser.add_argument("--list",   action="store_true",
                        help="Print dataset status table and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        print_status_table(args.output)
        sys.exit(0)

    datasets = [d.strip() for d in args.datasets.split(",")]
    print(f"[Download] Target datasets: {datasets}")
    print(f"[Download] Output directory: {args.output}")

    results = download_all(datasets, args.output)

    print(f"\n{'='*60}")
    print("[Download] Summary:")
    for ds, status in results.items():
        print(f"  {ds:<20}: {status}")

    print("\n[Download] Done. Manual datasets require authentication — see download_instructions.md")
