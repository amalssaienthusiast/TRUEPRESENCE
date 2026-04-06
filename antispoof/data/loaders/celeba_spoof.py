"""
data/loaders/celeba_spoof.py — CelebA-Spoof Dataset Loader

CelebA-Spoof is the largest face anti-spoofing dataset:
  - 625,537 images, 10,177 subjects
  - Spoof types: print, replay, 3D mask, partial attack, paper cut-out
  - Labels from metas/intra_test/train_label.json, attr[43]: 1=live, 0=spoof

Access requires a GitHub request to the authors. Fully implemented here;
will print download instructions if files not found.

Expected structure:
    <root>/CelebA_Spoof/
        data/{000000..}/
        metas/intra_test/train_label.json
        metas/intra_test/test_label.json
"""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class CelebASpoofDataset(Dataset):
    """PyTorch Dataset for CelebA-Spoof.

    Parses the label JSON from metas/ and builds a flat sample list.
    Label convention: 1=live (bonafide), 0=spoof.

    Args:
        root:   Path containing the CelebA_Spoof/ folder.
        split:  'train' or 'test'.
        transform: Optional transform pipeline.
        live_only: If True, return ONLY live (bonafide) samples.
                   Useful when assembling the combined dataset real-face pool.
        spoof_only: If True, return ONLY spoof samples.
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        live_only: bool = False,
        spoof_only: bool = False,
    ) -> None:
        """Initialise CelebA-Spoof dataset."""
        self.root = Path(root) / "CelebA_Spoof"
        self.split = split
        self.transform = transform or transforms.ToTensor()
        self.live_only = live_only
        self.spoof_only = spoof_only

        label_file = (
            self.root / "metas" / "intra_test" /
            ("train_label.json" if split == "train" else "test_label.json")
        )

        if not label_file.exists():
            print(f"[CelebASpoofDataset] WARNING: {label_file} not found. "
                  "Run download_instructions() for access steps.")
            self.samples: List[Tuple[Path, int]] = []
            return

        self.samples = self._load_samples(label_file)
        n_live = sum(1 for _, l in self.samples if l == 1)
        n_spoof = sum(1 for _, l in self.samples if l == 0)
        print(f"[CelebASpoofDataset] {split}: {len(self.samples)} images "
              f"(live={n_live}, spoof={n_spoof})")

    def _load_samples(self, label_file: Path) -> List[Tuple[Path, int]]:
        """Parse label JSON and build (image_path, label) list."""
        with open(label_file) as f:
            label_map: Dict[str, List] = json.load(f)

        samples = []
        for rel_path, attrs in label_map.items():
            # attr[43] is the liveness label: 1=live, 0=spoof
            label = int(attrs[43]) if len(attrs) > 43 else 0
            if self.live_only and label != 1:
                continue
            if self.spoof_only and label != 0:
                continue
            img_path = self.root / "data" / rel_path
            if img_path.suffix.lower() in self.IMAGE_EXTS:
                samples.append((img_path, label))
        return samples

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Return (image_tensor, label)."""
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[CelebASpoofDataset] WARNING: cannot open {img_path}: {e}")
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        return self.transform(img), label

    @staticmethod
    def download_instructions() -> None:
        """Print exact steps to request and download CelebA-Spoof."""
        print("""
╔══ CelebA-Spoof Download Instructions ══════════════════════════════╗
║                                                                     ║
║  CelebA-Spoof requires a request to the authors:                   ║
║                                                                     ║
║  1. Visit: https://github.com/ZhangYuanhan-AI/CelebA-Spoof         ║
║  2. Fill in the Google Form linked under "Dataset Download"        ║
║  3. Receive a download link via email                               ║
║                                                                     ║
║  Expected structure after extraction:                               ║
║    data/CelebA_Spoof/data/000000/....jpg                           ║
║    data/CelebA_Spoof/metas/intra_test/train_label.json             ║
║    data/CelebA_Spoof/metas/intra_test/test_label.json              ║
║                                                                     ║
║  Size: ~625,537 images — allocate ~50 GB disk space.               ║
╚═════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    CelebASpoofDataset.download_instructions()
    ds = CelebASpoofDataset(root="./data", split="train")
    print(f"Dataset length: {len(ds)}")
