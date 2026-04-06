"""
data/loaders/lcc_fasd.py — LCC FASD Face Anti-Spoofing Dataset Loader

LCC FASD (Light-Weight Face Anti-Spoofing) is a free, publicly available
dataset with real and fake face images. Label convention: 1=real, 0=spoof.

Structure expected:
    <root>/LCC_FASD_development/real/   (live faces)
    <root>/LCC_FASD_development/fake/   (spoof faces)
    <root>/LCC_FASD_evaluation/real/
    <root>/LCC_FASD_evaluation/fake/
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class LCCFASDDataset(Dataset):
    """PyTorch Dataset for the LCC FASD anti-spoofing dataset.

    Args:
        root:       Path to the directory containing LCC_FASD_development/
                    and LCC_FASD_evaluation/.
        split:      One of 'train' (development set) or 'val'/'test' (evaluation).
        transform:  Optional torchvision transform pipeline.
    """

    # Map folder names → integer labels
    LABEL_MAP = {"real": 1, "fake": 0}
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialise LCC FASD dataset."""
        self.root = Path(root)
        self.split = split
        self.transform = transform or transforms.ToTensor()

        # Map split name → sub-folder
        subfolder = "LCC_FASD_development" if split == "train" else "LCC_FASD_evaluation"
        split_root = self.root / subfolder

        if not split_root.exists():
            print(f"[LCCFASDDataset] WARNING: {split_root} not found. "
                  f"Run download_instructions() to get the data.")
            self.samples: list[Tuple[Path, int]] = []
            return

        self.samples = self._collect_samples(split_root)
        print(f"[LCCFASDDataset] {split}: {len(self.samples)} images "
              f"(real={sum(1 for _, l in self.samples if l == 1)}, "
              f"fake={sum(1 for _, l in self.samples if l == 0)})")

    def _collect_samples(self, root: Path) -> list[Tuple[Path, int]]:
        """Walk real/ and fake/ subdirectories and collect (path, label) pairs."""
        samples = []
        for class_name, label in self.LABEL_MAP.items():
            class_dir = root / class_name
            if not class_dir.exists():
                print(f"[LCCFASDDataset] WARNING: {class_dir} missing.")
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in self.IMAGE_EXTS:
                    samples.append((img_path, label))
        return samples

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Return (image_tensor, label) for the given index."""
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Return a blank image on corrupt file rather than crashing the loader
            print(f"[LCCFASDDataset] WARNING: cannot open {img_path}: {e}")
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        return self.transform(img), label

    @staticmethod
    def download_instructions() -> None:
        """Print exact steps to download the LCC FASD dataset."""
        print("""
╔══ LCC FASD Download Instructions ══════════════════════════════════╗
║                                                                     ║
║  1. Visit: https://github.com/kprokofi/light-weight-face-anti-spoofing║
║     OR search "LCC FASD face anti spoofing" on Kaggle.             ║
║                                                                     ║
║  2. Kaggle CLI download:                                            ║
║     kaggle datasets search "LCC FASD"                              ║
║     kaggle datasets download -d <dataset-slug> -p ./data           ║
║                                                                     ║
║  3. Expected structure after extraction:                            ║
║     data/LCC_FASD_development/real/  (live images)                 ║
║     data/LCC_FASD_development/fake/  (spoof images)                ║
║     data/LCC_FASD_evaluation/real/                                  ║
║     data/LCC_FASD_evaluation/fake/                                  ║
║                                                                     ║
║  No sign-up required — fully open dataset.                         ║
╚═════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    LCCFASDDataset.download_instructions()
    ds = LCCFASDDataset(root="./data", split="train")
    print(f"Dataset length: {len(ds)}")
    if len(ds):
        img, label = ds[0]
        print(f"Sample shape: {img.shape}, label: {label}")
