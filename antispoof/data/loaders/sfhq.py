"""
data/loaders/sfhq.py — SFHQ Synthetic Faces High Quality Dataset Loader

~425,000 AI-generated faces: StyleGAN3, DALL-E 2, Midjourney.
Used as FAKE-ONLY augmentation — load Part 1 initially (~50k images).
Mix with real FFHQ faces for balanced training.

Kaggle: kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-1
GitHub: github.com/SelfishGene/SFHQ-dataset

Expected structure:
    <root>/sfhq_part1/   (flat directory of .jpg images)
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class SFHQDataset(Dataset):
    """Synthetic Faces High Quality (SFHQ) dataset — fake faces only (label=0).

    Args:
        root:       Path containing sfhq_part1/ folder (or sfhq_part2/, etc.).
        split:      'train', 'val', or 'test' (deterministic split by index).
        part:       Dataset part number (1, 2, or 3). Default: 1.
        transform:  Optional transform pipeline.
        max_samples: Limit number of images loaded (useful for fast iteration).
        val_fraction: Fraction for val/test split.
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
    # SFHQ images are all fake — label is always 0
    LABEL = 0

    def __init__(
        self,
        root: str,
        split: str = "train",
        part: int = 1,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        val_fraction: float = 0.1,
    ) -> None:
        """Initialise SFHQ dataset."""
        self.root = Path(root) / f"sfhq_part{part}"
        self.split = split
        self.transform = transform or transforms.ToTensor()

        if not self.root.exists():
            print(f"[SFHQDataset] WARNING: {self.root} not found. "
                  "Run download_instructions() to get the data.")
            self.samples: List[Path] = []
            return

        all_imgs = sorted(
            p for p in self.root.iterdir()
            if p.suffix.lower() in self.IMAGE_EXTS
        )

        if max_samples:
            all_imgs = all_imgs[:max_samples]

        self.samples = self._split(all_imgs, split, val_fraction)
        print(f"[SFHQDataset] Part {part} / {split}: {len(self.samples)} fake images")

    def _split(self, imgs: list, split: str, val_frac: float) -> list:
        """Deterministic split: last val_frac for val/test, rest for train."""
        import math
        n_val = math.floor(len(imgs) * val_frac)
        if split == "train":
            return imgs[n_val:]
        return imgs[:n_val]

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Return (image_tensor, 0) — all SFHQ images are fake."""
        img_path = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[SFHQDataset] WARNING: {img_path}: {e}")
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        return self.transform(img), self.LABEL

    @staticmethod
    def download_instructions() -> None:
        """Print Kaggle CLI download steps for all SFHQ parts."""
        print("""
╔══ SFHQ Synthetic Faces Download ═══════════════════════════════════╗
║                                                                     ║
║  Part 1 (~50k images — start here):                                ║
║  kaggle datasets download \\                                         ║
║    -d selfishgene/synthetic-faces-high-quality-sfhq-part-1 \\      ║
║    -p ./data/sfhq_part1 --unzip                                    ║
║                                                                     ║
║  Part 2 (optional, ~175k):                                         ║
║  kaggle datasets download \\                                         ║
║    -d selfishgene/synthetic-faces-high-quality-sfhq-part-2 \\      ║
║    -p ./data/sfhq_part2 --unzip                                    ║
║                                                                     ║
║  GitHub reference: github.com/SelfishGene/SFHQ-dataset             ║
║  IMPORTANT: This dataset is FAKE-only. Use as augmentation only.  ║
╚═════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    SFHQDataset.download_instructions()
    ds = SFHQDataset(root="./data", split="train", max_samples=1000)
    print(f"Dataset length: {len(ds)}")
