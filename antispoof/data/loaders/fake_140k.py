"""
data/loaders/fake_140k.py — 140k Real and Fake Faces Dataset Loader

70k real faces (FFHQ) + 70k StyleGAN2-generated. Perfectly balanced.
Strongest dataset for synthetic face detection training.

Kaggle: kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
CLI:    kaggle datasets download -d xhlulu/140k-real-and-fake-faces

Expected structure:
    <root>/real_vs_fake/real_vs_fake/{train,valid,test}/{real,fake}/
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class Fake140kDataset(Dataset):
    """140k Real and Fake Faces dataset loader.

    Args:
        root:      Path containing the real_vs_fake/ folder.
        split:     'train', 'val' (maps to 'valid'), or 'test'.
        transform: Optional transform pipeline.
        real_only: Return only real (FFHQ) samples.
        fake_only: Return only StyleGAN2-generated fake samples.
    """

    LABEL_MAP = {"real": 1, "fake": 0}
    SPLIT_MAP = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        real_only: bool = False,
        fake_only: bool = False,
    ) -> None:
        """Initialise 140k dataset."""
        self.root = Path(root) / "real_vs_fake" / "real_vs_fake"
        self.transform = transform or transforms.ToTensor()
        self.real_only = real_only
        self.fake_only = fake_only

        folder_split = self.SPLIT_MAP.get(split, split)
        split_root = self.root / folder_split

        if not split_root.exists():
            print(f"[Fake140kDataset] WARNING: {split_root} not found. "
                  "Run download_instructions() to get the data.")
            self.samples: List[Tuple[Path, int]] = []
            return

        self.samples = self._collect(split_root)
        n_real = sum(1 for _, l in self.samples if l == 1)
        n_fake = sum(1 for _, l in self.samples if l == 0)
        print(f"[Fake140kDataset] {split}: {len(self.samples)} "
              f"(real={n_real}, fake={n_fake})")

    def _collect(self, split_root: Path) -> List[Tuple[Path, int]]:
        """Collect (path, label) pairs from real/ and fake/ subdirs."""
        samples = []
        for class_name, label in self.LABEL_MAP.items():
            if self.real_only and label != 1:
                continue
            if self.fake_only and label != 0:
                continue
            class_dir = split_root / class_name
            if not class_dir.exists():
                print(f"[Fake140kDataset] WARNING: {class_dir} missing.")
                continue
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in self.IMAGE_EXTS:
                    samples.append((f, label))
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
            print(f"[Fake140kDataset] WARNING: {img_path}: {e}")
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        return self.transform(img), label

    @staticmethod
    def download_instructions() -> None:
        """Print Kaggle CLI download instructions."""
        print("""
╔══ 140k Real and Fake Faces Download ═══════════════════════════════╗
║                                                                     ║
║  kaggle datasets download -d xhlulu/140k-real-and-fake-faces \\    ║
║    -p ./data --unzip                                                ║
║                                                                     ║
║  Expected structure:                                                ║
║    data/real_vs_fake/real_vs_fake/train/{real,fake}/               ║
║    data/real_vs_fake/real_vs_fake/valid/{real,fake}/               ║
║    data/real_vs_fake/real_vs_fake/test/{real,fake}/                ║
║                                                                     ║
║  Size: ~70k real (FFHQ) + ~70k StyleGAN2-generated images         ║
╚═════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    Fake140kDataset.download_instructions()
    ds = Fake140kDataset(root="./data", split="train")
    print(f"Dataset length: {len(ds)}")
