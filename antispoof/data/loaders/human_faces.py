"""
data/loaders/human_faces.py — Human Faces Dataset Loader (Kaustubh Dhote)

~5,000 real + ~4,630 AI-generated faces. Good class balance.
Used as the primary VALIDATION set — never for training.
Label: 1=real, 0=AI-generated(fake)

Kaggle: kaggle.com/datasets/kaustubhdhote/human-faces-dataset
CLI:    kaggle datasets download -d kaustubhdhote/human-faces-dataset

Expected structure:
    <root>/human_faces_dataset/
        Real_Faces/
        AI_Generated_Faces/
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class HumanFacesDataset(Dataset):
    """Dataset for real vs AI-generated faces (Kaustubh Dhote).

    Args:
        root:      Path containing the human_faces_dataset/ folder.
        split:     'train', 'val', or 'test' — this dataset is best used as val.
        transform: Optional transform pipeline.
        val_split: Fraction of data used for validation when split!='train'.
    """

    LABEL_MAP = {"Real_Faces": 1, "AI_Generated_Faces": 0}
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        root: str,
        split: str = "val",
        transform: Optional[Callable] = None,
        val_split: float = 0.15,
    ) -> None:
        """Initialise Human Faces dataset."""
        self.root = Path(root) / "human_faces_dataset"
        self.split = split
        self.transform = transform or transforms.ToTensor()

        if not self.root.exists():
            print(f"[HumanFacesDataset] WARNING: {self.root} not found. "
                  "Run download_instructions() to get the data.")
            self.samples: List[Tuple[Path, int]] = []
            return

        all_samples = self._collect_all()
        self.samples = self._apply_split(all_samples, split, val_split)
        n_real = sum(1 for _, l in self.samples if l == 1)
        n_fake = sum(1 for _, l in self.samples if l == 0)
        print(f"[HumanFacesDataset] {split}: {len(self.samples)} "
              f"(real={n_real}, ai_gen={n_fake})")

    def _collect_all(self) -> List[Tuple[Path, int]]:
        """Collect all (path, label) pairs from subdirectories."""
        samples = []
        for class_dir, label in self.LABEL_MAP.items():
            d = self.root / class_dir
            if not d.exists():
                print(f"[HumanFacesDataset] WARNING: {d} not found.")
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in self.IMAGE_EXTS:
                    samples.append((f, label))
        return samples

    def _apply_split(
        self,
        samples: List[Tuple[Path, int]],
        split: str,
        val_split: float,
    ) -> List[Tuple[Path, int]]:
        """Deterministically split samples by index."""
        import math
        n = len(samples)
        val_n = math.floor(n * val_split)
        if split == "train":
            return samples[val_n:]
        return samples[:val_n]   # val and test both use the held-out portion

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Return (image_tensor, label)."""
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[HumanFacesDataset] WARNING: {img_path}: {e}")
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        return self.transform(img), label

    @staticmethod
    def download_instructions() -> None:
        """Print Kaggle download instructions."""
        print("""
╔══ Human Faces Dataset Download ════════════════════════════════════╗
║                                                                     ║
║  kaggle datasets download -d kaustubhdhote/human-faces-dataset \\  ║
║    -p ./data --unzip                                                ║
║                                                                     ║
║  Expected structure:                                                ║
║    data/human_faces_dataset/Real_Faces/                            ║
║    data/human_faces_dataset/AI_Generated_Faces/                    ║
║                                                                     ║
║  NOTE: Use this dataset as VALIDATION ONLY — never train on it.   ║
╚═════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    HumanFacesDataset.download_instructions()
    ds = HumanFacesDataset(root="./data", split="val")
    print(f"Dataset length: {len(ds)}")
