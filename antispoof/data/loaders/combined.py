"""
data/loaders/combined.py — Multi-Dataset Combiner with WeightedRandomSampler

Assembles the Stage 3 classifier training split per the dataset strategy:

  REAL pool → CelebA-Spoof (live) + 140k FFHQ real
  FAKE pool → LCC FASD + CelebA-Spoof (spoof) + 140k fake + SFHQ Part 1
  Validation → Human Faces Dataset (held out entirely from training)
  Test       → Hard Dataset (never seen during any training)

Enforces 1:1 real:fake ratio per batch via WeightedRandomSampler.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler
from torchvision import transforms

from .lcc_fasd import LCCFASDDataset
from .celeba_spoof import CelebASpoofDataset
from .human_faces import HumanFacesDataset
from .fake_140k import Fake140kDataset
from .sfhq import SFHQDataset


# ---------------------------------------------------------------------------
# Default transforms
# ---------------------------------------------------------------------------

def train_transform(input_size: int = 224, resize_size: int = 256) -> transforms.Compose:
    """Build the training augmentation pipeline per Section 4 of the spec."""
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(0.05),    # handles B&W replay attacks
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),     # robustness to partial occlusion
    ])


def val_transform(input_size: int = 224, resize_size: int = 256) -> transforms.Compose:
    """Build the validation/test transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------

class CombinedAntiSpoofDataset(Dataset):
    """Multi-source anti-spoofing dataset combining all available sources.

    Args:
        root:          Base data directory.
        split:         'train', 'val', or 'test'.
        transform:     Optional transform (defaults depend on split).
        max_sfhq:      Cap SFHQ samples to avoid memory pressure.
        use_celeba:    Include CelebA-Spoof (requires auth).
        use_sfhq:      Include SFHQ fake augmentation.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_sfhq: int = 50_000,
        use_celeba: bool = False,     # requires auth — opt-in
        use_sfhq: bool = True,
    ) -> None:
        """Build the combined dataset."""
        self.split = split
        t = transform or (train_transform() if split == "train" else val_transform())

        # Validation: Human Faces Dataset (held out from all training)
        if split in ("val", "test"):
            self._dataset = HumanFacesDataset(root=root, split=split, transform=t)
            self.labels = [s[1] for s in self._dataset.samples]
            print(f"[CombinedDataset] {split}: {len(self)} samples (Human Faces only)")
            return

        # Training: assemble from multiple sources
        sub_datasets: List[Dataset] = []

        # ── REAL pool ─────────────────────────────────────────────────────
        # Real from 140k FFHQ
        real_140k = Fake140kDataset(root=root, split="train", transform=t, real_only=True)
        sub_datasets.append(real_140k)

        # Real from CelebA-Spoof (opt-in; requires auth)
        if use_celeba:
            cel_live = CelebASpoofDataset(root=root, split="train", transform=t, live_only=True)
            if len(cel_live):
                sub_datasets.append(cel_live)

        # ── FAKE pool ────────────────────────────────────────────────────
        # LCC FASD fake
        lcc_fake = LCCFASDDataset(root=root, split="train", transform=t)
        if len(lcc_fake):
            sub_datasets.append(lcc_fake)

        # 140k fake (StyleGAN2)
        fake_140k = Fake140kDataset(root=root, split="train", transform=t, fake_only=True)
        if len(fake_140k):
            sub_datasets.append(fake_140k)

        # CelebA-Spoof spoof (opt-in)
        if use_celeba:
            cel_spoof = CelebASpoofDataset(root=root, split="train", transform=t, spoof_only=True)
            if len(cel_spoof):
                sub_datasets.append(cel_spoof)

        # SFHQ synthetic fake (cap at max_sfhq for memory)
        if use_sfhq:
            sfhq = SFHQDataset(root=root, split="train", max_samples=max_sfhq, transform=t)
            if len(sfhq):
                sub_datasets.append(sfhq)

        # ── Concatenate ───────────────────────────────────────────────────
        if not sub_datasets:
            print("[CombinedDataset] WARNING: no datasets found! Check data root.")
            self._dataset = Fake140kDataset(root=root, split="train", transform=t)
        else:
            self._dataset = ConcatDataset(sub_datasets)

        # Build label list for sampler (needed for WeightedRandomSampler)
        self.labels = self._collect_labels(sub_datasets)
        n_real = sum(1 for l in self.labels if l == 1)
        n_fake = sum(1 for l in self.labels if l == 0)
        print(f"[CombinedDataset] train: {len(self.labels)} total "
              f"(real={n_real}, fake={n_fake})")

    def _collect_labels(self, datasets: List[Dataset]) -> List[int]:
        """Collect all labels from all sub-datasets for the sampler."""
        labels = []
        for ds in datasets:
            if hasattr(ds, "samples"):
                labels.extend(s[1] for s in ds.samples)
            elif hasattr(ds, "LABEL"):
                # SFHQ — all fake
                labels.extend([ds.LABEL] * len(ds))
        return labels

    def make_sampler(self) -> WeightedRandomSampler:
        """Create a WeightedRandomSampler that enforces 1:1 real:fake ratio."""
        n = len(self.labels)
        class_counts = [self.labels.count(c) for c in (0, 1)]
        # Weight for each sample = 1 / count_of_its_class
        weights = [1.0 / class_counts[lbl] for lbl in self.labels]
        return WeightedRandomSampler(weights, num_samples=n, replacement=True)

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Return (image_tensor, label)."""
        return self._dataset[idx]


if __name__ == "__main__":
    ds = CombinedAntiSpoofDataset(root="./data", split="train")
    print(f"Total samples: {len(ds)}")
    sampler = ds.make_sampler()
    print(f"Sampler type: {type(sampler).__name__}")
