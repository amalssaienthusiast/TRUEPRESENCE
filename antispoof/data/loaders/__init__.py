"""data/loaders/__init__.py — expose all dataset loader classes."""

from .lcc_fasd import LCCFASDDataset
from .celeba_spoof import CelebASpoofDataset
from .human_faces import HumanFacesDataset
from .fake_140k import Fake140kDataset
from .sfhq import SFHQDataset
from .mobile_screen import MobileScreenDataset
from .combined import CombinedAntiSpoofDataset

__all__ = [
    "LCCFASDDataset",
    "CelebASpoofDataset",
    "HumanFacesDataset",
    "Fake140kDataset",
    "SFHQDataset",
    "MobileScreenDataset",
    "CombinedAntiSpoofDataset",
]
