"""
data/loaders/mobile_screen.py — Mobile/Screen Object Detection Dataset Loader

Loads datasets for YOLOv8 phone/screen detection (Stage 2).
Wraps Roboflow-format YOLOv8 datasets — returns image paths and YOLO label paths
for use with the Ultralytics training API.

Datasets covered:
  C1. Mobile Person Dataset (Roboflow): detects person_on_screen, mobile_phone
  C2. Mobile Phone Dataset (DataCluster Labs): phone bounding boxes

Expected structure (after Roboflow download):
    <root>/mobile_person/
        train/images/   train/labels/
        valid/images/   valid/labels/
        data.yaml
"""

import os
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import yaml
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from torch import Tensor


class MobileScreenDataset(Dataset):
    """Loader for Roboflow YOLOv8-format mobile/screen detection datasets.

    This loader serves TWO purposes:
      1. For Ultralytics YOLO training: returns image_path strings via a
         path_list() method (YOLO reads them itself).
      2. For standalone inspection: returns (image_tensor, label_path) pairs.

    Args:
        root:      Path containing the mobile_person/ or mobile_phone/ folder.
        dataset:   'mobile_person' or 'mobile_phone'.
        split:     'train', 'val', or 'test'.
        transform: Optional image transform.
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        root: str,
        dataset: str = "mobile_person",
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialise mobile screen dataset."""
        self.root = Path(root) / dataset
        self.split = split
        self.transform = transform or transforms.ToTensor()

        # Roboflow uses 'valid' not 'val'
        folder_split = "valid" if split == "val" else split
        self.img_dir = self.root / folder_split / "images"
        self.lbl_dir = self.root / folder_split / "labels"

        if not self.img_dir.exists():
            print(f"[MobileScreenDataset] WARNING: {self.img_dir} not found. "
                  "Run download_instructions() to fetch via Roboflow.")
            self.samples: List[Tuple[Path, Path]] = []
            return

        self.samples = [
            (img, self.lbl_dir / (img.stem + ".txt"))
            for img in sorted(self.img_dir.iterdir())
            if img.suffix.lower() in self.IMAGE_EXTS
        ]
        print(f"[MobileScreenDataset] {dataset}/{split}: {len(self.samples)} images")

    def path_list(self) -> List[str]:
        """Return list of image path strings for Ultralytics YOLO training."""
        return [str(p) for p, _ in self.samples]

    def generate_yaml(self, output_path: str = "data/mobile_screen.yaml") -> str:
        """Generate a data.yaml file for Ultralytics YOLO training.

        Returns the path to the generated YAML file.
        """
        data = {
            "path": str(self.root.parent),
            "train": str(self.root / "train"  / "images"),
            "val":   str(self.root / "valid"  / "images"),
            "test":  str(self.root / "test"   / "images"),
            "nc": 3,
            "names": {0: "mobile_phone", 1: "person_on_screen", 2: "tablet_screen"},
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"[MobileScreenDataset] data.yaml written → {out}")
        return str(out)

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        """Return (image_tensor, label_path_str) for inspection."""
        img_path, lbl_path = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[MobileScreenDataset] WARNING: {img_path}: {e}")
            img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        return self.transform(img), str(lbl_path)

    @staticmethod
    def download_instructions() -> None:
        """Print Roboflow API download instructions."""
        print("""
╔══ Mobile Screen Dataset Download (Roboflow) ════════════════════════╗
║                                                                      ║
║  1. Set ROBOFLOW_API_KEY in your .env file                          ║
║  2. pip install roboflow                                             ║
║  3. Run in Python:                                                   ║
║                                                                      ║
║     from roboflow import Roboflow                                    ║
║     import os                                                        ║
║     rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])           ║
║                                                                      ║
║     # C1: Mobile Person Dataset                                      ║
║     proj = rf.workspace('shubham-vishwakarma-5olb4')\\               ║
║              .project('mobile-person-datset')                        ║
║     proj.version(1).download('yolov8', location='./data/mobile_person')║
║                                                                      ║
║     # C2: Mobile Phone Dataset                                       ║
║     proj = rf.workspace('datacluster-labs-agryi')\\                  ║
║              .project('mobile-phone-dataset')                        ║
║     proj.version(1).download('yolov8', location='./data/mobile_phone')║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    @staticmethod
    def roboflow_download(api_key: str, output_dir: str = "./data") -> None:
        """Programmatic download of Roboflow datasets using the SDK."""
        try:
            from roboflow import Roboflow
        except ImportError:
            print("[MobileScreenDataset] Install roboflow: pip install roboflow")
            return

        rf = Roboflow(api_key=api_key)

        # C1 — Mobile Person Dataset
        print("[MobileScreenDataset] Downloading Mobile Person Dataset (C1)...")
        rf.workspace("shubham-vishwakarma-5olb4").project(
            "mobile-person-datset"
        ).version(1).download("yolov8", location=os.path.join(output_dir, "mobile_person"))

        # C2 — Mobile Phone Dataset
        print("[MobileScreenDataset] Downloading Mobile Phone Dataset (C2)...")
        rf.workspace("datacluster-labs-agryi").project(
            "mobile-phone-dataset"
        ).version(1).download("yolov8", location=os.path.join(output_dir, "mobile_phone"))


if __name__ == "__main__":
    MobileScreenDataset.download_instructions()
    ds = MobileScreenDataset(root="./data", dataset="mobile_person", split="train")
    print(f"Dataset length: {len(ds)}")
    yaml_path = ds.generate_yaml()
    print(f"YAML: {yaml_path}")
