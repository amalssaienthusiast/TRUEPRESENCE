"""
models/antispoof_net.py — MobileNetV3-Small Anti-Spoofing Classifier (Stage 3)

Architecture: MobileNetV3-Small backbone (ImageNet pretrained)
              + custom classification head per Section 4 spec.

Custom head:
  Dropout(0.35) → Linear(→512) → Hardswish → BatchNorm1d → Dropout(0.25)
  → Linear(512→256) → Hardswish → Dropout(0.15) → Linear(256→2)
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# torchvision is loaded lazily inside __init__ so the module
# can be imported (e.g. for --help) without torchvision installed.


class AntiSpoofNet(nn.Module):
    """MobileNetV3-Small backbone with a custom anti-spoofing classification head.

    Output: 2 logits → [spoof=0, live=1]
    ~2.5M parameters — suitable for CPU / edge deployment.

    Args:
        pretrained:   Load ImageNet-pretrained backbone weights.
        num_classes:  Number of output classes (default: 2).
        freeze_backbone: If True, freeze all backbone layers for fast fine-tuning.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialise AntiSpoofNet."""
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        # Lazy-import torchvision to keep the module importable without it.
        try:
            import torchvision.models as tv_models
        except ImportError as e:
            raise ImportError(
                "torchvision is required to instantiate AntiSpoofNet. "
                "Install it with: pip install torchvision"
            ) from e

        weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.mobilenet_v3_small(weights=weights)

        # Remove the original classifier block (last Linear layer)
        # backbone.classifier = [Dropout, Linear(576→1000)]
        in_features = backbone.classifier[0].in_features   # 576

        # Keep only the backbone features (removes the classifier)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.pre_head = backbone.classifier[0]   # AdaptiveAvgPool already applied

        if freeze_backbone:
            # Freeze all backbone parameters — only train the head
            for p in self.features.parameters():
                p.requires_grad = False

        # ── Custom Classification Head (per Section 4 spec) ───────────────
        self.head = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.25),
            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.15),
            nn.Linear(256, num_classes),
        )

        # ── Weight initialisation for custom head ─────────────────────────
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, 3, 224, 224).

        Returns:
            Logits tensor of shape (N, num_classes).
        """
        x = self.features(x)     # (N, 576, 7, 7)
        x = self.avgpool(x)      # (N, 576, 1, 1)
        x = x.flatten(1)         # (N, 576)
        # pre_head is the first Dropout from the original backbone classifier
        x = self.pre_head(x)     # (N, 576) — identity dropout here
        return self.head(x)      # (N, num_classes)

    def get_embedding(self, x: Tensor) -> Tensor:
        """Return the 256-d embedding (before final Linear) for contrastive loss.

        Args:
            x: Input tensor of shape (N, 3, 224, 224).

        Returns:
            L2-normalised embedding of shape (N, 256).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.pre_head(x)
        # Run head up to (but not including) the final Linear layer
        for layer in self.head[:-1]:
            x = layer(x)
        import torch.nn.functional as F
        return F.normalize(x, dim=1)

    def export_onnx(
        self,
        output_path: str,
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
        opset: int = 17,
    ) -> str:
        """Export model to ONNX format.

        Args:
            output_path: Destination .onnx file path.
            input_size:  Dummy input shape (N, C, H, W).
            opset:       ONNX opset version.

        Returns:
            Path to the exported ONNX file.
        """
        import torch.onnx
        self.eval()
        dummy = torch.zeros(*input_size)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            self,
            dummy,
            output_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        print(f"[AntiSpoofNet] ONNX exported → {output_path}")
        return output_path

    def count_params(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AntiSpoofNet(pretrained=False)
    print(f"Parameters: {model.count_params():,}")

    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    print(f"Output shape: {logits.shape}")    # (4, 2)

    emb = model.get_embedding(x)
    print(f"Embedding shape: {emb.shape}")    # (4, 256)

    # Test ONNX export
    model.export_onnx("/tmp/antispoof_test.onnx")
    print("ONNX export OK")
