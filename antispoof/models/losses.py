"""
models/losses.py — Loss functions for anti-spoofing training.

Implements:
  - FocalLoss:         Addresses class imbalance by down-weighting easy examples.
  - SupConLoss:        Supervised Contrastive Loss for representation learning.
  - CombinedAntiSpoofLoss: 0.8 * FocalLoss + 0.2 * LabelSmoothing CE (Stage 3 spec).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Focal Loss for binary and multi-class classification.

    Proposed in: "Focal Loss for Dense Object Detection" (Lin et al., 2017).
    Down-weights easy examples so the model focuses on hard, misclassified ones.
    Critical for anti-spoofing with class imbalance.

    Args:
        alpha:    Weighting factor for rare class. 0.25 gives more weight to
                  the positive (real face) class.
        gamma:    Focusing parameter. gamma=0 == standard cross entropy.
                  gamma=2 is the standard recommendation.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Initialise FocalLoss."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            inputs:  Raw logits of shape (N, C).
            targets: Integer class labels of shape (N,).

        Returns:
            Scalar loss value (or per-sample if reduction='none').
        """
        # Standard cross-entropy gives log-probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        # p_t = probability of the true class
        p_t = torch.exp(-ce_loss)
        # Focal term: (1 - p_t)^gamma reduces weight for easy examples
        focal_term = (1.0 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_term * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    From: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020).
    Pulls embeddings of same class together, pushes different classes apart.
    Improves representation quality especially with limited data.

    Args:
        temperature: Scaling factor for dot products. Typically 0.07–0.1.
        base_temperature: Reference temperature (from original paper = 0.07).
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07) -> None:
        """Initialise SupConLoss."""
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        """Compute supervised contrastive loss.

        Args:
            features: L2-normalised embeddings of shape (N, d).
            labels:   Class labels of shape (N,).

        Returns:
            Scalar contrastive loss.
        """
        device = features.device
        batch_size = features.shape[0]

        # Build positive mask: mask[i, j] = 1 if i and j share the same label
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Dot product similarity matrix, scaled by temperature
        similarity = torch.matmul(features, features.T) / self.temperature

        # For numerical stability, subtract the max per row
        sim_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - sim_max.detach()

        # Exclude diagonal (self-similarity)
        logits_mask = 1 - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Log-sum-exp over all valid negatives and positives
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

        # Mean of log-prob over positives
        pos_per_row = mask.sum(dim=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / (pos_per_row + 1e-9)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        return loss.mean()


class CombinedAntiSpoofLoss(nn.Module):
    """Combined loss for Stage 3 anti-spoof classifier (per Section 4 spec).

    total_loss = 0.8 * FocalLoss(alpha=0.25, gamma=2.0)
               + 0.2 * CrossEntropy with LabelSmoothing(0.1)

    Args:
        focal_alpha:      Alpha parameter for FocalLoss.
        focal_gamma:      Gamma parameter for FocalLoss.
        label_smoothing:  Label smoothing factor for CE component.
        focal_weight:     Weight for focal loss component (default: 0.8).
        ce_weight:        Weight for cross-entropy component (default: 0.2).
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        focal_weight: float = 0.8,
        ce_weight: float = 0.2,
    ) -> None:
        """Initialise combined loss."""
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute combined focal + label-smoothed CE loss.

        Args:
            logits:  Raw model outputs of shape (N, num_classes).
            targets: Integer class labels of shape (N,).

        Returns:
            Weighted scalar loss.
        """
        return (
            self.focal_weight * self.focal(logits, targets)
            + self.ce_weight   * self.ce(logits, targets)
        )


if __name__ == "__main__":
    # Quick smoke test
    B, C = 8, 2
    logits  = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))

    fl = FocalLoss()(logits, targets)
    print(f"FocalLoss: {fl.item():.4f}")

    combined = CombinedAntiSpoofLoss()(logits, targets)
    print(f"CombinedLoss: {combined.item():.4f}")

    feats = F.normalize(torch.randn(B, 128), dim=1)
    con   = SupConLoss()(feats, targets)
    print(f"SupConLoss: {con.item():.4f}")
