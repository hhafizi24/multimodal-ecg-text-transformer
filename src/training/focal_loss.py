"""
Implementation of class-weighted focal loss for imbalanced multiclass classification.

Provides optional per-class weighting and configurable reduction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        """
        Args:
            gamma:     Focusing parameter. 0 reduces to standard cross-entropy.
                       Typical values: 0.5-5. Start with 2.0.
            weight:    Per-class weights, same as nn.CrossEntropyLoss(weight=...).
            reduction: "mean" or "sum".
        """
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}")
        
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [batch, num_classes] raw unnormalized scores
            targets: [batch] integer class indices

        Returns:
            Scalar loss.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Probability assigned to the ground-truth class for each sample.
        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Down-weight well-classified examples while preserving the
        # original cross-entropy objective.
        focal_loss = (1.0 - p_t) ** self.gamma * ce_loss

        if self.weight is not None:
            # Apply class weights after computing the focal modulation so that
            # p_t reflects the model's true predicted probability.
            class_weights = self.weight.to(logits.device)
            focal_loss = focal_loss * class_weights[targets]

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss