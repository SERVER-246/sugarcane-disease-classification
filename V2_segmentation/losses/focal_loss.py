"""
V2_segmentation/losses/focal_loss.py
=====================================
Focal loss for multi-label segmentation (sigmoid-based).

Addresses class imbalance by down-weighting well-classified pixels:
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

Unlike standard focal loss which is for single-label, this version works
with multi-label (each channel independently via BCE).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from V2_segmentation.config import FOCAL_ALPHA, FOCAL_GAMMA


class FocalLoss(nn.Module):
    """Focal loss for multi-label segmentation.

    Parameters
    ----------
    alpha : float
        Weighting factor for the positive class.
    gamma : float
        Focusing parameter (higher = more focus on hard examples).
    reduction : str
        ``"mean"`` or ``"sum"`` or ``"none"``.
    """

    def __init__(
        self,
        alpha: float = FOCAL_ALPHA,
        gamma: float = FOCAL_GAMMA,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        pred : (B, C, H, W)
            Raw logits (before sigmoid).
        target : (B, C, H, W)
            Ground-truth masks (0/1 or soft labels).
        confidence : (B, 1, H, W) or None
            Per-pixel confidence for masking (optional).

        Returns
        -------
        Scalar focal loss.
        """
        # BCE with logits (numerically stable)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # p_t = probability of the correct class
        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)

        # Focal modulating factor
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = alpha_t * focal_weight * bce  # (B, C, H, W)

        # Apply confidence mask if provided
        if confidence is not None:
            if confidence.dim() == 3:
                confidence = confidence.unsqueeze(1)
            from V2_segmentation.config import UNCERTAINTY_THRESHOLD
            mask = (confidence >= UNCERTAINTY_THRESHOLD).float()
            loss = loss * mask.expand_as(loss)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
