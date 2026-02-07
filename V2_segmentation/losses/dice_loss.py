"""
V2_segmentation/losses/dice_loss.py
====================================
Uncertainty-aware Dice loss for multi-label segmentation.

Supports:
  - Per-channel Dice computation (macro average across 5 channels)
  - Uncertainty masking: low-confidence pixels excluded from loss
  - Pseudo-label tier weighting: Tier-A (high conf) weighted more than Tier-B
  - Smooth factor to prevent division by zero
"""

from __future__ import annotations

import torch
import torch.nn as nn

from V2_segmentation.config import (
    DICE_SMOOTH,
    UNCERTAINTY_THRESHOLD,
    TIER_A_THRESHOLD,
    TIER_A_LOSS_WEIGHT,
    TIER_B_LOSS_WEIGHT,
)


class DiceLoss(nn.Module):
    """Uncertainty-aware Dice loss for multi-label segmentation.

    Parameters
    ----------
    smooth : float
        Smoothing factor (Laplace smoothing).
    apply_sigmoid : bool
        If True, apply sigmoid to predictions before computing Dice.
    uncertainty_threshold : float
        Pixels with confidence below this are masked out (0 weight).
    tier_a_threshold : float
        Pixels above this confidence get full weight.
    tier_a_weight : float
        Loss weight for high-confidence (Tier-A) pixels.
    tier_b_weight : float
        Loss weight for medium-confidence (Tier-B) pixels.
    """

    def __init__(
        self,
        smooth: float = DICE_SMOOTH,
        apply_sigmoid: bool = True,
        uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
        tier_a_threshold: float = TIER_A_THRESHOLD,
        tier_a_weight: float = TIER_A_LOSS_WEIGHT,
        tier_b_weight: float = TIER_B_LOSS_WEIGHT,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
        self.uncertainty_threshold = uncertainty_threshold
        self.tier_a_threshold = tier_a_threshold
        self.tier_a_weight = tier_a_weight
        self.tier_b_weight = tier_b_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute uncertainty-weighted Dice loss.

        Parameters
        ----------
        pred : (B, C, H, W)
            Raw segmentation logits.
        target : (B, C, H, W)
            Ground-truth / pseudo-label masks (soft or hard).
        confidence : (B, 1, H, W) or (B, H, W), optional
            Per-pixel confidence map.  If None, all pixels weighted equally.

        Returns
        -------
        Scalar Dice loss (1 - Dice coefficient, averaged over channels).
        """
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        B, C, _H, _W = pred.shape

        # Build per-pixel weight map from confidence
        if confidence is not None:
            if confidence.dim() == 3:
                confidence = confidence.unsqueeze(1)  # (B, 1, H, W)
            # Tier assignment
            weight = torch.where(
                confidence >= self.tier_a_threshold,
                torch.full_like(confidence, self.tier_a_weight),
                torch.full_like(confidence, self.tier_b_weight),
            )
            # Mask out low-confidence pixels
            weight = torch.where(
                confidence >= self.uncertainty_threshold,
                weight,
                torch.zeros_like(weight),
            )
            weight = weight.expand_as(pred)  # (B, C, H, W)
        else:
            weight = torch.ones_like(pred)

        # Per-channel Dice
        pred_flat = pred.reshape(B, C, -1)       # (B, C, H*W)
        target_flat = target.reshape(B, C, -1)
        weight_flat = weight.reshape(B, C, -1)

        intersection = (pred_flat * target_flat * weight_flat).sum(dim=2)
        union = (pred_flat * weight_flat).sum(dim=2) + (target_flat * weight_flat).sum(dim=2)

        dice_per_channel = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_per_channel.mean()

        return dice_loss
