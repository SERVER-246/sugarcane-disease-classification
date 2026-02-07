"""
V2_segmentation/losses/distillation_loss.py
============================================
Knowledge distillation loss for segmentation.

Used in pseudo-label refinement: a teacher ensemble's soft predictions
guide a student model via KL-divergence on segmentation logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegDistillationLoss(nn.Module):
    """Per-pixel knowledge distillation for segmentation masks.

    Computes KL-divergence between teacher (soft) and student (soft)
    segmentation predictions at temperature T.

    Since our segmentation is multi-label (sigmoid, not softmax), we use
    per-channel binary KL:

        KL = t * log(t / s) + (1-t) * log((1-t) / (1-s))

    averaged over channels and pixels.

    Parameters
    ----------
    temperature : float
        Softening temperature (applied to logits before sigmoid).
    alpha : float
        Weight of distillation loss vs hard-label loss (0-1).
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_targets: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute distillation loss.

        Parameters
        ----------
        student_logits : (B, C, H, W)
            Student model's raw segmentation logits.
        teacher_logits : (B, C, H, W)
            Teacher model's raw segmentation logits.
        hard_targets : (B, C, H, W) or None
            Hard pseudo-labels for the non-distillation component.
        confidence : (B, 1, H, W) or None
            Per-pixel confidence mask.

        Returns
        -------
        Scalar distillation loss.
        """
        T = self.temperature

        # Soften with temperature
        student_soft = torch.sigmoid(student_logits / T)
        teacher_soft = torch.sigmoid(teacher_logits / T).detach()

        # Per-channel binary KL divergence
        eps = 1e-7
        teacher_soft = teacher_soft.clamp(eps, 1 - eps)
        student_soft = student_soft.clamp(eps, 1 - eps)

        kl = (
            teacher_soft * torch.log(teacher_soft / student_soft)
            + (1 - teacher_soft) * torch.log((1 - teacher_soft) / (1 - student_soft))
        )

        # Apply confidence mask
        if confidence is not None:
            if confidence.dim() == 3:
                confidence = confidence.unsqueeze(1)
            from V2_segmentation.config import UNCERTAINTY_THRESHOLD
            mask = (confidence >= UNCERTAINTY_THRESHOLD).float()
            kl = kl * mask.expand_as(kl)

        distill_loss = kl.mean() * (T ** 2)

        # Combine with hard-label loss if provided
        if hard_targets is not None and self.alpha < 1.0:
            hard_loss = F.binary_cross_entropy_with_logits(
                student_logits, hard_targets, reduction="mean"
            )
            return self.alpha * distill_loss + (1 - self.alpha) * hard_loss

        return distill_loss
