"""
V2_segmentation/losses/joint_loss.py
=====================================
Joint loss combining segmentation (Dice + Focal) and classification (CE).

Loss = lambda_seg * (Dice + Focal) + lambda_cls * CrossEntropy

The lambdas are phase-dependent:
  - Phase A (seg head only):    lambda_seg=1.0, lambda_cls=0.0
  - Phase B (joint fine-tune):  lambda_seg=0.4, lambda_cls=0.6
  - Phase C (cls refinement):   lambda_seg=0.0, lambda_cls=1.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dice_loss import DiceLoss
from .focal_loss import FocalLoss


class JointLoss(nn.Module):
    """Phase-aware joint loss for dual-head training.

    Parameters
    ----------
    lambda_seg : float
        Weight for segmentation loss component.
    lambda_cls : float
        Weight for classification loss component.
    num_classes : int
        Number of classification classes (for CE label smoothing).
    label_smoothing : float
        Label smoothing for classification CE.
    dice_kwargs : dict
        Extra kwargs for DiceLoss.
    focal_kwargs : dict
        Extra kwargs for FocalLoss.
    """

    def __init__(
        self,
        lambda_seg: float = 0.4,
        lambda_cls: float = 0.6,
        num_classes: int = 13,
        label_smoothing: float = 0.1,
        dice_kwargs: dict | None = None,
        focal_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_cls = lambda_cls

        # Segmentation losses
        self.dice_loss = DiceLoss(**(dice_kwargs or {}))
        self.focal_loss = FocalLoss(**(focal_kwargs or {}))

        # Classification loss
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def set_phase(self, phase: str) -> None:
        """Update loss weights for a training phase.

        Parameters
        ----------
        phase : str
            ``"A"`` (seg only), ``"B"`` (joint), or ``"C"`` (cls only).
        """
        from V2_segmentation.config import PHASE_A, PHASE_B, PHASE_C
        phase_map = {"A": PHASE_A, "B": PHASE_B, "C": PHASE_C}
        cfg = phase_map.get(phase.upper())
        if cfg is None:
            raise ValueError(f"Unknown phase '{phase}'. Expected A/B/C.")
        self.lambda_seg = cfg["lambda_seg"]
        self.lambda_cls = cfg["lambda_cls"]

    def forward(
        self,
        cls_logits: torch.Tensor,
        seg_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        seg_targets: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the joint loss.

        Parameters
        ----------
        cls_logits : (B, num_classes)
            Classification logits.
        seg_logits : (B, C_seg, H, W)
            Segmentation logits.
        cls_targets : (B,)
            Classification labels (long).
        seg_targets : (B, C_seg, H, W)
            Segmentation masks.
        confidence : (B, 1, H, W) or None
            Per-pixel confidence map.

        Returns
        -------
        dict with keys:
            ``loss``      — total scalar loss
            ``loss_seg``  — segmentation component
            ``loss_cls``  — classification component
            ``loss_dice`` — Dice subcomponent
            ``loss_focal``— Focal subcomponent
        """
        loss_dict: dict[str, torch.Tensor] = {}
        device = cls_logits.device

        # ── segmentation ────────────────────────────────────────────────
        if self.lambda_seg > 0 and seg_targets is not None:
            l_dice = self.dice_loss(seg_logits, seg_targets, confidence)
            l_focal = self.focal_loss(seg_logits, seg_targets, confidence)
            loss_seg = l_dice + l_focal
            loss_dict["loss_dice"] = l_dice.detach()
            loss_dict["loss_focal"] = l_focal.detach()
        else:
            loss_seg = torch.tensor(0.0, device=device)
            loss_dict["loss_dice"] = torch.tensor(0.0, device=device)
            loss_dict["loss_focal"] = torch.tensor(0.0, device=device)

        # ── classification ──────────────────────────────────────────────
        if self.lambda_cls > 0 and cls_targets is not None:
            loss_cls = self.cls_loss(cls_logits, cls_targets)
        else:
            loss_cls = torch.tensor(0.0, device=device)

        # ── combine ─────────────────────────────────────────────────────
        total = self.lambda_seg * loss_seg + self.lambda_cls * loss_cls

        loss_dict["loss"] = total
        loss_dict["loss_seg"] = loss_seg.detach()
        loss_dict["loss_cls"] = loss_cls.detach()
        return loss_dict
