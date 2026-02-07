"""
V2_segmentation/models/dual_head.py
====================================
Dual-head model:  V1 backbone  +  segmentation decoder  +  classification head.

The backbone's weights come from V1 (pre-trained on disease classification).
Two output branches are trained:
  - **Segmentation head** — DeepLabV3+ decoder producing 5-channel pixel masks
  - **Classification head** — original backbone classifier (re-used from V1)

Training phases (A/B/C) freeze/unfreeze different components;  this module only
defines the architecture — the training loop is in ``training/``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from .backbone_adapter import BackboneFeatureExtractor, BACKBONE_HOOK_SPEC
from .decoder import DeepLabV3PlusDecoder

logger = logging.getLogger(__name__)


class DualHeadModel(nn.Module):
    """Dual-head model wrapping a V1 backbone with segmentation decoder.

    Parameters
    ----------
    backbone : nn.Module
        Pre-trained V1 backbone (e.g. ``CustomEfficientNetV4``).
    backbone_name : str
        Key into ``BACKBONE_HOOK_SPEC``.
    num_classes : int
        Number of disease classes (13).
    num_seg_channels : int
        Number of segmentation output channels (5).
    decoder_channels : int
        Internal width of the decoder (tier-adaptive).
    skip_channels : int
        Reduced skip-connection channels (48 or 32).
    aspp_rates : tuple[int, ...]
        ASPP dilation rates.
    img_size : int
        Input image resolution (224).
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_name: str,
        num_classes: int = 13,
        num_seg_channels: int = 5,
        decoder_channels: int = 256,
        skip_channels: int = 48,
        aspp_rates: tuple[int, ...] = (6, 12, 18),
        img_size: int = 224,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.num_seg_channels = num_seg_channels
        self.img_size = img_size

        # ── backbone (V1 pre-trained) ──────────────────────────────────
        self.backbone = backbone

        # ── hook-based feature extractor ───────────────────────────────
        self._extractor = BackboneFeatureExtractor(backbone, backbone_name)
        spec = BACKBONE_HOOK_SPEC[backbone_name]

        # ── segmentation decoder ───────────────────────────────────────
        self.seg_decoder = DeepLabV3PlusDecoder(
            high_level_channels=spec.high_level.channels,
            low_level_channels=spec.low_level.channels,
            num_seg_channels=num_seg_channels,
            decoder_channels=decoder_channels,
            skip_channels=skip_channels,
            aspp_rates=aspp_rates,
        )

        logger.info(
            f"DualHeadModel({backbone_name}): "
            f"low_level={spec.low_level.module_path}({spec.low_level.channels}ch@{spec.low_level.spatial}²), "
            f"high_level={spec.high_level.module_path}({spec.high_level.channels}ch@{spec.high_level.spatial}²), "
            f"decoder={decoder_channels}ch, seg_out={num_seg_channels}ch"
        )

    # ── forward ─────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass producing both classification and segmentation outputs.

        Parameters
        ----------
        x : (B, 3, H, W) input images.

        Returns
        -------
        dict with keys:
            ``cls_logits``  — (B, num_classes)
            ``seg_logits``  — (B, num_seg_channels, H, W)
        """
        # 1. Run backbone forward (hooks capture intermediate features)
        self._extractor.clear_features()
        cls_logits = self.backbone(x)

        # 2. Retrieve hooked features
        low_level = self._get_spatial_feature("low_level")
        high_level = self._get_spatial_feature("high_level")

        # 3. Segmentation decoder
        seg_logits = self.seg_decoder(
            high_level, low_level, target_size=(x.shape[2], x.shape[3])
        )

        return {
            "cls_logits": cls_logits,
            "seg_logits": seg_logits,
        }

    def _get_spatial_feature(self, name: str) -> torch.Tensor:
        """Retrieve a hooked feature and reshape to BCHW if needed."""
        import math as _math

        feat = self._extractor.get_feature(name)
        if feat is None:
            raise RuntimeError(
                f"Hook '{name}' did not fire for {self.backbone_name}. "
                "Ensure the backbone's forward() was executed."
            )
        spec_item = getattr(self._extractor.spec, name)
        if spec_item.needs_reshape and feat.dim() == 3:
            B, N, C = feat.shape
            h = w = spec_item.spatial
            if N != h * w:
                h = w = int(_math.sqrt(N))
            feat = feat.transpose(1, 2).reshape(B, C, h, w).contiguous()
        return feat

    # ── freeze / unfreeze helpers ───────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Phase A: seg head training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        logger.info(f"  Backbone FROZEN ({self.backbone_name})")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (Phase B: joint fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()
        logger.info(f"  Backbone UNFROZEN ({self.backbone_name})")

    def freeze_seg_head(self) -> None:
        """Freeze segmentation decoder (Phase C: classification refinement)."""
        for param in self.seg_decoder.parameters():
            param.requires_grad = False
        self.seg_decoder.eval()
        logger.info("  Segmentation decoder FROZEN")

    def unfreeze_seg_head(self) -> None:
        """Unfreeze segmentation decoder."""
        for param in self.seg_decoder.parameters():
            param.requires_grad = True
        self.seg_decoder.train()

    def freeze_cls_head(self) -> None:
        """Freeze classification head (Phase A)."""
        head = self._extractor.get_classifier_head()
        for param in head.parameters():
            param.requires_grad = False
        head.eval()
        logger.info("  Classification head FROZEN")

    def unfreeze_cls_head(self) -> None:
        """Unfreeze classification head."""
        head = self._extractor.get_classifier_head()
        for param in head.parameters():
            param.requires_grad = True
        head.train()

    # ── parameter groups for per-component LR ───────────────────────────

    def get_param_groups(
        self,
        backbone_lr: float = 1e-5,
        seg_head_lr: float = 1e-4,
        cls_head_lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> list[dict[str, Any]]:
        """Return optimizer parameter groups with per-component learning rates.

        Parameters with ``requires_grad=False`` are excluded automatically.
        """
        cls_head = self._extractor.get_classifier_head()
        cls_head_ids = {id(p) for p in cls_head.parameters()}

        backbone_params = [
            p for p in self.backbone.parameters()
            if p.requires_grad and id(p) not in cls_head_ids
        ]
        seg_params = [
            p for p in self.seg_decoder.parameters()
            if p.requires_grad
        ]
        cls_params = [
            p for p in cls_head.parameters()
            if p.requires_grad
        ]

        groups = []
        if backbone_params and backbone_lr > 0:
            groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "name": "backbone",
            })
        if seg_params and seg_head_lr > 0:
            groups.append({
                "params": seg_params,
                "lr": seg_head_lr,
                "weight_decay": weight_decay,
                "name": "seg_head",
            })
        if cls_params and cls_head_lr > 0:
            groups.append({
                "params": cls_params,
                "lr": cls_head_lr,
                "weight_decay": weight_decay,
                "name": "cls_head",
            })
        return groups

    # ── utilities ───────────────────────────────────────────────────────

    def seg_parameter_count(self) -> int:
        """Number of parameters in the segmentation decoder."""
        return sum(p.numel() for p in self.seg_decoder.parameters())

    def total_parameter_count(self) -> int:
        """Total parameters (backbone + seg decoder)."""
        return sum(p.numel() for p in self.parameters())

    def cleanup(self) -> None:
        """Remove hooks when done with model."""
        self._extractor.remove_hooks()
