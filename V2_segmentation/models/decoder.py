"""
V2_segmentation/models/decoder.py
=================================
DeepLabV3+ decoder with Atrous Spatial Pyramid Pooling (ASPP).

Architecture:
  1. ASPP on high-level features (bottleneck)
  2. 1×1 conv to reduce low-level feature channels
  3. Bilinear upsample ASPP output → low-level spatial size
  4. Concatenate + 3×3 conv refinement
  5. Final upsample to original resolution → per-pixel seg logits

Channel widths are tier-adaptive (set via ``decoder_channels`` in config).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
#  ASPP  (Atrous Spatial Pyramid Pooling)
# ============================================================================

class ASPPConv(nn.Sequential):
    """Single ASPP branch: atrous (dilated) 3×3 conv + BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    """Global-average-pooling ASPP branch (image-level features)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x = self.pool(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module.

    Parallel branches:
      - 1×1 conv
      - 3×3 conv at dilation rates (default 6, 12, 18)
      - Global average pooling + 1×1 conv

    All branches concatenated → 1×1 projection → dropout.

    Parameters
    ----------
    in_channels : int
        Input channels (from backbone's high-level features).
    atrous_rates : tuple[int, ...]
        Dilation rates for ASPP branches.
    out_channels : int
        Output channels after projection.
    dropout : float
        Dropout rate after projection.
    """

    def __init__(
        self,
        in_channels: int,
        atrous_rates: tuple[int, ...] = (6, 12, 18),
        out_channels: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []

        # Branch 1: 1×1 conv
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Branches 2–4: Atrous convolutions
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # Branch 5: Image-level pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Projection after concatenation: 5 × out_channels → out_channels
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [conv(x) for conv in self.convs]
        x = torch.cat(branches, dim=1)
        return self.project(x)


# ============================================================================
#  DeepLabV3+ Decoder
# ============================================================================

class DeepLabV3PlusDecoder(nn.Module):
    """DeepLabV3+-style decoder for segmentation.

    Combines ASPP-processed high-level features with low-level skip features.

    Parameters
    ----------
    high_level_channels : int
        Channels of the backbone's high-level (bottleneck) features.
    low_level_channels : int
        Channels of the backbone's low-level (skip) features.
    num_seg_channels : int
        Number of output segmentation channels (default 5 for our pipeline).
    decoder_channels : int
        Internal decoder width (tier-adaptive: 256/192/128).
    skip_channels : int
        Reduced channel count for the low-level skip connection (48 or 32).
    aspp_rates : tuple[int, ...]
        Dilation rates for ASPP.
    dropout : float
        Dropout in ASPP and final classifier.
    """

    def __init__(
        self,
        high_level_channels: int,
        low_level_channels: int,
        num_seg_channels: int = 5,
        decoder_channels: int = 256,
        skip_channels: int = 48,
        aspp_rates: tuple[int, ...] = (6, 12, 18),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ASPP on high-level features
        self.aspp = ASPP(
            in_channels=high_level_channels,
            atrous_rates=aspp_rates,
            out_channels=decoder_channels,
            dropout=dropout,
        )

        # 1×1 conv to reduce low-level feature channels
        self.skip_reduce = nn.Sequential(
            nn.Conv2d(low_level_channels, skip_channels, 1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True),
        )

        # Refinement after concatenation (ASPP + skip)
        self.refine = nn.Sequential(
            nn.Conv2d(decoder_channels + skip_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Final per-pixel classifier → segmentation logits
        # NOTE: No activation here — loss functions expect raw logits
        self.seg_classifier = nn.Conv2d(decoder_channels, num_seg_channels, 1)

    def forward(
        self,
        high_level: torch.Tensor,
        low_level: torch.Tensor,
        target_size: tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """Produce segmentation logits at ``target_size`` resolution.

        Parameters
        ----------
        high_level : (B, C_high, H_high, W_high)
            Backbone's bottleneck features.
        low_level : (B, C_low, H_low, W_low)
            Backbone's low-level skip features.
        target_size : (H, W)
            Output spatial resolution.

        Returns
        -------
        seg_logits : (B, num_seg_channels, H, W)
        """
        # 1. ASPP on bottleneck features
        x = self.aspp(high_level)                          # (B, dec_ch, H_high, W_high)

        # 2. Reduce low-level channels
        skip = self.skip_reduce(low_level)                 # (B, skip_ch, H_low, W_low)

        # 3. Upsample ASPP output to match low-level spatial size
        x = F.interpolate(
            x, size=skip.shape[2:], mode="bilinear", align_corners=False
        )                                                  # (B, dec_ch, H_low, W_low)

        # 4. Concatenate + refine
        x = torch.cat([x, skip], dim=1)                   # (B, dec_ch+skip_ch, H_low, W_low)
        x = self.refine(x)                                 # (B, dec_ch, H_low, W_low)

        # 5. Per-pixel segmentation logits
        x = self.seg_classifier(x)                         # (B, n_seg, H_low, W_low)

        # 6. Upsample to target size
        x = F.interpolate(
            x, size=target_size, mode="bilinear", align_corners=False
        )                                                  # (B, n_seg, H, W)
        return x
