"""
V2_segmentation/models/model_factory.py
=======================================
Factory functions:  build V2 dual-head models and load V1 checkpoints into them.

Workflow
-------
1. ``create_v1_backbone(name, num_classes)``
       — Instantiate a bare V1 backbone from Base_backbones.py
2. ``load_v1_weights(model, ckpt_path)``
       — Load a V1 checkpoint (handles all known formats)
3. ``build_v2_model(backbone_name, ...)``
       — Full pipeline:  create backbone → load V1 weights → wrap in DualHeadModel
4. ``load_v1_into_v2(backbone_name, ckpt_path, ...)``
       — Convenience: build + load in one call
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .dual_head import DualHeadModel

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependency with Base_backbones.py
_backbone_classes: dict[str, type] | None = None
_backbone_factory = None


def _get_backbone_classes() -> dict[str, type]:
    """Lazy-import backbone CLASSES (not factory) from Base_backbones.py.

    This returns the raw nn.Module classes, allowing minimal instantiation
    without the heavy verification that create_custom_backbone performs.
    """
    global _backbone_classes
    if _backbone_classes is not None:
        return _backbone_classes

    import sys
    from V2_segmentation.config import BASE_DIR

    bb_path = BASE_DIR / "Base_backbones.py"
    if not bb_path.exists():
        raise FileNotFoundError(f"Cannot find Base_backbones.py at {bb_path}")

    parent = str(BASE_DIR)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # Import individual backbone classes directly
    from Base_backbones import (  # type: ignore
        CustomConvNeXt, CustomEfficientNetV4, CustomGhostNetV2,
        CustomResNetMish, CustomCSPDarkNet, CustomInceptionV4,
        CustomViTHybrid, CustomSwinTransformer, CustomCoAtNet,
        CustomRegNet, CustomDenseNetHybrid, CustomDeiTStyle,
        CustomMaxViT, CustomMobileOne, CustomDynamicConvNet,
    )

    _backbone_classes = {
        "CustomConvNeXt": CustomConvNeXt,
        "CustomEfficientNetV4": CustomEfficientNetV4,
        "CustomGhostNetV2": CustomGhostNetV2,
        "CustomResNetMish": CustomResNetMish,
        "CustomCSPDarkNet": CustomCSPDarkNet,
        "CustomInceptionV4": CustomInceptionV4,
        "CustomViTHybrid": CustomViTHybrid,
        "CustomSwinTransformer": CustomSwinTransformer,
        "CustomCoAtNet": CustomCoAtNet,
        "CustomRegNet": CustomRegNet,
        "CustomDenseNetHybrid": CustomDenseNetHybrid,
        "CustomDeiTStyle": CustomDeiTStyle,
        "CustomMaxViT": CustomMaxViT,
        "CustomMobileOne": CustomMobileOne,
        "CustomDynamicConvNet": CustomDynamicConvNet,
    }
    return _backbone_classes


def _get_backbone_factory():
    """Lazy-import ``create_custom_backbone`` from Base_backbones.py.

    NOTE: This factory runs expensive architecture verification.
    For lightweight inference-only use, prefer create_v1_backbone_minimal().
    """
    global _backbone_factory
    if _backbone_factory is not None:
        return _backbone_factory

    import sys
    from V2_segmentation.config import BASE_DIR

    bb_path = BASE_DIR / "Base_backbones.py"
    if not bb_path.exists():
        raise FileNotFoundError(f"Cannot find Base_backbones.py at {bb_path}")

    parent = str(BASE_DIR)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from Base_backbones import create_custom_backbone  # type: ignore
    _backbone_factory = create_custom_backbone
    return _backbone_factory


# ============================================================================
#  V1 backbone creation
# ============================================================================

def create_v1_backbone(name: str, num_classes: int = 13) -> nn.Module:
    """Instantiate a V1 backbone MINIMALLY (no verification, no pretrained load).

    This is the FAST path for inference-only use cases like GradCAM generation.
    It directly instantiates the backbone class without the expensive
    architecture verification that create_custom_backbone performs.

    Parameters
    ----------
    name : str
        Backbone name (e.g. ``"CustomEfficientNetV4"``).
    num_classes : int
        Number of output classes.

    Returns
    -------
    nn.Module
    """
    classes = _get_backbone_classes()
    if name not in classes:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(classes.keys())}")

    model = classes[name](_num_classes=num_classes)
    logger.debug(f"Created V1 backbone (minimal): {name} with {num_classes} classes")
    return model


def create_v1_backbone_verified(name: str, num_classes: int = 13) -> nn.Module:
    """Instantiate a V1 backbone WITH full verification.

    This uses the original create_custom_backbone from Base_backbones.py
    which runs architecture verification, pretrained loading, etc.
    Use this for training or when you need verified model correctness.

    Parameters
    ----------
    name : str
        Backbone name (e.g. ``"CustomEfficientNetV4"``).
    num_classes : int
        Number of output classes.

    Returns
    -------
    nn.Module
    """
    factory = _get_backbone_factory()
    model = factory(name, num_classes)
    logger.info(f"Created V1 backbone (verified): {name} with {num_classes} classes")
    return model


# ============================================================================
#  V1 checkpoint loading
# ============================================================================

def load_v1_weights(
    model: nn.Module,
    ckpt_path: str | Path,
    strict: bool = False,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a V1 checkpoint into a backbone model.

    Handles all known checkpoint formats:
      - ``{"model_state_dict": ...}``
      - ``{"model": ...}``
      - ``{"state_dict": ...}``
      - Raw ``state_dict``

    Also strips ``module.`` prefix from DataParallel wrappers.

    Parameters
    ----------
    model : nn.Module
        V1 backbone to load weights into.
    ckpt_path : str | Path
        Path to ``.pth`` checkpoint.
    strict : bool
        If True, require exact key match.
    map_location : str
        Device for loading.

    Returns
    -------
    dict with keys ``matched``, ``missing``, ``unexpected``.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)

    # Extract state dict from various formats
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                logger.info(f"  Extracted state_dict from checkpoint['{key}']")
                break
        else:
            # Assume the entire dict IS the state dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip 'module.' prefix (DataParallel)
    cleaned = {}
    for k, v in state_dict.items():
        clean_key = k[7:] if k.startswith("module.") else k
        cleaned[clean_key] = v
    state_dict = cleaned

    # Load with mismatch reporting
    result = model.load_state_dict(state_dict, strict=strict)
    missing = list(result.missing_keys) if hasattr(result, "missing_keys") else []
    unexpected = list(result.unexpected_keys) if hasattr(result, "unexpected_keys") else []
    matched = len(state_dict) - len(unexpected)

    logger.info(
        f"  Loaded V1 weights: {matched} matched, "
        f"{len(missing)} missing, {len(unexpected)} unexpected"
    )
    return {"matched": matched, "missing": missing, "unexpected": unexpected}


# ============================================================================
#  Build V2 Dual-Head Model
# ============================================================================

def build_v2_model(
    backbone_name: str,
    num_classes: int = 13,
    num_seg_channels: int = 5,
    decoder_channels: int = 256,
    skip_channels: int = 48,
    aspp_rates: tuple[int, ...] = (6, 12, 18),
    img_size: int = 224,
    device: str = "cpu",
) -> "DualHeadModel":
    """Create a DualHeadModel with a fresh (random-init) backbone.

    Parameters
    ----------
    backbone_name : str
        One of the 15 backbone names.
    num_classes : int
        Disease classes.
    num_seg_channels : int
        Segmentation output channels (5).
    decoder_channels : int
        Decoder internal width.
    skip_channels : int
        Skip connection reduced channels.
    aspp_rates : tuple[int, ...]
        ASPP dilation rates.
    img_size : int
        Input image size.
    device : str
        Target device.

    Returns
    -------
    DualHeadModel
    """
    from .dual_head import DualHeadModel

    backbone = create_v1_backbone(backbone_name, num_classes)
    model = DualHeadModel(
        backbone=backbone,
        backbone_name=backbone_name,
        num_classes=num_classes,
        num_seg_channels=num_seg_channels,
        decoder_channels=decoder_channels,
        skip_channels=skip_channels,
        aspp_rates=aspp_rates,
        img_size=img_size,
    )
    model.to(device)
    return model


def load_v1_into_v2(
    backbone_name: str,
    ckpt_path: str | Path,
    num_classes: int = 13,
    num_seg_channels: int = 5,
    decoder_channels: int | None = None,
    skip_channels: int | None = None,
    device: str = "cpu",
) -> "DualHeadModel":
    """Build a DualHeadModel and load V1 checkpoint weights.

    If ``decoder_channels`` / ``skip_channels`` are None, they are looked up
    from ``config.BACKBONE_PROFILES``.

    Parameters
    ----------
    backbone_name : str
        Backbone name.
    ckpt_path : str | Path
        Path to V1 ``.pth`` checkpoint.
    num_classes : int
        Disease classes.
    num_seg_channels : int
        Segmentation channels.
    decoder_channels : int | None
        Override decoder width.
    skip_channels : int | None
        Override skip channels.
    device : str
        Target device.

    Returns
    -------
    DualHeadModel with V1 backbone weights loaded.
    """
    from V2_segmentation.config import BACKBONE_PROFILES
    from .dual_head import DualHeadModel

    profile = BACKBONE_PROFILES.get(backbone_name, {})
    if decoder_channels is None:
        decoder_channels = profile.get("decoder_channels", 256)
    if skip_channels is None:
        skip_channels = profile.get("skip_channels", 48)

    # Ensure non-None for type checker
    assert isinstance(decoder_channels, int), f"decoder_channels must be int, got {type(decoder_channels)}"
    assert isinstance(skip_channels, int), f"skip_channels must be int, got {type(skip_channels)}"

    # 1. Create backbone + dual-head wrapper
    backbone = create_v1_backbone(backbone_name, num_classes)

    # 2. Load V1 weights into backbone
    load_v1_weights(backbone, ckpt_path, strict=False, map_location="cpu")

    # 3. Wrap in DualHeadModel
    model = DualHeadModel(
        backbone=backbone,
        backbone_name=backbone_name,
        num_classes=num_classes,
        num_seg_channels=num_seg_channels,
        decoder_channels=decoder_channels,
        skip_channels=skip_channels,
    )

    model.to(device)
    logger.info(
        f"Built V2 model for {backbone_name}: "
        f"decoder={decoder_channels}ch, "
        f"seg_params={model.seg_parameter_count():,}, "
        f"total_params={model.total_parameter_count():,}"
    )
    return model
