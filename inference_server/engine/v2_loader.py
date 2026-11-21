"""
Model 2 (V2 dual-head) Classifier Loader
========================================
Loads the *classification* path of the V2 dual-head models so their 13-class
predictions can be folded into the broad Expert View ensemble alongside V1.

Only the backbone (which produces ``cls_logits``) is needed for classification,
so the heavy DeepLabV3+ segmentation decoder in the V2 checkpoint is skipped.
This keeps memory modest and avoids decoder channel-matching concerns.

Opt-in (``DBT_ENABLE_V2=1``) and limited by default to the light CNN backbones
(``DBT_V2_BACKBONES``), because V2 dual-head checkpoints are large.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch import nn

from inference_server import config


logger = logging.getLogger("inference_server.v2_loader")

_v2_models: dict[str, nn.Module] = {}
_v2_device: torch.device = torch.device("cpu")


def _ensure_project_root_on_path() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_v2_classifiers(device: str | None = None) -> dict[str, nn.Module]:
    """Load the V2 classification backbones. Cached. Fails loudly on real errors."""
    global _v2_models, _v2_device  # noqa: PLW0603

    if _v2_models:
        return _v2_models
    if not config.ENABLE_V2:
        return {}

    _ensure_project_root_on_path()
    from Base_backbones import create_custom_backbone

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _v2_device = torch.device(device)

    loaded: dict[str, nn.Module] = {}
    for name in config.V2_BACKBONES:
        ckpt_path = config.CHECKPOINT_V2_DIR / f"{name}{config.V2_CKPT_SUFFIX}"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"V2 checkpoint not found: {ckpt_path}. Set DBT_V2_BACKBONES / "
                f"DBT_CKPT_V2_DIR / DBT_V2_CKPT_SUFFIX, or disable with DBT_ENABLE_V2=0."
            )

        ckpt = torch.load(ckpt_path, map_location=_v2_device, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        # Keep only the backbone weights (classification path) and strip the prefix.
        backbone_state = {
            k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")
        }
        if not backbone_state:
            raise RuntimeError(f"[{name}] V2 checkpoint has no 'backbone.*' weights at {ckpt_path}.")

        model = create_custom_backbone(name, config.NUM_CLASSES)
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        # The classifier head + feature weights must be present; a largely-empty load is a real error.
        loaded_frac = 1.0 - (len(missing) / max(1, len(model.state_dict())))
        if loaded_frac < 0.5:
            raise RuntimeError(
                f"[{name}] only {loaded_frac:.0%} of backbone weights loaded from V2 checkpoint; "
                f"architecture mismatch (missing={len(missing)}, unexpected={len(unexpected)})."
            )

        model = model.to(_v2_device).eval()
        loaded[f"{name}_V2"] = model
        logger.info("  V ok Loaded V2 classifier %s (%.0f%% weights)", name, loaded_frac * 100)

    _v2_models = loaded
    logger.info("Loaded %d V2 classifiers for the ensemble.", len(_v2_models))
    return _v2_models


def get_v2_classifiers() -> dict[str, nn.Module]:
    return _v2_models


def get_v2_device() -> torch.device:
    return _v2_device
