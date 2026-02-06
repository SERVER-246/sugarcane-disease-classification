"""
Model Loader
=============
Load a trained backbone checkpoint and prepare it for inference.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Ensure project root is importable so we can reach Base_backbones
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from inference_server import config  # noqa: E402


logger = logging.getLogger("inference_server.loader")


# ---------------------------------------------------------------------------
# Module-level singleton — populated by load_model()
# ---------------------------------------------------------------------------
_loaded_model: nn.Module | None = None
_model_device: torch.device = torch.device("cpu")
_model_backbone: str = ""


def load_model(
    backbone_name: str | None = None,
    checkpoint_dir: Path | None = None,
    device: str | None = None,
) -> nn.Module:
    """
    Load a trained backbone model from a *_final.pth checkpoint.

    The model is stored in the module-level singleton so subsequent calls
    are essentially free.

    Parameters
    ----------
    backbone_name : str, optional
        Which backbone to load.  Defaults to ``config.DEFAULT_BACKBONE``.
    checkpoint_dir : Path, optional
        Directory containing checkpoint files.  Defaults to ``config.CHECKPOINT_DIR``.
    device : str, optional
        ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).

    Returns
    -------
    nn.Module
        The model in eval mode, on the target device.
    """
    global _loaded_model, _model_device, _model_backbone  # noqa: PLW0603

    backbone_name = backbone_name or config.DEFAULT_BACKBONE
    checkpoint_dir = Path(checkpoint_dir or config.CHECKPOINT_DIR)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device)

    # Return cached model if it matches the request
    if (
        _loaded_model is not None
        and _model_backbone == backbone_name
        and _model_device == target_device
    ):
        return _loaded_model

    # Locate checkpoint
    checkpoint_path = checkpoint_dir / f"{backbone_name}_final.pth"
    if not checkpoint_path.exists():
        # Fall back to head-best or finetune-best
        for suffix in ("_finetune_best.pth", "_head_best.pth"):
            alt = checkpoint_dir / f"{backbone_name}{suffix}"
            if alt.exists():
                checkpoint_path = alt
                break

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for {backbone_name} in {checkpoint_dir}"
        )

    logger.info("Loading %s from %s …", backbone_name, checkpoint_path.name)

    # Import the factory function from Base_backbones
    from Base_backbones import create_custom_backbone  # noqa: E402

    model: nn.Module = create_custom_backbone(backbone_name, config.NUM_CLASSES)

    # Load state-dict
    state = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)

    model = model.to(target_device)
    model.eval()

    # Cache
    _loaded_model = model
    _model_device = target_device
    _model_backbone = backbone_name

    logger.info(
        "✓ %s loaded on %s (%d classes)",
        backbone_name,
        target_device,
        config.NUM_CLASSES,
    )
    return model


def get_model() -> nn.Module | None:
    """Return the cached model (or ``None`` if not yet loaded)."""
    return _loaded_model


def get_model_device() -> torch.device:
    """Return the device the cached model lives on."""
    return _model_device


def get_model_backbone() -> str:
    """Return the backbone name of the currently loaded model."""
    return _model_backbone
