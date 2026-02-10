"""
Multi-Model Loader
==================
Load ALL available trained models for ensemble inference.
Supports: 15 backbones, student model, and meta-learners.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from inference_server import config  # noqa: I001


logger = logging.getLogger("inference_server.multi_loader")


# ---------------------------------------------------------------------------
# Model Registry — populated by load_all_models()
# ---------------------------------------------------------------------------
_loaded_models: dict[str, nn.Module] = {}
_model_device: torch.device = torch.device("cpu")
_student_model: nn.Module | None = None
_meta_learner: Any = None  # XGBoost or MLP


def get_available_backbones() -> list[str]:
    """Get list of backbone names with available checkpoints."""
    checkpoint_dir = config.CHECKPOINT_DIR
    available = []
    for ckpt in checkpoint_dir.glob("*_final.pth"):
        backbone_name = ckpt.stem.replace("_final", "")
        available.append(backbone_name)
    return sorted(available)


def load_all_models(device: str | None = None) -> dict[str, nn.Module]:
    """
    Load ALL available backbone models.

    Returns a dict of {backbone_name: model} with all models in eval mode.
    """
    global _loaded_models, _model_device, _student_model, _meta_learner  # noqa: PLW0603

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device)
    _model_device = target_device

    # Return cached if already loaded
    if _loaded_models:
        return _loaded_models

    import sys
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from Base_backbones import create_custom_backbone  # noqa: E402

    checkpoint_dir = config.CHECKPOINT_DIR
    available_backbones = get_available_backbones()

    logger.info("Loading %d backbone models...", len(available_backbones))

    for backbone_name in available_backbones:
        checkpoint_path = checkpoint_dir / f"{backbone_name}_final.pth"
        try:
            model = create_custom_backbone(backbone_name, config.NUM_CLASSES)

            state = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=False)

            model = model.to(target_device)
            model.eval()

            _loaded_models[backbone_name] = model
            logger.info("  ✓ Loaded %s", backbone_name)

        except Exception as e:
            logger.warning("  ✗ Failed to load %s: %s", backbone_name, e)

    # Load student model if available
    global _student_model  # noqa: PLW0603
    student_path = Path(config.BASE_DIR) / "ensembles" / "stage7_distillation" / "student_model.pth"
    if student_path.exists():
        try:
            # Student model architecture - typically smaller
            from Base_backbones import create_custom_backbone
            student_state = torch.load(student_path, map_location=target_device, weights_only=False)
            student_net: nn.Module
            if isinstance(student_state, dict) and "model_state_dict" in student_state:
                student_net = create_custom_backbone("CustomMobileOne", config.NUM_CLASSES)
                student_net.load_state_dict(student_state["model_state_dict"], strict=False)
            elif isinstance(student_state, nn.Module):
                student_net = student_state
            else:
                student_net = create_custom_backbone("CustomMobileOne", config.NUM_CLASSES)
                student_net.load_state_dict(student_state, strict=False)

            student_net = student_net.to(target_device)
            student_net.eval()
            _student_model = student_net
            logger.info("  ✓ Loaded student model")
        except Exception as e:
            logger.warning("  ✗ Failed to load student model: %s", e)

    # Load XGBoost meta-learner if available
    meta_path = Path(config.BASE_DIR) / "ensembles" / "stage6_meta" / "xgboost" / "xgboost_meta.pkl"
    if meta_path.exists():
        try:
            import pickle
            with open(meta_path, "rb") as f:
                _meta_learner = pickle.load(f)
            logger.info("  ✓ Loaded XGBoost meta-learner")
        except Exception as e:
            logger.warning("  ✗ Failed to load meta-learner: %s", e)

    logger.info("Total models loaded: %d backbones + %s student + %s meta",
                len(_loaded_models),
                "1" if _student_model else "0",
                "1" if _meta_learner else "0")

    return _loaded_models


def get_all_models() -> dict[str, nn.Module]:
    """Return the cached model dictionary."""
    return _loaded_models


def get_student_model() -> nn.Module | None:
    """Return the student (distilled) model."""
    return _student_model


def get_meta_learner() -> Any:
    """Return the meta-learner (XGBoost/MLP)."""
    return _meta_learner


def get_model_device() -> torch.device:
    """Return the device models are loaded on."""
    return _model_device


def get_loaded_model_count() -> int:
    """Return the number of loaded models."""
    return len(_loaded_models) + (1 if _student_model else 0)
