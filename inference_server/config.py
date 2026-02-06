"""
Server Configuration
====================
Centralized configuration for the inference server.
All settings are read from environment variables with sensible defaults.
"""
from __future__ import annotations

import os
from pathlib import Path


# =============================================================================
# Path Configuration
# =============================================================================
BASE_DIR = Path(os.environ.get("DBT_BASE_DIR", Path(__file__).parent.parent))
CHECKPOINT_DIR = Path(os.environ.get("DBT_CKPT_DIR", BASE_DIR / "checkpoints"))
SPLIT_DIR = Path(os.environ.get("DBT_SPLIT_DIR", BASE_DIR / "split_dataset"))

# =============================================================================
# Model Configuration
# =============================================================================
# Default backbone to load for inference
DEFAULT_BACKBONE: str = os.environ.get("INFERENCE_BACKBONE", "CustomConvNeXt")

# Number of disease classes (auto-detected from split_dataset/train if available)
NUM_CLASSES: int = int(os.environ.get("INFERENCE_NUM_CLASSES", "13"))

# Image preprocessing
IMG_SIZE: int = int(os.environ.get("INFERENCE_IMG_SIZE", "224"))
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# =============================================================================
# Server Configuration
# =============================================================================
HOST: str = os.environ.get("INFERENCE_HOST", "0.0.0.0")
PORT: int = int(os.environ.get("INFERENCE_PORT", "8001"))
WORKERS: int = int(os.environ.get("INFERENCE_WORKERS", "1"))

# =============================================================================
# Class Names (auto-detected or from environment)
# =============================================================================


def get_class_names() -> list[str]:
    """
    Auto-detect class names from the split_dataset/train directory.
    Falls back to a sorted directory listing or a hardcoded default.
    """
    train_dir = SPLIT_DIR / "train"
    if train_dir.exists():
        classes = sorted(
            d.name for d in train_dir.iterdir() if d.is_dir()
        )
        if classes:
            return classes

    # Fallback: hardcoded class names from the 13-class dataset
    return [
        "Black_stripe",
        "Brown_spot",
        "Grassy_shoot_disease",
        "Healthy",
        "Leaf_flecking",
        "Leaf_scorching",
        "Mosaic",
        "Pokkah_boeng",
        "Red_rot",
        "Ring_spot",
        "Smut",
        "Wilt",
        "Yellow_leaf_Disease",
    ]
