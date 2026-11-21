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
# Model 2 (V2 dual-head classifiers) — opt-in addition to the broad ensemble
# =============================================================================
# V2 adds a second trained classifier's votes. Dual-head checkpoints are large,
# so this is OFF by default and limited to the light CNN backbones.
ENABLE_V2: bool = os.environ.get("DBT_ENABLE_V2", "1") in ("1", "true", "True")

CHECKPOINT_V2_DIR: Path = Path(os.environ.get("DBT_CKPT_V2_DIR", BASE_DIR / "checkpoints_v2"))

V2_BACKBONES: list[str] = [
    b.strip()
    for b in os.environ.get(
        "DBT_V2_BACKBONES",
        "CustomDenseNetHybrid,CustomInceptionV4,CustomEfficientNetV4,"
        "CustomCSPDarkNet,CustomGhostNetV2",
    ).split(",")
    if b.strip()
]

# Checkpoint filename pattern: {backbone}{V2_CKPT_SUFFIX}
V2_CKPT_SUFFIX: str = os.environ.get("DBT_V2_CKPT_SUFFIX", "_v2_phaseC_best.pth")


# =============================================================================
# Red Rot Segmentation Specialist (Stage 2 of Expert View)
# =============================================================================
# Best individual backbone on the Snapshot 3 retrain weights (~22.8% mIoU).
REDROT_SEG_BACKBONE: str = os.environ.get("DBT_REDROT_SEG_BACKBONE", "CustomResNetMish")
REDROT_SEG_IMG_SIZE: int = int(os.environ.get("DBT_REDROT_SEG_IMG_SIZE", "512"))

# Default to the migrated Red Rot package; override with env in other deployments.
_REDROT_PKG = Path(
    os.environ.get("DBT_REDROT_DIR", r"F:\Migration_RedRot_20260727\Paper_redrot")
)
REDROT_SEG_CKPT: Path = Path(
    os.environ.get(
        "DBT_REDROT_SEG_CKPT",
        _REDROT_PKG / "seg_checkpoints_retrain" / "CustomResNetMish_retrain_best.pth",
    )
)
REDROT_SEG_RETRAIN_DIR: Path = Path(os.environ.get("DBT_REDROT_SEG_DIR", REDROT_SEG_CKPT.parent))

# Soft-voting ensemble of Red Rot segmentation backbones (robust, like the
# classification ensemble). Each needs a {name}_retrain_best.pth in the dir above.
REDROT_SEG_BACKBONES: list[str] = [
    b.strip()
    for b in os.environ.get(
        "DBT_REDROT_SEG_BACKBONES",
        "CustomResNetMish,CustomDenseNetHybrid,CustomRegNet",
    ).split(",")
    if b.strip()
]

REDROT_SEG_CATEGORY_MAP: Path = Path(
    os.environ.get(
        "DBT_REDROT_CATEGORY_MAP",
        Path(__file__).parent / "data" / "redrot_category_mapping.json",
    )
)

# The 13-class classifier label that triggers the Red Rot specialist (Stage 2).
REDROT_CLASS_NAME: str = os.environ.get("DBT_REDROT_CLASS_NAME", "Red_rot")

# =============================================================================
# Expert View behaviour
# =============================================================================
# Reject non-sugarcane images before classification by default.
EXPERT_VALIDATE_DEFAULT: bool = os.environ.get("DBT_EXPERT_VALIDATE", "1") not in ("0", "false", "False")

# Verdict reconciliation thresholds (Expert View).
# Flag a disagreement when the broad classifier's top class is not Red Rot but the
# specialist's affected area reaches this percentage.
REDROT_DISAGREE_AREA_PCT: float = float(os.environ.get("DBT_REDROT_DISAGREE_AREA_PCT", "2.0"))
# Affected-area at/above which the reconciled verdict DECLARES Red Rot (specialist
# overrides a wrong "Healthy" from the broad classifier). Between the disagree and
# this threshold, the status is amber ("possible Red Rot — recheck").
REDROT_SAY_PCT: float = float(os.environ.get("DBT_REDROT_SAY_PCT", "5.0"))
# Surface a low-confidence note when the classifier's top confidence is below this.
CLS_LOW_CONF: float = float(os.environ.get("DBT_CLS_LOW_CONF", "0.60"))
# Run the Red Rot specialist on every Expert-pipeline image during the pilot, so the
# Farmer verdict can catch Red Rot even when the broad classifier predicts Healthy.
RUN_SPECIALIST_ALWAYS: bool = os.environ.get("DBT_RUN_SPECIALIST_ALWAYS", "1") in ("1", "true", "True")

# Validation leniency: the sugarcane gate's green-vegetation heuristic wrongly rejects
# diseased close-ups (brown/purple tissue). These override the validator's defaults so
# legitimate diseased images pass; the Farmer View also offers an "analyze anyway" fallback.
VALIDATION_MIN_VEGETATION: float = float(os.environ.get("DBT_VALIDATION_MIN_VEG", "0.20"))
VALIDATION_MIN_SUGARCANE: float = float(os.environ.get("DBT_VALIDATION_MIN_SUGARCANE", "0.40"))

# Remedial-measures content served with results (placeholder until agronomy content lands).
REMEDIAL_DB_PATH: Path = Path(
    os.environ.get("DBT_REMEDIAL_DB", Path(__file__).parent / "data" / "remedial_measures.json")
)

# =============================================================================
# CORS (Web Expert View client)
# =============================================================================
# Comma-separated allowed origins; "*" allows any (local development default).
CORS_ORIGINS: list[str] = [
    o.strip() for o in os.environ.get("DBT_CORS_ORIGINS", "*").split(",") if o.strip()
]


# =============================================================================
# Class Names (auto-detected or from environment)
# =============================================================================


def get_redrot_class_name() -> str:
    """Resolve the Red Rot class label as it appears in the active class list."""
    target = REDROT_CLASS_NAME.lower().replace(" ", "_")
    for name in get_class_names():
        if name.lower().replace(" ", "_") == target:
            return name
    return REDROT_CLASS_NAME


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
