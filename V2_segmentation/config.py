"""
V2_segmentation/config.py
=========================
Central configuration for the segmentation-aware V2 pipeline.
All hyperparameters, paths, memory budgets, and gate thresholds live here.
"""

import os
import torch
from pathlib import Path

# ============================================================================
#  PATHS
# ============================================================================

BASE_DIR = Path(os.environ.get("DBT_BASE_DIR", r"F:\DBT-Base-DIr"))
RAW_DIR = BASE_DIR / "Data"
SPLIT_DIR = BASE_DIR / "split_dataset"

# V1 artifacts (read-only — never modified)
V1_CKPT_DIR = BASE_DIR / "checkpoints"
V1_ENSEMBLE_DIR = BASE_DIR / "ensembles"
V1_PLOTS_DIR = BASE_DIR / "plots_metrics"

# V2 artifacts
V2_ROOT = BASE_DIR / "V2_segmentation"
CKPT_V2_DIR = BASE_DIR / "checkpoints_v2"
ENSEMBLE_V2_DIR = CKPT_V2_DIR / "ensembles_v2"
PLOTS_V2_DIR = BASE_DIR / "plots_metrics_v2"
SEG_MASKS_DIR = BASE_DIR / "segmentation_masks"
PSEUDO_LABELS_DIR = BASE_DIR / "pseudo_labels"
GOLD_LABELS_DIR = BASE_DIR / "gold_labels"
DEPLOY_V2_DIR = BASE_DIR / "deployment_models_v2"
ANALYSIS_DIR = BASE_DIR / "analysis_output"
EVAL_DIR = BASE_DIR / "evaluation"
OOF_DIR = EVAL_DIR / "oof_predictions"
DEBUG_DIR = BASE_DIR / "debug_logs"

# ============================================================================
#  DEVICE
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = True  # FP16 mixed precision
VRAM_TOTAL_GB = 24.0
VRAM_HEADROOM_GB = 2.0  # Reserved for OS/driver
VRAM_SAFE_LIMIT_GB = VRAM_TOTAL_GB - VRAM_HEADROOM_GB  # 22.0 GB

# ============================================================================
#  BACKBONES
# ============================================================================

BACKBONES = [
    "CustomConvNeXt", "CustomEfficientNetV4", "CustomGhostNetV2",
    "CustomResNetMish", "CustomCSPDarkNet", "CustomInceptionV4",
    "CustomViTHybrid", "CustomSwinTransformer", "CustomCoAtNet",
    "CustomRegNet", "CustomDenseNetHybrid", "CustomDeiTStyle",
    "CustomMaxViT", "CustomMobileOne", "CustomDynamicConvNet",
]

NUM_CLASSES = 13  # 13 sugarcane disease/healthy classes
IMG_SIZE = 224
NUM_SEG_CHANNELS = 5  # BG, Healthy, Structural, Surface, Degradation

# ============================================================================
#  SEGMENTATION CHANNEL DEFINITIONS
# ============================================================================

SEG_CHANNELS = {
    0: "Background",
    1: "Healthy_Plant_Tissue",
    2: "Structural_Anomaly",        # Smut whip, grassy shoot, Pokkah_boeng
    3: "Surface_Disease_Sign",       # Spots, stripes, mosaic, rings, flecks
    4: "Tissue_Degradation",         # Wilt, scorch, yellowing, rot
}

# Disease → primary/secondary segmentation channels
DISEASE_CHANNEL_MAP = {
    "Healthy":               {"primary": 1, "secondary": None},
    "Red_rot":               {"primary": 4, "secondary": 3},
    "Brown_spot":            {"primary": 3, "secondary": None},
    "Mosaic":                {"primary": 3, "secondary": None},
    "Smut":                  {"primary": 2, "secondary": None},
    "Grassy_shoot_disease":  {"primary": 2, "secondary": None},
    "Wilt":                  {"primary": 4, "secondary": None},
    "Leaf_scorching":        {"primary": 4, "secondary": None},
    "Yellow_leaf_Disease":   {"primary": 4, "secondary": 3},
    "Pokkah_boeng":          {"primary": 2, "secondary": 4},
    "Ring_spot":             {"primary": 3, "secondary": None},
    "Leaf_flecking":         {"primary": 3, "secondary": None},
    "Black_stripe":          {"primary": 3, "secondary": None},
}

# Class name → index (alphabetical, matching V1 dataset ordering)
CLASS_NAMES = sorted(DISEASE_CHANNEL_MAP.keys())
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ============================================================================
#  MEMORY TIERS  (from profiled data — Section 3 of plan)
# ============================================================================

class MemoryTier:
    LIGHT = "LIGHT"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    HEAVY = "HEAVY"

# Per-backbone memory tier assignment + decoder configuration
BACKBONE_PROFILES = {
    # LIGHT tier: BS=32, grad_accum=1, decoder_channels=256
    "CustomEfficientNetV4": {
        "tier": MemoryTier.LIGHT, "params_m": 1.9, "v1_bs32_gb": 0.68,
        "batch_size": 32, "grad_accum": 1, "grad_checkpoint": False,
        "decoder_channels": 256, "aspp_channels": 256, "skip_channels": 48,
        "final_feat_channels": 1280,
        "stage_channels": [16, 24, 40, 80, 112, 192, 320],
    },
    "CustomDenseNetHybrid": {
        "tier": MemoryTier.LIGHT, "params_m": 2.1, "v1_bs32_gb": 1.63,
        "batch_size": 32, "grad_accum": 1, "grad_checkpoint": False,
        "decoder_channels": 256, "aspp_channels": 256, "skip_channels": 48,
        "final_feat_channels": 512,
        "stage_channels": [64, 128, 256, 512],
    },
    "CustomInceptionV4": {
        "tier": MemoryTier.LIGHT, "params_m": 2.4, "v1_bs32_gb": 2.62,
        "batch_size": 32, "grad_accum": 1, "grad_checkpoint": False,
        "decoder_channels": 256, "aspp_channels": 256, "skip_channels": 48,
        "final_feat_channels": 512,
        "stage_channels": [64, 128, 256, 512],
    },
    "CustomMobileOne": {
        "tier": MemoryTier.LIGHT, "params_m": 4.7, "v1_bs32_gb": 0.20,
        "batch_size": 32, "grad_accum": 1, "grad_checkpoint": False,
        "decoder_channels": 256, "aspp_channels": 256, "skip_channels": 48,
        "final_feat_channels": 384,
        "stage_channels": [48, 96, 192, 384],
    },
    "CustomGhostNetV2": {
        "tier": MemoryTier.LIGHT, "params_m": 9.6, "v1_bs32_gb": 1.85,
        "batch_size": 32, "grad_accum": 1, "grad_checkpoint": False,
        "decoder_channels": 256, "aspp_channels": 256, "skip_channels": 48,
        "final_feat_channels": 1280,
        "stage_channels": [16, 24, 40, 80, 112, 192, 320],
    },
    # MEDIUM tier: BS=16, grad_accum=2, decoder_channels=256
    "CustomConvNeXt": {
        "tier": MemoryTier.MEDIUM, "params_m": 27.8, "v1_bs32_gb": 2.18,
        "batch_size": 16, "grad_accum": 2, "grad_checkpoint": False,
        "decoder_channels": 256, "aspp_channels": 256, "skip_channels": 48,
        "final_feat_channels": 768,
        "stage_channels": [96, 192, 384, 768],
    },
    "CustomResNetMish": {
        "tier": MemoryTier.MEDIUM, "params_m": 23.5, "v1_bs32_gb": 2.16,
        "batch_size": 16, "grad_accum": 2, "grad_checkpoint": False,
        "decoder_channels": 256, "aspp_channels": 256, "skip_channels": 48,
        "final_feat_channels": 2048,
        "stage_channels": [64, 256, 512, 1024, 2048],
    },
    "CustomRegNet": {
        "tier": MemoryTier.MEDIUM, "params_m": 18.7, "v1_bs32_gb": 3.08,
        "batch_size": 16, "grad_accum": 2, "grad_checkpoint": False,
        "decoder_channels": 256, "aspp_channels": 256, "skip_channels": 48,
        "final_feat_channels": 1536,
        "stage_channels": [64, 128, 384, 768, 1536],
    },
    # HIGH tier: BS=8, grad_accum=4, decoder_channels=192
    "CustomCSPDarkNet": {
        "tier": MemoryTier.HIGH, "params_m": 3.9, "v1_bs32_gb": 10.27,
        "batch_size": 8, "grad_accum": 4, "grad_checkpoint": False,
        "decoder_channels": 192, "aspp_channels": 192, "skip_channels": 32,
        "final_feat_channels": 1024,
        "stage_channels": [64, 128, 256, 512, 1024],
    },
    "CustomDynamicConvNet": {
        "tier": MemoryTier.HIGH, "params_m": 72.6, "v1_bs32_gb": 15.61,
        "batch_size": 8, "grad_accum": 4, "grad_checkpoint": False,
        "decoder_channels": 192, "aspp_channels": 192, "skip_channels": 32,
        "final_feat_channels": 1024,
        "stage_channels": [64, 128, 256, 512, 1024],
    },
    # HEAVY tier: BS=4, grad_accum=8, gradient checkpointing ON, decoder_channels=128
    "CustomDeiTStyle": {
        "tier": MemoryTier.HEAVY, "params_m": 93.9, "v1_bs32_gb": 5.34,
        "batch_size": 4, "grad_accum": 8, "grad_checkpoint": True,
        "decoder_channels": 128, "aspp_channels": 128, "skip_channels": 32,
        "final_feat_channels": 768,
        "stage_channels": [192, 384, 768],
    },
    "CustomSwinTransformer": {
        "tier": MemoryTier.HEAVY, "params_m": 89.3, "v1_bs32_gb": 5.54,
        "batch_size": 4, "grad_accum": 8, "grad_checkpoint": True,
        "decoder_channels": 128, "aspp_channels": 128, "skip_channels": 32,
        "final_feat_channels": 1024,
        "stage_channels": [128, 256, 512, 1024],
    },
    "CustomMaxViT": {
        "tier": MemoryTier.HEAVY, "params_m": 106.4, "v1_bs32_gb": 10.35,
        "batch_size": 4, "grad_accum": 8, "grad_checkpoint": True,
        "decoder_channels": 128, "aspp_channels": 128, "skip_channels": 32,
        "final_feat_channels": 768,
        "stage_channels": [96, 192, 384, 768],
    },
    "CustomCoAtNet": {
        "tier": MemoryTier.HEAVY, "params_m": 117.4, "v1_bs32_gb": 9.99,
        "batch_size": 4, "grad_accum": 8, "grad_checkpoint": True,
        "decoder_channels": 128, "aspp_channels": 128, "skip_channels": 32,
        "final_feat_channels": 768,
        "stage_channels": [64, 96, 192, 384, 768],
    },
    "CustomViTHybrid": {
        "tier": MemoryTier.HEAVY, "params_m": 136.2, "v1_bs32_gb": 7.37,
        "batch_size": 4, "grad_accum": 8, "grad_checkpoint": True,
        "decoder_channels": 128, "aspp_channels": 128, "skip_channels": 32,
        "final_feat_channels": 768,
        "stage_channels": [64, 128, 256, 512, 768],
    },
}

# ============================================================================
#  TRAINING HYPERPARAMETERS
# ============================================================================

# Phase A: Segmentation head training (backbone FROZEN)
PHASE_A = {
    "epochs": 30,
    "backbone_lr": 0.0,  # frozen
    "seg_head_lr": 1e-3,
    "cls_head_lr": 0.0,  # frozen
    "lambda_seg": 1.0,
    "lambda_cls": 0.0,
    "patience": 5,
    "weight_decay": 1e-4,
}

# Phase B: Joint fine-tuning (backbone UNFROZEN, both heads)
PHASE_B = {
    "epochs": 25,
    "backbone_lr": 1e-5,
    "seg_head_lr": 1e-4,
    "cls_head_lr": 1e-4,
    "lambda_seg": 0.4,
    "lambda_cls": 0.6,
    "patience": 5,
    "weight_decay": 1e-4,
}

# Phase C: Classification refinement (seg head FROZEN)
PHASE_C = {
    "epochs": 15,
    "backbone_lr": 1e-6,
    "seg_head_lr": 0.0,  # frozen
    "cls_head_lr": 1e-4,
    "lambda_seg": 0.0,
    "lambda_cls": 1.0,
    "patience": 3,
    "weight_decay": 1e-4,
}

EFFECTIVE_BATCH_SIZE = 32  # All tiers achieve this via gradient accumulation
OPTIMIZER = "AdamW"
SCHEDULER = "CosineAnnealingWarmRestarts"
WARMUP_EPOCHS = 3
K_FOLDS = 5

# ============================================================================
#  SEGMENTATION DECODER
# ============================================================================

ASPP_RATES = [6, 12, 18]  # Atrous spatial pyramid pooling dilation rates
DECODER_DROPOUT = 0.1

# ============================================================================
#  PSEUDO-LABEL QUALITY
# ============================================================================

FUSION_WEIGHTS = {"grabcut": 0.3, "gradcam": 0.5, "sam": 0.2}
UNCERTAINTY_THRESHOLD = 0.6  # Pixels below this confidence masked out in loss
TIER_A_THRESHOLD = 0.80
TIER_B_THRESHOLD = 0.50
TIER_A_LOSS_WEIGHT = 1.0
TIER_B_LOSS_WEIGHT = 0.5
MAX_REFINEMENT_ROUNDS = 3
REFINEMENT_CONVERGENCE_PCT = 15.0  # Below this % pixel change = converged

# ============================================================================
#  LOSS FUNCTION PARAMETERS
# ============================================================================

DICE_SMOOTH = 1.0
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# ============================================================================
#  VALIDATION GATE THRESHOLDS (defaults — calibrated in Phase 0.5)
# ============================================================================

GATE_DEFAULT_MIN_PLANT_RATIO = 0.15
GATE_MIN_COMPONENT_SIZE_PX = 500
GATE_MIN_LESION_CONFIDENCE = 0.3
GATE_ENSEMBLE_SEG_MODELS = 3

# Per-class plant_ratio thresholds (overridden by Phase 0.5 calibration)
GATE_THRESHOLDS = {
    "Healthy": 0.35,
    "Red_rot": 0.12,
    "Brown_spot": 0.20,
    "Mosaic": 0.20,
    "Smut": 0.08,
    "Grassy_shoot_disease": 0.10,
    "Wilt": 0.10,
    "Leaf_scorching": 0.15,
    "Yellow_leaf_Disease": 0.20,
    "Pokkah_boeng": 0.12,
    "Ring_spot": 0.20,
    "Leaf_flecking": 0.20,
    "Black_stripe": 0.20,
}

# ============================================================================
#  ROLLBACK CRITERIA
# ============================================================================

ROLLBACK_VAL_ACC_DROP_THRESHOLD = 0.005  # 0.5% drop → revert to V1
ROLLBACK_MEAN_IOU_THRESHOLD = 0.50       # Below this → revert to V1

# ============================================================================
#  REPRODUCIBILITY
# ============================================================================

RUN_SEED = int(os.environ.get("V2_RUN_SEED", 42))

# ============================================================================
#  SERVING
# ============================================================================

SEG_GATE_BACKBONE = os.environ.get("SEG_GATE_BACKBONE", "CustomMobileOne")
