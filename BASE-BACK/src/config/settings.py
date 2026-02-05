"""
Configuration settings for Disease Classification Framework
All paths, hyperparameters, and export configurations
"""

import os
from pathlib import Path


# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

# Detect if running from BASE-BACK subdirectory and adjust BASE_DIR accordingly
_current_dir = Path(__file__).resolve().parent.parent.parent  # Go from config/ to src/ to BASE-BACK/
if _current_dir.name == 'BASE-BACK':
    # We're inside BASE-BACK, so BASE_DIR should be parent
    BASE_DIR = Path(os.environ.get('DBT_BASE_DIR', str(_current_dir.parent)))
else:
    # We're at root level
    BASE_DIR = Path(os.environ.get('DBT_BASE_DIR', r"F:\DBT-Base-DIr"))

CKPT_DIR = BASE_DIR / 'checkpoints'
PLOTS_DIR = BASE_DIR / 'plots_metrics'
METRICS_DIR = BASE_DIR / 'metrics_output'
KFOLD_DIR = BASE_DIR / 'kfold_results'
DEPLOY_DIR = BASE_DIR / 'deployment_models'
SMOKE_CHECK_DIR = BASE_DIR / 'smoke_checks'
PRETRAINED_DIR = BASE_DIR / 'pretrained_weights'
DEBUG_LOG_DIR = BASE_DIR / 'debug_logs'

# Create all directories
for d in [CKPT_DIR, PLOTS_DIR, METRICS_DIR, KFOLD_DIR, DEPLOY_DIR, SMOKE_CHECK_DIR, PRETRAINED_DIR, DEBUG_LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset paths
RAW_DIR = Path(os.environ.get('DBT_RAW_DIR', str(BASE_DIR / 'Data')))
SPLIT_DIR = Path(os.environ.get('DBT_SPLIT_DIR', str(BASE_DIR / 'split_dataset')))
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

BACKBONES = [
    'CustomConvNeXt', 'CustomEfficientNetV4', 'CustomGhostNetV2',
    'CustomResNetMish', 'CustomCSPDarkNet', 'CustomInceptionV4',
    'CustomViTHybrid', 'CustomSwinTransformer', 'CustomCoAtNet',
    'CustomRegNet', 'CustomDenseNetHybrid', 'CustomDeiTStyle',
    'CustomMaxViT', 'CustomMobileOne', 'CustomDynamicConvNet'
]

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 13  # 13 disease classes (fixed from None to prevent 1000-class bug)
K_FOLDS = 5
SEED = 42

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

EPOCHS_HEAD = 40
EPOCHS_FINETUNE = 25
PATIENCE_HEAD = 5
PATIENCE_FT = 5
WEIGHT_DECAY = 1e-4
BACKBONE_LR = 1e-6
HEAD_LR = 1e-3

# =============================================================================
# FEATURE FLAGS
# =============================================================================

ENABLE_KFOLD_CV = True
ENABLE_EXPORT = True
ENABLE_TRANSFER_LEARNING = True
TRANSFER_LEARNING_MODE = 'fine_tuning'  # vs 'feature_extraction'

# =============================================================================
# TRANSFER LEARNING & PRETRAINED WEIGHTS
# =============================================================================

PRETRAINED_DOWNLOAD_MAP = {
    'swin_base_patch4_window7_224.pth': {
        'url': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
        'expected_size_mb': 418.0,
    },
    'convnext_base_22k_1k_224.pth': {
        'url': 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth',
        'expected_size_mb': 338.4,
    },
    'efficientnetv2_s.pth': {
        'url': 'https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth',
        'expected_size_mb': 82.0,
    },
    'resnet50_imagenet.pth': {
        'url': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
        'expected_size_mb': 97.8,
    },
    'densenet121_imagenet.pth': {
        'url': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
        'expected_size_mb': 30.8,
    },
    'vit_base_patch16_224.pth': {
        'url': 'https://download.pytorch.org/models/vit_b_16-c867db91.pth',
        'expected_size_mb': 330.0,
    },
    'regnety_008.pth': {
        'url': 'https://download.pytorch.org/models/regnet_y_8gf-dc2b1b54.pth',
        'expected_size_mb': 150.0,
    },
}

PRETRAINED_PATHS = {
    'CustomSwinTransformer': PRETRAINED_DIR / 'swin_base_patch4_window7_224.pth',
    'CustomConvNeXt': PRETRAINED_DIR / 'convnext_base_22k_1k_224.pth',
    'CustomEfficientNetV4': PRETRAINED_DIR / 'efficientnetv2_s.pth',
    'CustomResNetMish': PRETRAINED_DIR / 'resnet50_imagenet.pth',
    'CustomDenseNetHybrid': PRETRAINED_DIR / 'densenet121_imagenet.pth',
    'CustomViTHybrid': PRETRAINED_DIR / 'vit_base_patch16_224.pth',
    'CustomDeiTStyle': PRETRAINED_DIR / 'vit_base_patch16_224.pth',
    'CustomRegNet': PRETRAINED_DIR / 'regnety_008.pth',
    'CustomGhostNetV2': PRETRAINED_DIR / 'efficientnetv2_s.pth',
    'CustomCSPDarkNet': PRETRAINED_DIR / 'resnet50_imagenet.pth',
    'CustomInceptionV4': PRETRAINED_DIR / 'resnet50_imagenet.pth',
    'CustomCoAtNet': PRETRAINED_DIR / 'vit_base_patch16_224.pth',
    'CustomMaxViT': PRETRAINED_DIR / 'vit_base_patch16_224.pth',
    'CustomMobileOne': PRETRAINED_DIR / 'efficientnetv2_s.pth',
    'CustomDynamicConvNet': PRETRAINED_DIR / 'resnet50_imagenet.pth',
}

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

EXPORT_TYPES = [
    'state_dict',
    'checkpoint',
    'torchscript',
    'onnx',
    'torchserve',
    'tensorrt',
    'tflite',
    'custom',
]

EXPORT_FORMATS = EXPORT_TYPES if ENABLE_EXPORT else []

EXPORT_CONFIG = {
    'state_dict': {},
    'checkpoint': {
        'include_optimizer': True,
        'include_scheduler': True,
        'include_training_metadata': True,
    },
    'torchscript': {
        'trace_mode': True,
        'script_mode': True,
        'optimize_for_inference': True,
    },
    'onnx': {
        'opset_version': 13,
        'dynamic_axes': {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        'export_params': True,
        'do_constant_folding': True,
        'input_names': ['input'],
        'output_names': ['output'],
        'verbose': False,
    },
    'torchserve': {
        'model_version': '1.0',
        'handler': 'image_classifier',
        'extra_files': [],
        'requirements_file': None,
    },
    'tensorrt': {
        'enable_quantization': False,
        'fp16_mode': False,
        'max_batch_size': 32,
        'max_workspace_size': 1 << 30,
        'strict_type_constraints': False,
    },
    'tflite': {
        'enable_quantization': False,
        'quantization_dtype': 'float16',
        'representative_dataset': None,
        'optimizations': [],
    },
    'custom': {
        'format_name': 'custom',
        'exporter_function': None,
        'file_extension': '.bin',
        'additional_metadata': {},
    }
}

# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================

DEBUG_MODE = os.environ.get('DBT_DEBUG_MODE', 'false').lower() == 'true'
DEBUG_BACKBONE = os.environ.get('DBT_DEBUG_BACKBONE', 'CustomMaxViT')
DEBUG_FUNCTION = os.environ.get('DBT_DEBUG_FUNCTION', 'full_training')

DEBUG_FUNCTIONS = {
    'model_creation': 'Test model creation only',
    'forward_pass': 'Test forward pass with dummy input',
    'backward_pass': 'Test backward pass and gradients',
    'single_epoch': 'Train for 1 epoch only',
    'overfit_batch': 'Overfit on single batch (sanity check)',
    'dataset_loading': 'Test dataset loading and transforms',
    'full_training': 'Full training pipeline for single backbone',
    'export_only': 'Test export functionality only',
    'smoke_tests': 'Run smoke tests only',
    'architecture_verify': 'Verify architecture only',
    'pretrained_loading': 'Test pretrained weight loading',
    'learning_rate': 'Test learning rate scheduling',
    'all_checks': 'Run all debug checks'
}

if DEBUG_MODE:
    DEBUG_EPOCHS_HEAD = int(os.environ.get('DBT_DEBUG_HEAD_EPOCHS', '15'))
    DEBUG_EPOCHS_FINETUNE = int(os.environ.get('DBT_DEBUG_FT_EPOCHS', '10'))
    DEBUG_BATCH_SIZE = int(os.environ.get('DBT_DEBUG_BATCH_SIZE', '8'))
    DEBUG_ENABLE_EXPORT = os.environ.get('DBT_DEBUG_EXPORT', 'False').lower() == 'true'
else:
    DEBUG_EPOCHS_HEAD = EPOCHS_HEAD
    DEBUG_EPOCHS_FINETUNE = EPOCHS_FINETUNE
    DEBUG_BATCH_SIZE = BATCH_SIZE
    DEBUG_ENABLE_EXPORT = ENABLE_EXPORT
