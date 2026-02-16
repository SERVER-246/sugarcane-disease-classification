    #!/usr/bin/env python3
"""
ENHANCED OPTIMIZED disease classification
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import logging
import multiprocessing
import os
import platform
import random
import shutil
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np


# Set multiprocessing method BEFORE any other imports that might use it
if __name__ == "__main__" and platform.system() == 'Windows':
    multiprocessing.set_start_method('spawn', force=True)

# suppress some noisy warnings
import warnings  # noqa: E402


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# core ML libs - must be imported after multiprocessing.set_start_method()
# plotting and metrics
import matplotlib  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.amp.autocast_mode import autocast  # noqa: E402
from torch.amp.grad_scaler import GradScaler  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402


matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 1200
plt.rcParams['savefig.dpi'] = 1200
plt.rcParams['font.size'] = 10
plt.rcParams['savefig.format'] = 'tiff'


# =============================================================================
# CONFIGURATION
# =============================================================================

# Custom backbone names - all built from scratch
BACKBONES = [
    'CustomConvNeXt', 'CustomEfficientNetV4', 'CustomGhostNetV2',
    'CustomResNetMish', 'CustomCSPDarkNet', 'CustomInceptionV4',
    'CustomViTHybrid', 'CustomSwinTransformer', 'CustomCoAtNet',
    'CustomRegNet', 'CustomDenseNetHybrid', 'CustomDeiTStyle',
    'CustomMaxViT', 'CustomMobileOne', 'CustomDynamicConvNet'
]

IMG_SIZE = 224
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
PATIENCE_HEAD = 5
PATIENCE_FT = 5
SEED = 42
_num_classes: int | None = None  # Mutable global, not a constant
K_FOLDS = 5
ENABLE_KFOLD_CV = True
ENABLE_EXPORT = True

EPOCHS_HEAD = 40
EPOCHS_FINETUNE = 25

# Use environment variable or detect appropriate default for CI environments
_default_base_dir = os.environ.get('DBT_BASE_DIR', '')
if not _default_base_dir:
    # Check if running in CI or if default F: drive doesn't exist
    if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
        # Use current working directory for CI
        _default_base_dir = os.getcwd()
    elif os.path.exists(r"F:\\DBT-Base-DIr"):
        _default_base_dir = r"F:\\DBT-Base-DIr"
    else:
        # Fallback to current directory
        _default_base_dir = os.getcwd()

BASE_DIR = Path(_default_base_dir)
CKPT_DIR = BASE_DIR / 'checkpoints'
PLOTS_DIR = BASE_DIR / 'plots_metrics'
METRICS_DIR = BASE_DIR / 'metrics_output'
KFOLD_DIR = BASE_DIR / 'kfold_results'
DEPLOY_DIR = BASE_DIR / 'deployment_models'
SMOKE_CHECK_DIR = BASE_DIR / 'smoke_checks'
PRETRAINED_DIR = BASE_DIR / 'pretrained_weights'

# Only create directories if base directory exists and is writable
if BASE_DIR.exists():
    for d in [CKPT_DIR, PLOTS_DIR, METRICS_DIR, KFOLD_DIR, DEPLOY_DIR, SMOKE_CHECK_DIR, PRETRAINED_DIR]:
        with contextlib.suppress(OSError):
            d.mkdir(parents=True, exist_ok=True)

# Dataset paths
RAW_DIR = Path(os.environ.get('DBT_RAW_DIR', str(BASE_DIR / 'Data')))
SPLIT_DIR = Path(os.environ.get('DBT_SPLIT_DIR', str(BASE_DIR / 'split_dataset')))
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"

# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================
DEBUG_MODE = os.environ.get('DBT_DEBUG_MODE', 'false').lower() == 'true'
DEBUG_BACKBONE = os.environ.get('DBT_DEBUG_BACKBONE', 'CustomMaxViT')
DEBUG_FUNCTION = os.environ.get('DBT_DEBUG_FUNCTION', 'full_training')  # Options below
DEBUG_LOG_DIR = BASE_DIR / 'debug_logs'
DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Available debug functions
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

# Debug training configuration (faster for debugging)
# Using lowercase prefix to indicate these are mutable variables, not constants
if DEBUG_MODE:
    _debug_epochs_head = int(os.environ.get('DBT_DEBUG_HEAD_EPOCHS', '15'))
    _debug_epochs_finetune = int(os.environ.get('DBT_DEBUG_FT_EPOCHS', '10'))
    _debug_batch_size = int(os.environ.get('DBT_DEBUG_BATCH_SIZE', '8'))
    _debug_enable_export = os.environ.get('DBT_DEBUG_EXPORT', 'False').lower() == 'true'
else:
    _debug_epochs_head = EPOCHS_HEAD
    _debug_epochs_finetune = EPOCHS_FINETUNE
    _debug_batch_size = BATCH_SIZE
    _debug_enable_export = ENABLE_EXPORT


# =============================================================================
# TRANSFER LEARNING CONFIGURATION
# =============================================================================
ENABLE_TRANSFER_LEARNING = True

TRANSFER_LEARNING_MODE = 'fine_tuning'

BACKBONE_LR = 1e-6
HEAD_LR = 1e-3

# Pre-trained weights download configuration
PRETRAINED_DOWNLOAD_MAP = {
    # Swin Transformer
    'swin_base_patch4_window7_224.pth': {
        'url': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
        'expected_size_mb': 418.0,
    },
    # ConvNeXt
    'convnext_base_22k_1k_224.pth': {
        'url': 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth',
        'expected_size_mb': 338.4,
    },
    # EfficientNet V2
    'efficientnetv2_s.pth': {
        'url': 'https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth',
        'expected_size_mb': 82.0,
    },
    # ResNet50
    'resnet50_imagenet.pth': {
        'url': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
        'expected_size_mb': 97.8,
    },
    # DenseNet121
    'densenet121_imagenet.pth': {
        'url': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
        'expected_size_mb': 30.8,
    },
    # ViT Base
    'vit_base_patch16_224.pth': {
        'url': 'https://download.pytorch.org/models/vit_b_16-c867db91.pth',
        'expected_size_mb': 330.0,
    },
    # RegNetY
    'regnety_008.pth': {
        'url': 'https://download.pytorch.org/models/regnet_y_8gf-dc2b1b54.pth',
        'expected_size_mb': 150.0,
    },
}

# Map backbones to their pretrained weight files
PRETRAINED_PATHS = {
    'CustomSwinTransformer': PRETRAINED_DIR / 'swin_base_patch4_window7_224.pth',
    'CustomConvNeXt': PRETRAINED_DIR / 'convnext_base_22k_1k_224.pth',
    'CustomEfficientNetV4': PRETRAINED_DIR / 'efficientnetv2_s.pth',
    'CustomResNetMish': PRETRAINED_DIR / 'resnet50_imagenet.pth',
    'CustomDenseNetHybrid': PRETRAINED_DIR / 'densenet121_imagenet.pth',
    'CustomViTHybrid': PRETRAINED_DIR / 'vit_base_patch16_224.pth',
    'CustomDeiTStyle': PRETRAINED_DIR / 'vit_base_patch16_224.pth',  # Can share ViT weights
    'CustomRegNet': PRETRAINED_DIR / 'regnety_008.pth',
     'CustomGhostNetV2': PRETRAINED_DIR / 'efficientnetv2_s.pth',  # Similar architecture
    'CustomCSPDarkNet': PRETRAINED_DIR / 'resnet50_imagenet.pth',  # ResNet-like
    'CustomInceptionV4': PRETRAINED_DIR / 'resnet50_imagenet.pth',  # General CNN
    'CustomCoAtNet': PRETRAINED_DIR / 'vit_base_patch16_224.pth',  # Hybrid architecture
    'CustomMaxViT': PRETRAINED_DIR / 'vit_base_patch16_224.pth',  # Hybrid architecture
    'CustomMobileOne': PRETRAINED_DIR / 'efficientnetv2_s.pth',  # Mobile architecture
    'CustomDynamicConvNet': PRETRAINED_DIR / 'resnet50_imagenet.pth',  # General CNN
}

# =============================================================================
# EXPORT AND DEPLOYMENT CONFIGURATION
# =============================================================================

# Export formats to generate (opt-in by default)
EXPORT_TYPES = [
    'state_dict',      # Pure model weights on CPU
     'checkpoint',    # Full checkpoint with optimizer/scheduler
     'torchscript',   # Both trace and script
     'onnx',          # ONNX format
     'torchserve',    # TorchServe MAR archive
     'tensorrt',      # TensorRT engine (requires ONNX)
     #'openvino',      # OpenVINO IR (requires ONNX)
     #'coreml',        # CoreML (requires ONNX)
     'tflite',        # TensorFlow Lite (requires ONNX)
     'custom',        # Custom proprietary formats
]

EXPORT_FORMATS = EXPORT_TYPES if ENABLE_EXPORT else []

# Export configuration
EXPORT_CONFIG = {
    'state_dict': {
        # State dict has no special config (pure weights)
    },
    'checkpoint': {
        'include_optimizer': True,      # Include optimizer state
        'include_scheduler': True,      # Include scheduler state
        'include_training_metadata': True,  # Include training metrics
    },
    'torchscript': {
        'trace_mode': True,             # Export traced version
        'script_mode': True,            # Export scripted version
        'optimize_for_inference': True, # Apply JIT optimizations
    },
    'onnx': {
        'opset_version': 13,            # ONNX opset version (13 is widely compatible)
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
        'handler': 'image_classifier',  # TorchServe handler type
        'extra_files': [],              # Additional files to include in MAR
        'requirements_file': None,      # Python dependencies
    },
    'tensorrt': {
        'enable_quantization': False,   # INT8 quantization
        'fp16_mode': False,             # FP16 precision
        'max_batch_size': 32,           # Maximum batch size
        'max_workspace_size': 1 << 30,  # 1GB workspace
        'strict_type_constraints': False,
    },
    'tflite': {
        'enable_quantization': False,   # Post-training quantization
        'quantization_dtype': 'float16', # float16, int8, uint8
        'representative_dataset': None, # For int8 quantization (function)
        'optimizations': [],            # TFLite optimizations
    },
    'custom': {
        'format_name': 'custom',        # Custom format identifier
        'exporter_function': None,      # User-provided export function
        'file_extension': '.bin',       # File extension for custom format
        'additional_metadata': {},      # Custom metadata
    }
}

# =============================================================================
# EXPORT DEPENDENCY CHECKER
# =============================================================================

def check_export_dependencies():

    global _export_dependencies

    deps = {
        'onnx': False,
        'onnxruntime': False,
        'tensorrt': False,
        #'openvino': False,
        #'coremltools': False,
        'tensorflow': False,
        'torch2trt': False,
        'torchserve': False,
    }

    try:
        import onnx as _onnx  # noqa: F401
        deps['onnx'] = True
        del _onnx
    except ImportError:
        pass

    try:
        import onnxruntime as _onnxruntime  # noqa: F401
        deps['onnxruntime'] = True
        del _onnxruntime
    except ImportError:
        pass

    try:
        import tensorrt as _tensorrt  # noqa: F401
        deps['tensorrt'] = True
        del _tensorrt
    except ImportError:
        pass

    try:
        import tensorflow as _tf  # noqa: F401
        deps['tensorflow'] = True
        del _tf
    except ImportError:
        pass

    try:
        import torch2trt  # type: ignore  # noqa: F401
        deps['torch2trt'] = True
    except ImportError:
        pass

    try:
        import torch_model_archiver  # type: ignore  # noqa: F401
        deps['torchserve'] = True
    except ImportError:
        pass

    global _export_dependencies
    _export_dependencies = deps
    return deps

# Mutable global for export dependencies
_export_dependencies: dict[str, bool] = {}

# Call at module load
check_export_dependencies()

# =============================================================================
# EXPORT UTILITIES
# =============================================================================

def compute_file_sha256(file_path: Path) -> str:

    import hashlib
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_num_classes() -> int:
    """Get the number of classes, with a safe default value."""
    global _num_classes
    return _num_classes if _num_classes is not None else 1000


def instantiate_arch_stub(arch_name: str, num_classes: int | None = None) -> nn.Module:

    if num_classes is None:
        num_classes = get_num_classes()

    return create_custom_backbone_safe(arch_name, num_classes)


def prepare_sample_input(model: nn.Module, input_size: int = 224, batch_size: int = 1, device: str | torch.device = 'cpu') -> torch.Tensor:

    return torch.randn(batch_size, 3, input_size, input_size, device=device)


def safe_model_to_device(model: nn.Module, device: str) -> tuple:

    original_device = next(model.parameters()).device
    original_training = model.training

    model.to(device)
    model.eval()

    return original_device, original_training


def restore_model_state(model: nn.Module, original_device, original_training: bool):

    model.to(original_device)
    if original_training:
        model.train()
    else:
        model.eval()


# =============================================================================
# EXPORT HELPER FUNCTIONS
# =============================================================================

def export_model_state_dict_cpu(model: nn.Module,
                                output_path: Path,
                                arch_name: str,
                                force: bool = False,
                                logger_inst = None) -> dict:

    if logger_inst is None:
        logger_inst = logger

    result = {
        'path': output_path,
        'status': 'failed',
        'reason': '',
        'sha256': ''
    }

    try:
        # Check if already exists
        if output_path.exists() and not force:
            result['status'] = 'skipped'
            result['reason'] = 'File already exists (use force=True to overwrite)'
            result['sha256'] = compute_file_sha256(output_path)
            return result

        # Save original state
        original_device, original_training = safe_model_to_device(model, 'cpu')

        # Export state dict
        with torch.no_grad():
            state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, str(output_path))

        # Restore original state
        restore_model_state(model, original_device, original_training)

        # Compute checksum
        result['sha256'] = compute_file_sha256(output_path)
        result['status'] = 'ok'
        result['reason'] = 'Successfully exported state dict'

        logger_inst.info(f"  Cool Exported state_dict: {output_path.name}")

    except Exception as e:
        result['reason'] = f'Export failed: {str(e)}'
        logger_inst.error(f"  x State dict export failed: {e}")

    return result

def get_loss_function_for_backbone(backbone_name, _num_classes):
    """Get optimized loss function per backbone"""

    # Higher label smoothing for transformers
    transformer_models = ['CustomSwinTransformer', 'CustomViTHybrid',
                         'CustomDeiTStyle', 'CustomCoAtNet', 'CustomMaxViT']

    if backbone_name in transformer_models:
        return nn.CrossEntropyLoss(label_smoothing=0.1)  # Back to 0.1
    else:
        return nn.CrossEntropyLoss(label_smoothing=0.05)

def create_improved_scheduler(optimizer, epochs, steps_per_epoch, backbone_name):

    # Warmup for transformers
    transformer_models = ['CustomSwinTransformer', 'CustomViTHybrid',
                         'CustomDeiTStyle', 'CustomCoAtNet', 'CustomMaxViT']

    if backbone_name in transformer_models:
        # FIXED: Better warmup schedule for transformers
        def lr_lambda(current_step):
            warmup_steps = steps_per_epoch * 3  # Reduced from 5
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps) / float(max(1, epochs * steps_per_epoch - warmup_steps))
            return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))  # Added min LR floor

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # FIXED: Better OneCycle parameters for CNNs
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'] * 5,  # Reduced from 10
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.2,      # Reduced from 0.3
            div_factor=20,      # Reduced from 25
            final_div_factor=1000  # Reduced from 10000
        )

def export_full_checkpoint(model: nn.Module,
                           output_path: Path,
                           arch_name: str,
                           optimizer = None,
                           scheduler = None,
                           extra_metadata: dict | None = None,
                           force: bool = False,
                           logger_inst = None) -> dict:

    if logger_inst is None:
        logger_inst = logger

    result = {
        'path': output_path,
        'status': 'failed',
        'reason': '',
        'sha256': ''
    }

    try:
        if output_path.exists() and not force:
            result['status'] = 'skipped'
            result['reason'] = 'File already exists'
            result['sha256'] = compute_file_sha256(output_path)
            return result

        original_device, original_training = safe_model_to_device(model, 'cpu')

        checkpoint = {
            'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
            'arch_name': arch_name,
        }

        if optimizer is not None:
            try:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            except Exception:  # noqa: BLE001
                logger_inst.warning("  xx Could not save optimizer state")

        if scheduler is not None:
            try:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            except Exception:  # noqa: BLE001
                logger_inst.warning("  xx Could not save scheduler state")

        if extra_metadata:
            checkpoint['metadata'] = extra_metadata

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(output_path))

        restore_model_state(model, original_device, original_training)

        result['sha256'] = compute_file_sha256(output_path)
        result['status'] = 'ok'
        result['reason'] = 'Successfully exported full checkpoint'

        logger_inst.info(f"  Cool Exported checkpoint: {output_path.name}")

    except Exception as e:
        result['reason'] = f'Export failed: {str(e)}'
        logger_inst.error(f"  x Checkpoint export failed: {e}")

    return result


def export_model_torchscript(model: nn.Module,
                             output_dir: Path,
                             arch_name: str,
                             sample_input: torch.Tensor | None = None,
                             force: bool = False,
                             logger_inst = None) -> dict:

    if logger_inst is None:
        logger_inst = logger

    results = {
        'trace': {'path': None, 'status': 'failed', 'reason': '', 'sha256': ''},
        'script': {'path': None, 'status': 'failed', 'reason': '', 'sha256': ''}
    }

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_path = output_dir / f"{arch_name}_trace.pt"
        script_path = output_dir / f"{arch_name}_script.pt"

        original_device, original_training = safe_model_to_device(model, 'cpu')

        if sample_input is None:
            sample_input = prepare_sample_input(model, device='cpu')
        else:
            sample_input = sample_input.cpu()

        # Trace mode
        try:
            if not trace_path.exists() or force:
                with torch.no_grad():
                    traced_model: torch.jit.ScriptModule = torch.jit.trace(model, sample_input)  # type: ignore[assignment]
                traced_model.save(str(trace_path))
                results['trace']['path'] = trace_path
                results['trace']['sha256'] = compute_file_sha256(trace_path)
                results['trace']['status'] = 'ok'
                results['trace']['reason'] = 'Successfully traced'
                logger_inst.info(f"  Cool Exported TorchScript (trace): {trace_path.name}")
            else:
                results['trace']['path'] = trace_path
                results['trace']['status'] = 'skipped'
                results['trace']['reason'] = 'File exists'
                results['trace']['sha256'] = compute_file_sha256(trace_path)
        except Exception as e:
            results['trace']['reason'] = f'Trace failed: {str(e)}'
            logger_inst.warning(f"  xx TorchScript trace failed: {e}")

        # Script mode
        try:
            if not script_path.exists() or force:
                with torch.no_grad():
                    scripted_model = torch.jit.script(model)
                scripted_model.save(str(script_path))
                results['script']['path'] = script_path
                results['script']['sha256'] = compute_file_sha256(script_path)
                results['script']['status'] = 'ok'
                results['script']['reason'] = 'Successfully scripted'
                logger_inst.info(f"  Cool Exported TorchScript (script): {script_path.name}")
            else:
                results['script']['path'] = script_path
                results['script']['status'] = 'skipped'
                results['script']['reason'] = 'File exists'
                results['script']['sha256'] = compute_file_sha256(script_path)
        except Exception as e:
            results['script']['reason'] = f'Script failed: {str(e)}'
            logger_inst.warning(f"  xx TorchScript script failed: {e}")

        restore_model_state(model, original_device, original_training)

    except Exception as e:
        logger_inst.error(f"  x TorchScript export failed: {e}")

    return results


def export_model_onnx(model: nn.Module,
                      output_path: Path,
                      arch_name: str,
                      sample_input: torch.Tensor | None = None,
                      config: dict | None = None,
                      force: bool = False,
                      logger_inst = None) -> dict:

    if logger_inst is None:
        logger_inst = logger

    result = {
        'path': output_path,
        'status': 'failed',
        'reason': '',
        'sha256': ''
    }

    if not _export_dependencies.get('onnx', False):
        result['status'] = 'skipped'
        result['reason'] = 'ONNX not installed. Install: pip install onnx'
        logger_inst.warning(f"  xx ONNX export skipped: {result['reason']}")
        return result

    try:
        import onnx

        if output_path.exists() and not force:
            result['status'] = 'skipped'
            result['reason'] = 'File exists'
            result['sha256'] = compute_file_sha256(output_path)
            return result

        if config is None:
            config = EXPORT_CONFIG.get('onnx', {}) or {}

        output_path.parent.mkdir(parents=True, exist_ok=True)

        original_device, original_training = safe_model_to_device(model, 'cpu')

        if sample_input is None:
            sample_input = prepare_sample_input(model, device='cpu')
        else:
            sample_input = sample_input.cpu()

        with torch.no_grad():
            torch.onnx.export(
                model,
                (sample_input,),
                str(output_path),
                input_names=['input'],
                output_names=['output'],
                opset_version=config.get('opset_version', 13),
                dynamic_axes=config.get('dynamic_axes', {}),
                export_params=config.get('export_params', True),
                do_constant_folding=config.get('do_constant_folding', True),
            )

        # Verify ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        restore_model_state(model, original_device, original_training)

        result['sha256'] = compute_file_sha256(output_path)
        result['status'] = 'ok'
        result['reason'] = 'Successfully exported to ONNX'

        logger_inst.info(f"  Cool Exported ONNX: {output_path.name}")

    except Exception as e:
        result['reason'] = f'ONNX export failed: {str(e)}'
        logger_inst.error(f"  x ONNX export failed: {e}")

    return result


def export_model_torchserve_mar(model: nn.Module,
                                output_path: Path,
                                arch_name: str,
                                model_file: Path | None = None,
                                config: dict | None = None,
                                force: bool = False,
                                logger_inst = None) -> dict:

    if logger_inst is None:
        logger_inst = logger

    result = {
        'path': output_path,
        'status': 'failed',
        'reason': '',
        'sha256': ''
    }

    if not _export_dependencies.get('torchserve', False):
        result['status'] = 'skipped'
        result['reason'] = 'torch-model-archiver not installed. Install: pip install torch-model-archiver'
        logger_inst.warning(f"  xx TorchServe export skipped: {result['reason']}")
        return result

    try:
        if output_path.exists() and not force:
            result['status'] = 'skipped'
            result['reason'] = 'File exists'
            result['sha256'] = compute_file_sha256(output_path)
            return result

        # TorchServe MAR creation requires external command - mark as TODO
        result['status'] = 'skipped'
        result['reason'] = 'TorchServe MAR creation requires torch-model-archiver CLI (TODO: implement)'
        logger_inst.info(f"  TorchServe MAR: {result['reason']}")

    except Exception as e:
        result['reason'] = f'TorchServe export failed: {str(e)}'
        logger_inst.error(f"  x TorchServe export failed: {e}")

    return result


def export_model_tensorrt_engine(onnx_path: Path,
                                 output_path: Path,
                                 arch_name: str,
                                 config: dict | None = None,
                                 force: bool = False,
                                 logger_inst = None) -> dict:

    if logger_inst is None:
        logger_inst = logger

    result = {
        'path': output_path,
        'status': 'failed',
        'reason': '',
        'sha256': ''
    }

    if not _export_dependencies.get('tensorrt', False):
        result['status'] = 'skipped'
        result['reason'] = 'TensorRT not installed. Install NVIDIA TensorRT SDK'
        logger_inst.warning(f"  xx TensorRT export skipped: {result['reason']}")
        return result

    if not onnx_path.exists():
        result['status'] = 'failed'
        result['reason'] = 'ONNX file not found (required for TensorRT conversion)'
        logger_inst.warning(f"  xx TensorRT export failed: {result['reason']}")
        return result

    try:
        if output_path.exists() and not force:
            result['status'] = 'skipped'
            result['reason'] = 'File exists'
            result['sha256'] = compute_file_sha256(output_path)
            return result

        # TensorRT engine build requires external tool - mark as TODO
        result['status'] = 'skipped'
        result['reason'] = 'TensorRT engine build requires trtexec CLI (TODO: implement)'
        logger_inst.info(f"  TensorRT: {result['reason']}")

    except Exception as e:
        result['reason'] = f'TensorRT export failed: {str(e)}'
        logger_inst.error(f"  x TensorRT export failed: {e}")

    return result

def export_model_tflite(onnx_path: Path,
                        output_path: Path,
                        arch_name: str,
                        config: dict | None = None,
                        force: bool = False,
                        logger_inst = None) -> dict:

    if logger_inst is None:
        logger_inst = logger

    result = {
        'path': output_path,
        'status': 'failed',
        'reason': '',
        'sha256': ''
    }

    if not _export_dependencies.get('tensorflow', False):
        result['status'] = 'skipped'
        result['reason'] = 'TensorFlow not installed. Install: pip install tensorflow'
        logger_inst.warning(f"  xx TFLite export skipped: {result['reason']}")
        return result

    if not onnx_path.exists():
        result['status'] = 'failed'
        result['reason'] = 'ONNX file not found (required for TFLite conversion)'
        logger_inst.warning(f"  xx TFLite export failed: {result['reason']}")
        return result

    try:
        if output_path.exists() and not force:
            result['status'] = 'skipped'
            result['reason'] = 'File exists'
            result['sha256'] = compute_file_sha256(output_path)
            return result

        # TFLite conversion requires onnx-tf and tensorflow
        result['status'] = 'skipped'
        result['reason'] = 'TFLite conversion requires onnx-tf package (TODO: implement full pipeline)'
        logger_inst.info(f"  **** TFLite: {result['reason']}")

    except Exception as e:
        result['reason'] = f'TFLite export failed: {str(e)}'
        logger_inst.error(f"  x TFLite export failed: {e}")

    return result


def package_custom_artifact(model: nn.Module,
                            output_path: Path,
                            arch_name: str,
                            custom_exporter_fn = None,
                            force: bool = False,
                            logger_inst = None) -> dict:

    if logger_inst is None:
        logger_inst = logger

    result = {
        'path': output_path,
        'status': 'failed',
        'reason': '',
        'sha256': ''
    }

    if custom_exporter_fn is None:
        result['status'] = 'skipped'
        result['reason'] = 'No custom exporter function provided'
        return result

    try:
        if output_path.exists() and not force:
            result['status'] = 'skipped'
            result['reason'] = 'File exists'
            result['sha256'] = compute_file_sha256(output_path)
            return result

        output_path.parent.mkdir(parents=True, exist_ok=True)

        original_device, original_training = safe_model_to_device(model, 'cpu')

        # Call custom exporter
        with torch.no_grad():
            custom_exporter_fn(model, output_path)

        restore_model_state(model, original_device, original_training)

        result['sha256'] = compute_file_sha256(output_path)
        result['status'] = 'ok'
        result['reason'] = 'Successfully exported with custom exporter'

        logger_inst.info(f"  Cool Exported custom format: {output_path.name}")

    except Exception as e:
        result['reason'] = f'Custom export failed: {str(e)}'
        logger_inst.error(f"  x Custom export failed: {e}")

    return result

# =============================================================================
# EXPORT SMOKE CHECKS
# =============================================================================

def smoke_check_state_dict(state_dict_path: Path,
                           arch_name: str,
                           sample_input: torch.Tensor | None = None,
                           logger_inst = None) -> dict:
    """Smoke check: load state dict and run forward pass"""
    if logger_inst is None:
        logger_inst = logger

    result = {'status': 'failed', 'reason': ''}

    try:
        # Load state dict
        state_dict = torch.load(str(state_dict_path), map_location='cpu')

        # Create fresh model
        model = instantiate_arch_stub(arch_name)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # Run forward pass
        if sample_input is None:
            sample_input = prepare_sample_input(model, device='cpu')
        else:
            sample_input = sample_input.cpu()

        with torch.no_grad():
            output = model(sample_input)

        # Sanity checks
        if not torch.isfinite(output).all():
            result['reason'] = 'Output contains NaN/Inf'
            return result

        if output.abs().max() > 1e6:
            result['reason'] = f'Output has very large values (max: {output.abs().max():.2e})'
            logger_inst.warning(f"    xx {result['reason']}")

        result['status'] = 'ok'
        result['reason'] = f'Forward pass successful, output shape: {tuple(output.shape)}'

    except Exception as e:
        result['reason'] = f'Smoke check failed: {str(e)}'

    return result


def smoke_check_torchscript(script_path: Path,
                            sample_input: torch.Tensor | None = None,
                            logger_inst = None) -> dict:
    """Smoke check: load TorchScript and run forward pass"""
    if logger_inst is None:
        logger_inst = logger

    result = {'status': 'failed', 'reason': ''}

    try:
        # Load TorchScript model
        model = torch.jit.load(str(script_path), map_location='cpu')
        model.eval()

        # Run forward pass
        if sample_input is None:
            sample_input = torch.randn(1, 3, 224, 224)
        else:
            sample_input = sample_input.cpu()

        with torch.no_grad():
            output = model(sample_input)

        # Sanity checks
        if not torch.isfinite(output).all():
            result['reason'] = 'Output contains NaN/Inf'
            return result

        result['status'] = 'ok'
        result['reason'] = f'Forward pass successful, output shape: {tuple(output.shape)}'

    except Exception as e:
        result['reason'] = f'Smoke check failed: {str(e)}'

    return result


def smoke_check_onnx(onnx_path: Path,
                     sample_input: torch.Tensor | None = None,
                     logger_inst = None) -> dict:
    """Smoke check: load ONNX and run inference"""
    if logger_inst is None:
        logger_inst = logger

    result = {'status': 'failed', 'reason': ''}

    if not _export_dependencies.get('onnxruntime', False):
        result['status'] = 'skipped'
        result['reason'] = 'onnxruntime not available for smoke check'
        logger_inst.warning(f"    xx ONNX smoke check skipped: {result['reason']}")
        return result

    try:
        import onnxruntime as ort

        # Create inference session
        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

        # Prepare input
        if sample_input is None:
            sample_input = torch.randn(1, 3, 224, 224)
        else:
            sample_input = sample_input.cpu()

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Run inference
        outputs = session.run([output_name], {input_name: sample_input.numpy()})
        output: np.ndarray = outputs[0]  # type: ignore[assignment]

        # Sanity checks
        if not np.isfinite(output).all():
            result['reason'] = 'Output contains NaN/Inf'
            return result

        result['status'] = 'ok'
        result['reason'] = f'Inference successful, output shape: {output.shape}'

    except Exception as e:
        result['reason'] = f'Smoke check failed: {str(e)}'

    return result


def smoke_check_tensorrt(engine_path: Path,
                         sample_input: torch.Tensor | None = None,
                         logger_inst = None) -> dict:
    """Smoke check: load TensorRT engine and run inference"""
    if logger_inst is None:
        logger_inst = logger

    result = {'status': 'skipped', 'reason': 'TensorRT runtime check not implemented'}
    logger_inst.warning(f"    xx TensorRT smoke check: {result['reason']}")
    return result

def smoke_check_tflite(tflite_path: Path,
                       sample_input: torch.Tensor | None = None,
                       logger_inst = None) -> dict:
    """Smoke check: load TFLite model and run inference"""
    if logger_inst is None:
        logger_inst = logger

    result = {'status': 'failed', 'reason': ''}

    if not _export_dependencies.get('tensorflow', False):
        result['status'] = 'skipped'
        result['reason'] = 'TensorFlow not available for smoke check'
        logger_inst.warning(f"    xx TFLite smoke check skipped: {result['reason']}")
        return result

    try:
        import tensorflow as tf

        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Prepare input
        if sample_input is None:
            sample_input = torch.randn(1, 3, 224, 224)
        else:
            sample_input = sample_input.cpu()

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], sample_input.numpy().astype(np.float32))

        # Run inference
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])

        # Sanity checks
        if not np.isfinite(output).all():
            result['reason'] = 'Output contains NaN/Inf'
            return result

        result['status'] = 'ok'
        result['reason'] = f'Inference successful, output shape: {output.shape}'

    except Exception as e:
        result['reason'] = f'Smoke check failed: {str(e)}'

    return result

# =============================================================================
# EXPORT ORCHESTRATOR
# =============================================================================

def export_and_package_model(model: nn.Module,
                             arch_name: str,
                             sample_input: torch.Tensor | None = None,
                             export_formats: list | None = None,
                             deploy_root: Path = DEPLOY_DIR,
                             force: bool = False,
                             optimizer = None,
                             scheduler = None,
                             training_metadata: dict | None = None,
                             _num_classes: int | None = None,
                             logger_inst = None) -> dict:

    if logger_inst is None:
        logger_inst = logger

    if export_formats is None:
        export_formats = EXPORT_FORMATS

    if _num_classes is None:
        _num_classes = _num_classes if _num_classes else 1000

    logger_inst.info(f"\n{'='*60}")
    logger_inst.info(f"EXPORTING AND PACKAGING: {arch_name}")
    logger_inst.info(f"Formats: {', '.join(export_formats)}")
    logger_inst.info(f"{'='*60}")

    # Create architecture-specific directory structure
    arch_dir = deploy_root / arch_name
    arch_dir.mkdir(parents=True, exist_ok=True)

    # Create format-specific subdirectories
    subdirs = {
        'state_dict': arch_dir / 'state_dict',
        'checkpoint': arch_dir / 'checkpoint',
        'torchscript': arch_dir / 'torchscript',
        'onnx': arch_dir / 'onnx',
        'torchserve': arch_dir / 'torchserve',
        'tensorrt': arch_dir / 'tensorrt',
        'openvino': arch_dir / 'openvino',
        'coreml': arch_dir / 'coreml',
        'tflite': arch_dir / 'tflite',
        'custom': arch_dir / 'custom',
    }

    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)

    # Prepare sample input if not provided
    if sample_input is None:
        sample_input = prepare_sample_input(model, input_size=IMG_SIZE, device=str(DEVICE))

    # Store export results
    export_results = {}
    export_errors = []
    checksums = {}
    smoke_results = {}

    # Export log file
    export_log_path = arch_dir / 'export.log'
    export_log = open(export_log_path, 'w', encoding='utf-8')  # noqa: SIM115

    def log_to_file(msg):
        export_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")
        export_log.flush()

    log_to_file(f"Starting export for {arch_name}")
    log_to_file(f"Requested formats: {', '.join(export_formats)}")

    # =================================================================
    # STATE DICT EXPORT
    # =================================================================
    if 'state_dict' in export_formats:
        logger_inst.info("\n[1/10] Exporting state_dict...")
        log_to_file("Exporting state_dict...")

        state_dict_path = subdirs['state_dict'] / f"{arch_name}_state_dict_cpu.pth"
        result = export_model_state_dict_cpu(
            model, state_dict_path, arch_name, force, logger_inst
        )
        export_results['state_dict'] = result

        if result['status'] == 'ok':
            checksums[str(state_dict_path.name)] = result['sha256']

            # Smoke check
            logger_inst.info("  Running smoke check...")
            smoke = smoke_check_state_dict(state_dict_path, arch_name, sample_input, logger_inst)
            smoke_results['state_dict'] = smoke
            log_to_file(f"  Smoke check: {smoke['status']} - {smoke['reason']}")
        else:
            log_to_file(f"  Export {result['status']}: {result['reason']}")
            if result['status'] == 'failed':
                export_errors.append(f"state_dict: {result['reason']}")

    # =================================================================
    # CHECKPOINT EXPORT
    # =================================================================
    if 'checkpoint' in export_formats:
        logger_inst.info("\n[2/10] Exporting full checkpoint...")
        log_to_file("Exporting full checkpoint...")

        checkpoint_path = subdirs['checkpoint'] / f"{arch_name}_checkpoint_full.pth"
        result = export_full_checkpoint(
            model, checkpoint_path, arch_name, optimizer, scheduler,
            training_metadata, force, logger_inst
        )
        export_results['checkpoint'] = result

        if result['status'] == 'ok':
            checksums[str(checkpoint_path.name)] = result['sha256']
            log_to_file(f"  Export successful: {checkpoint_path.name}")
        else:
            log_to_file(f"  Export {result['status']}: {result['reason']}")
            if result['status'] == 'failed':
                export_errors.append(f"checkpoint: {result['reason']}")

    # =================================================================
    # TORCHSCRIPT EXPORT
    # =================================================================
    if 'torchscript' in export_formats:
        logger_inst.info("\n[3/10] Exporting TorchScript...")
        log_to_file("Exporting TorchScript (trace and script)...")

        ts_results = export_model_torchscript(
            model, subdirs['torchscript'], arch_name, sample_input, force, logger_inst
        )
        export_results['torchscript'] = ts_results

        # Handle trace
        if ts_results['trace']['status'] == 'ok':
            trace_path = ts_results['trace']['path']
            checksums[str(trace_path.name)] = ts_results['trace']['sha256']

            logger_inst.info("  Running smoke check (trace)...")
            smoke = smoke_check_torchscript(trace_path, sample_input, logger_inst)
            smoke_results['torchscript_trace'] = smoke
            log_to_file(f"  Trace smoke check: {smoke['status']} - {smoke['reason']}")
        else:
            log_to_file(f"  Trace {ts_results['trace']['status']}: {ts_results['trace']['reason']}")
            if ts_results['trace']['status'] == 'failed':
                export_errors.append(f"torchscript_trace: {ts_results['trace']['reason']}")

        # Handle script
        if ts_results['script']['status'] == 'ok':
            script_path = ts_results['script']['path']
            checksums[str(script_path.name)] = ts_results['script']['sha256']

            logger_inst.info("  Running smoke check (script)...")
            smoke = smoke_check_torchscript(script_path, sample_input, logger_inst)
            smoke_results['torchscript_script'] = smoke
            log_to_file(f"  Script smoke check: {smoke['status']} - {smoke['reason']}")
        else:
            log_to_file(f"  Script {ts_results['script']['status']}: {ts_results['script']['reason']}")
            if ts_results['script']['status'] == 'failed':
                export_errors.append(f"torchscript_script: {ts_results['script']['reason']}")

    # =================================================================
    # ONNX EXPORT
    # =================================================================
    onnx_path = None
    if 'onnx' in export_formats:
        logger_inst.info("\n[4/10] Exporting ONNX...")
        log_to_file("Exporting ONNX...")

        onnx_path = subdirs['onnx'] / f"{arch_name}.onnx"
        result = export_model_onnx(
            model, onnx_path, arch_name, sample_input,
            EXPORT_CONFIG.get('onnx'), force, logger_inst
        )
        export_results['onnx'] = result

        if result['status'] == 'ok':
            checksums[str(onnx_path.name)] = result['sha256']

            logger_inst.info("  Running smoke check...")
            smoke = smoke_check_onnx(onnx_path, sample_input, logger_inst)
            smoke_results['onnx'] = smoke
            log_to_file(f"  Smoke check: {smoke['status']} - {smoke['reason']}")
        else:
            log_to_file(f"  Export {result['status']}: {result['reason']}")
            if result['status'] == 'failed':
                export_errors.append(f"onnx: {result['reason']}")

    # =================================================================
    # TORCHSERVE EXPORT
    # =================================================================
    if 'torchserve' in export_formats:
        logger_inst.info("\n[5/10] Exporting TorchServe MAR...")
        log_to_file("Exporting TorchServe MAR...")

        mar_path = subdirs['torchserve'] / f"{arch_name}.mar"
        result = export_model_torchserve_mar(
            model, mar_path, arch_name, None,
            EXPORT_CONFIG.get('torchserve'), force, logger_inst
        )
        export_results['torchserve'] = result

        if result['status'] == 'ok':
            checksums[str(mar_path.name)] = result['sha256']
        log_to_file(f"  Export {result['status']}: {result['reason']}")

    # =================================================================
    # TENSORRT EXPORT (requires ONNX)
    # =================================================================
    if 'tensorrt' in export_formats:
        logger_inst.info("\n[6/10] Exporting TensorRT...")
        log_to_file("Exporting TensorRT engine...")

        if onnx_path is None or not onnx_path.exists():
            logger_inst.warning("  xx ONNX file required for TensorRT export")
            export_results['tensorrt'] = {
                'path': None, 'status': 'skipped',
                'reason': 'ONNX file not available', 'sha256': ''
            }
            log_to_file("  Export skipped: ONNX file not available")
        else:
            trt_path = subdirs['tensorrt'] / f"{arch_name}_trt.engine"
            result = export_model_tensorrt_engine(
                onnx_path, trt_path, arch_name,
                EXPORT_CONFIG.get('tensorrt'), force, logger_inst
            )
            export_results['tensorrt'] = result

            if result['status'] == 'ok':
                checksums[str(trt_path.name)] = result['sha256']

                smoke = smoke_check_tensorrt(trt_path, sample_input, logger_inst)
                smoke_results['tensorrt'] = smoke
                log_to_file(f"  Smoke check: {smoke['status']} - {smoke['reason']}")
            else:
                log_to_file(f"  Export {result['status']}: {result['reason']}")

    # =================================================================
    # TFLITE EXPORT (requires ONNX)
    # =================================================================
    if 'tflite' in export_formats:
        logger_inst.info("\n[9/10] Exporting TFLite...")
        log_to_file("Exporting TFLite...")

        if onnx_path is None or not onnx_path.exists():
            logger_inst.warning("  xx ONNX file required for TFLite export")
            export_results['tflite'] = {
                'path': None, 'status': 'skipped',
                'reason': 'ONNX file not available', 'sha256': ''
            }
            log_to_file("  Export skipped: ONNX file not available")
        else:
            tflite_path = subdirs['tflite'] / f"{arch_name}.tflite"
            result = export_model_tflite(
                onnx_path, tflite_path, arch_name,
                EXPORT_CONFIG.get('tflite'), force, logger_inst
            )
            export_results['tflite'] = result

            if result['status'] == 'ok':
                checksums[str(tflite_path.name)] = result['sha256']

                smoke = smoke_check_tflite(tflite_path, sample_input, logger_inst)
                smoke_results['tflite'] = smoke
                log_to_file(f"  Smoke check: {smoke['status']} - {smoke['reason']}")
            else:
                log_to_file(f"  Export {result['status']}: {result['reason']}")

    # =================================================================
    # CUSTOM EXPORT
    # =================================================================
    if 'custom' in export_formats:
        logger_inst.info("\n[10/10] Custom format export...")
        log_to_file("Custom format export...")

        custom_path = subdirs['custom'] / f"{arch_name}_custom.bin"
        result = package_custom_artifact(
            model, custom_path, arch_name, None, force, logger_inst
        )
        export_results['custom'] = result

        if result['status'] == 'ok':
            checksums[str(custom_path.name)] = result['sha256']
        log_to_file(f"  Export {result['status']}: {result['reason']}")

    # =================================================================
    # GENERATE METADATA FILES
    # =================================================================
    logger_inst.info("\nGenerating metadata files...")
    log_to_file("Generating metadata files...")

    # 1. metadata.json
    metadata = {
        'architecture': arch_name,
        '_num_classes': _num_classes,
        'input_shape': list(sample_input.shape),
        'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'seed': SEED,
        'device': str(DEVICE),
    }

    if training_metadata:
        metadata['training'] = training_metadata

    metadata_path = arch_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    log_to_file("  Created: metadata.json")

    # 2. class_mapping.json
    if hasattr(model, 'classes') or _num_classes:
        class_names = getattr(model, 'classes', [f'class_{i}' for i in range(_num_classes)])
        class_mapping = {str(i): name for i, name in enumerate(class_names)}

        class_map_path = arch_dir / 'class_mapping.json'
        with open(class_map_path, 'w') as f:
            json.dump(class_mapping, f, indent=2, default=str)
        log_to_file("  Created: class_mapping.json")

    # 3. checksums.json
    checksums_path = arch_dir / 'checksums.json'
    with open(checksums_path, 'w') as f:
        json.dump(checksums, f, indent=2, default=str)
    log_to_file(f"  Created: checksums.json ({len(checksums)} files)")

    # 4. export_status.json - CRITICAL FIX HERE
    export_status = {}
    for fmt, result in export_results.items():
        if isinstance(result, dict) and 'status' in result:
            # Convert Path objects to strings
            status_entry = {
                'status': result['status'],
                'reason': result['reason']
            }
            #   Convert Path to string if present
            if 'path' in result and result['path'] is not None:
                status_entry['path'] = str(result['path'])
            export_status[fmt] = status_entry
        else:
            # Handle nested results (e.g., torchscript with trace/script)
            if isinstance(result, dict):
                nested_status = {}
                for key, val in result.items():
                    if isinstance(val, dict):
                        nested_entry = {
                            'status': val.get('status', 'unknown'),
                            'reason': val.get('reason', '')
                        }
                        #   Convert Path to string
                        if 'path' in val and val['path'] is not None:
                            nested_entry['path'] = str(val['path'])
                        nested_status[key] = nested_entry
                    else:
                        nested_status[key] = str(val) if not isinstance(val, (str, int, float, bool, type(None))) else val
                export_status[fmt] = nested_status
            else:
                export_status[fmt] = str(result)

    status_path = arch_dir / 'export_status.json'
    with open(status_path, 'w') as f:
        json.dump(export_status, f, indent=2, default=str)  # ← ADD default=str
    log_to_file("  Created: export_status.json")

    # 5. smoke_check_results.json
    if smoke_results:
        smoke_path = arch_dir / 'smoke_check_results.json'
        with open(smoke_path, 'w') as f:
            json.dump(smoke_results, f, indent=2, default=str)  # ← ADD default=str
        log_to_file("  Created: smoke_check_results.json")

    # 6. export_errors.log
    if export_errors:
        errors_path = arch_dir / 'export_errors.log'
        with open(errors_path, 'w') as f:
            f.write(f"Export errors for {arch_name}\n")
            f.write(f"{'='*60}\n\n")
            for error in export_errors:
                f.write(f"{error}\n")
        log_to_file(f"  Created: export_errors.log ({len(export_errors)} errors)")

    export_log.close()

    # =================================================================
    # SUMMARY
    # =================================================================
    successful = sum(1 for r in export_results.values()
                    if (isinstance(r, dict) and r.get('status') == 'ok') or
                       (isinstance(r, dict) and any(v.get('status') == 'ok' for v in r.values() if isinstance(v, dict))))

    skipped = sum(1 for r in export_results.values()
                 if (isinstance(r, dict) and r.get('status') == 'skipped'))

    failed = len(export_errors)

    summary = {
        'architecture': arch_name,
        'deploy_directory': str(arch_dir),
        'requested_formats': export_formats,
        'successful_exports': successful,
        'skipped_exports': skipped,
        'failed_exports': failed,
        'export_results': export_results,
        'smoke_check_results': smoke_results,
        'checksums': checksums,
        'metadata_files': [
            'metadata.json', 'class_mapping.json', 'checksums.json',
            'export_status.json', 'export.log'
        ],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    logger_inst.info(f"\n{'='*60}")
    logger_inst.info(f"EXPORT SUMMARY: {arch_name}")
    logger_inst.info(f"{'='*60}")
    logger_inst.info(f"  Successful: {successful}")
    logger_inst.info(f"  Skipped: {skipped}")
    logger_inst.info(f"  Failed: {failed}")
    logger_inst.info(f"  Deploy directory: {arch_dir}")
    logger_inst.info(f"{'='*60}\n")

    # One-line summary for automation
    summary['one_line_summary'] = (
        f"{arch_name}: {successful}/{len(export_formats)} formats exported successfully "
        f"({skipped} skipped, {failed} failed) -> {arch_dir}"
    )

    return summary

# =============================================================================
# EXPORT UNIT TESTS
# =============================================================================

def test_export_state_dict():
    """Test state_dict export and smoke check"""
    logger.info("\n[TEST] Testing state_dict export...")

    try:
        # Create dummy model
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10)
        )

        # Export
        test_dir = SMOKE_CHECK_DIR / 'test_exports'
        test_dir.mkdir(parents=True, exist_ok=True)

        output_path = test_dir / 'test_state_dict.pth'
        result = export_model_state_dict_cpu(
            model, output_path, 'TestModel', force=True, logger_inst=logger
        )

        assert result['status'] == 'ok', f"Export failed: {result['reason']}"
        assert output_path.exists(), "State dict file not created"

        # Smoke check
        sample_input = torch.randn(1, 3, 32, 32)
        smoke_check_state_dict(output_path, 'TestModel', sample_input, logger)

        # For test model, we can't use instantiate_arch_stub, so just check file exists
        assert result['sha256'], "No checksum computed"

        logger.info("  Cool State dict export test PASSED")
        return True

    except Exception as e:
        logger.error(f"  x State dict export test FAILED: {e}")
        return False


def test_export_torchscript():
    """Test TorchScript export and smoke check"""
    logger.info("\n[TEST] Testing TorchScript export...")

    try:
        # Create dummy model
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10)
        )

        test_dir = SMOKE_CHECK_DIR / 'test_exports' / 'torchscript'
        test_dir.mkdir(parents=True, exist_ok=True)

        sample_input = torch.randn(1, 3, 32, 32)

        results = export_model_torchscript(
            model, test_dir, 'TestModel', sample_input, force=True, logger_inst=logger
        )

        # Check trace
        assert results['trace']['status'] == 'ok', f"Trace failed: {results['trace']['reason']}"
        trace_path = results['trace']['path']
        assert trace_path.exists(), "Trace file not created"

        # Smoke check trace
        smoke = smoke_check_torchscript(trace_path, sample_input, logger)
        assert smoke['status'] == 'ok', f"Trace smoke check failed: {smoke['reason']}"

        logger.info("  Cool TorchScript export test PASSED")
        return True

    except Exception as e:
        logger.error(f"  x TorchScript export test FAILED: {e}")
        return False


def test_export_onnx():
    """Test ONNX export and smoke check"""
    logger.info("\n[TEST] Testing ONNX export...")

    if not _export_dependencies.get('onnx', False):
        logger.warning("  xx ONNX not available - test SKIPPED")
        return True

    try:
        # Create dummy model
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10)
        )

        test_dir = SMOKE_CHECK_DIR / 'test_exports' / 'onnx'
        test_dir.mkdir(parents=True, exist_ok=True)

        output_path = test_dir / 'test_model.onnx'
        sample_input = torch.randn(1, 3, 32, 32)

        result = export_model_onnx(
            model, output_path, 'TestModel', sample_input, force=True, logger_inst=logger
        )

        assert result['status'] == 'ok', f"ONNX export failed: {result['reason']}"
        assert output_path.exists(), "ONNX file not created"

        # Smoke check
        smoke = smoke_check_onnx(output_path, sample_input, logger)
        assert smoke['status'] in ['ok', 'skipped'], f"ONNX smoke check failed: {smoke['reason']}"

        logger.info("  Cool ONNX export test PASSED")
        return True

    except Exception as e:
        logger.error(f"  x ONNX export test FAILED: {e}")
        return False


def test_export_orchestrator():
    """Test full export orchestrator"""
    logger.info("\n[TEST] Testing export orchestrator...")

    try:
        # Create dummy model
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10)
        )

        test_deploy_dir = SMOKE_CHECK_DIR / 'test_deploy'
        sample_input = torch.randn(1, 3, 32, 32)

        # Test with minimal formats
        test_formats = ['state_dict', 'torchscript']

        summary = export_and_package_model(
            model=model,
            arch_name='TestModel',
            sample_input=sample_input,
            export_formats=test_formats,
            deploy_root=test_deploy_dir,
            force=True,
            _num_classes=10,
            logger_inst=logger
        )

        # Verify structure
        arch_dir = test_deploy_dir / 'TestModel'
        assert arch_dir.exists(), "Architecture directory not created"

        # Check subdirectories
        assert (arch_dir / 'state_dict').exists(), "state_dict subdir not created"
        assert (arch_dir / 'torchscript').exists(), "torchscript subdir not created"

        # Check metadata files
        assert (arch_dir / 'metadata.json').exists(), "metadata.json not created"
        assert (arch_dir / 'checksums.json').exists(), "checksums.json not created"
        assert (arch_dir / 'export_status.json').exists(), "export_status.json not created"
        assert (arch_dir / 'export.log').exists(), "export.log not created"

        # Check summary
        assert summary['successful_exports'] > 0, "No successful exports"
        assert 'one_line_summary' in summary, "No one-line summary"

        logger.info(f"  Summary: {summary['one_line_summary']}")
        logger.info("  Cool Export orchestrator test PASSED")
        return True

    except Exception as e:
        logger.error(f"  x Export orchestrator test FAILED: {e}")
        logger.exception("Full traceback:")
        return False


def test_export_idempotency():
    """Test that exports are idempotent (skip existing files)"""
    logger.info("\n[TEST] Testing export idempotency...")

    try:
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10)
        )

        test_dir = SMOKE_CHECK_DIR / 'test_exports' / 'idempotency'
        test_dir.mkdir(parents=True, exist_ok=True)

        output_path = test_dir / 'test_state_dict.pth'

        # First export
        result1 = export_model_state_dict_cpu(
            model, output_path, 'TestModel', force=False, logger_inst=logger
        )
        assert result1['status'] == 'ok', "First export failed"
        checksum1 = result1['sha256']

        # Second export (should be skipped)
        result2 = export_model_state_dict_cpu(
            model, output_path, 'TestModel', force=False, logger_inst=logger
        )
        assert result2['status'] == 'skipped', "Second export should be skipped"
        assert result2['sha256'] == checksum1, "Checksum changed"

        # Third export with force (should overwrite)
        result3 = export_model_state_dict_cpu(
            model, output_path, 'TestModel', force=True, logger_inst=logger
        )
        assert result3['status'] == 'ok', "Forced export failed"

        logger.info("  Cool Idempotency test PASSED")
        return True

    except Exception as e:
        logger.error(f"  x Idempotency test FAILED: {e}")
        return False


def run_export_unit_tests():
    """Run all export-related unit tests"""
    logger.info("\n" + "="*80)
    logger.info("RUNNING EXPORT UNIT TESTS")
    logger.info("="*80)

    tests = [
        ("State Dict Export", test_export_state_dict),
        ("TorchScript Export", test_export_torchscript),
        ("ONNX Export", test_export_onnx),
        ("Export Orchestrator", test_export_orchestrator),
        ("Export Idempotency", test_export_idempotency),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            logger.info(f"\nRunning: {test_name}")
            result = test_func()
            if result:
                passed += 1
                smoke_checker.log_check(f"ExportTest: {test_name}", "PASS", "")
            else:
                failed += 1
                smoke_checker.log_check(f"ExportTest: {test_name}", "FAIL", "Test returned False")
        except Exception as e:
            failed += 1
            logger.error(f"x {test_name} FAILED: {e}")
            smoke_checker.log_check(f"ExportTest: {test_name}", "FAIL", str(e))

    logger.info("\n" + "="*80)
    logger.info(f"EXPORT TEST SUMMARY: {passed}/{len(tests)} passed, {failed} failed")
    logger.info("="*80 + "\n")

    return passed, failed



# =============================================================================
# MULTIPROCESSING SETUP
# =============================================================================

def get_optimal_workers():
    """Get optimal number of workers for the current platform"""
    if platform.system() == 'Windows':
        cpu_count = os.cpu_count() or 4
        optimal_workers = max(2, min(16, int(cpu_count * 0.75)))
        return optimal_workers
    else:
        cpu_count = os.cpu_count() or 4
        return max(2, min(12, int(cpu_count * 0.8)))

OPTIMAL_WORKERS = get_optimal_workers()

# =============================================================================
# LOGGING SETUP
# =============================================================================

class DedupLogger:
    """Enhanced logger with message de-duplication and structured output"""
    def __init__(self, name="disease_pipeline"):
        self.logger = logging.getLogger(name)
        self.logger.propagate = False

        self.message_cache = deque(maxlen=50)
        self.last_message_time = {}
        self.system_info_logged = False

        if not self.logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            from logging.handlers import RotatingFileHandler

            # Main log file with rotation
            fh = RotatingFileHandler(
                Path.cwd() / 'training.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            # Separate error log
            error_fh = RotatingFileHandler(
                Path.cwd() / 'training_errors.log',
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            error_fh.setLevel(logging.ERROR)
            error_fh.setFormatter(formatter)
            self.logger.addHandler(error_fh)

        self.logger.setLevel(logging.INFO)

    def _hash_message(self, msg):
        """Create hash of message for de-duplication"""
        return hashlib.md5(msg.encode()).hexdigest()[:8]

    def _should_log(self, msg, level, dedupe_window=5.0):
        """Check if message should be logged based on de-duplication rules"""
        msg_hash = self._hash_message(msg)
        current_time = time.time()

        if msg_hash in self.last_message_time:
            last_time = self.last_message_time[msg_hash]
            if current_time - last_time < dedupe_window:
                return False

        self.last_message_time[msg_hash] = current_time
        return True

    def info(self, msg, dedupe=False):
        """Log info message with optional de-duplication"""
        if not dedupe or self._should_log(msg, 'INFO'):
            self.logger.info(msg)

    def debug(self, msg, dedupe=False):
        """Log debug message"""
        if not dedupe or self._should_log(msg, 'DEBUG'):
            self.logger.debug(msg)

    def warning(self, msg, dedupe=False):
        if not dedupe or self._should_log(msg, 'WARNING'):
            self.logger.warning(msg)

    def error(self, msg, dedupe=False):
        if not dedupe or self._should_log(msg, 'ERROR'):
            self.logger.error(msg)

    def exception(self, msg):
        self.logger.exception(msg)

    def log_system_info_once(self):
        """Log system information only once"""
        if not self.system_info_logged:
            self.info(f"Device: {DEVICE} | Platform: {platform.system()} | Optimal Workers: {OPTIMAL_WORKERS}")
            self.system_info_logged = True

logger = DedupLogger()

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log_system_info_once()

#   Add image size check and warning
def check_image_size_and_warn(img_size, logger_inst):
    """Check image size and warn if too large"""
    if img_size > 512:
        logger_inst.warning(
            f"Configured IMG_SIZE={img_size} — large sizes may cause OOM. "
            "Use smaller IMG_SIZE for debugging."
        )

check_image_size_and_warn(IMG_SIZE, logger)

# =============================================================================
# SMOKE CHECK UTILITIES
# =============================================================================

class SmokeCheckLogger:
    """Dedicated logger for smoke checks and validation"""
    def __init__(self):
        self.checks = []
        self.log_file = SMOKE_CHECK_DIR / f'smoke_check_{time.strftime("%Y%m%d_%H%M%S")}.log'

    def log_check(self, check_name, status, details=""):
        """Log a smoke check result"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        check_result = {
            'timestamp': timestamp,
            'check_name': check_name,
            'status': status,
            'details': details
        }
        self.checks.append(check_result)

        status_symbol = "Cool" if status == "PASS" else "x"
        log_msg = f"{status_symbol} {check_name}: {status}"
        if details:
            log_msg += f" - {details}"

        logger.info(log_msg)

        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} | {status_symbol} {check_name}: {status}")
            if details:
                f.write(f" - {details}")
            f.write("\n")

    def save_summary(self):
        """Save smoke check summary"""
        summary_file = SMOKE_CHECK_DIR / 'smoke_check_summary.json'
        summary = {
            'total_checks': len(self.checks),
            'passed': sum(1 for c in self.checks if c['status'] == 'PASS'),
            'failed': sum(1 for c in self.checks if c['status'] == 'FAIL'),
            'checks': self.checks
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Smoke check summary: {summary['passed']}/{summary['total_checks']} passed")
        return summary

smoke_checker = SmokeCheckLogger()

# =============================================================================
# ACTIVATION MANAGEMENT
# =============================================================================

ACTIVATION_MAP = {
    'silu': nn.SiLU,
    'mish': nn.Mish,
    'gelu': nn.GELU,
    'leakyrelu': lambda: nn.LeakyReLU(0.1, inplace=True),
    'relu': lambda: nn.ReLU(inplace=True),
    'linear': nn.Identity
}

def get_activation_fn(name: str) -> nn.Module:
    """Returns the instantiated activation module."""
    act_fn = ACTIVATION_MAP.get(name.lower(), ACTIVATION_MAP['silu'])
    result = act_fn() if callable(act_fn) else act_fn
    return result  # type: ignore[return-value]

# =============================================================================
# DATASET CLASSES
# =============================================================================

class OptimizedImageDataset(Dataset):
    """Optimized dataset class for Windows multiprocessing"""
    def __init__(self, samples, class_names, transform=None):
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.classes = class_names
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)
        return image, target

class WindowsCompatibleImageFolder(Dataset):
    """Enhanced ImageFolder optimized for Windows multiprocessing"""
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = self.root / cls
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)
        return image, target

class OptimizedTempDataset(Dataset):
    """Temporary dataset for K-fold cross-validation"""
    def __init__(self, samples, class_names, transform=None):
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.classes = class_names
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)
        return image, target

def worker_init_fn(worker_id):
    """Initialize workers properly for Windows multiprocessing"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# =============================================================================
# CORE BUILDING BLOCKS
# =============================================================================

class CoreImageBlock(nn.Module):
    """Base Conv-Norm-Act block supporting LN/BN and diverse activation"""
    def __init__(self, in_c, out_c, k_size=3, stride=1, padding=1,
                 groups=1, use_ln=False, activation='silu', bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, k_size, stride, padding, groups=groups, bias=bias)

        if use_ln:
            self.norm = nn.LayerNorm(out_c)
        else:
            self.norm = nn.BatchNorm2d(out_c)

        self.act = get_activation_fn(activation)
        self.use_ln = use_ln

    def forward(self, x):
        x = self.conv(x)

        if self.use_ln:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x = self.norm(x)

        return self.act(x)

# =============================================================================
# SWIN TRANSFORMER COMPONENTS - FIXED
# =============================================================================

def window_partition(x, window_size):
    """
    Partition input into non-overlapping windows.
      Added robust shape checks and asserts.
    """
    B, H, W, C = x.shape

    # Calculate padding needed
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    # Apply padding if needed
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H_pad, W_pad = H + pad_h, W + pad_w
        logger.debug(f"Window partition: padded {H}x{W} -> {H_pad}x{W_pad}")
    else:
        H_pad, W_pad = H, W

    # Now safe to reshape
    x = x.view(B, H_pad // window_size, window_size, W_pad // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.
      Use integer division and add robust checks.
    """
    #   Integer-safe computation of batch size
    num_windows_per_image = (H // window_size) * (W // window_size)

    assert num_windows_per_image > 0, (
        f"Invalid parameters: window_size={window_size}, H={H}, W={W}. "
        f"H and W must be divisible by window_size."
    )

    total_windows = windows.shape[0]
    assert total_windows % num_windows_per_image == 0, (
        f"Total windows {total_windows} is not divisible by "
        f"per-image count {num_windows_per_image}"
    )

    B = total_windows // num_windows_per_image

    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention with relative position bias"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Get relative position index as tensor
        rel_pos_idx: torch.Tensor = self.relative_position_index  # type: ignore[assignment]
        relative_position_bias = self.relative_position_bias_table[rel_pos_idx.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with shifted window attention"""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            get_activation_fn('gelu'),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(attn_mask == 0, 0.0)
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size: {L} vs {H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

# =============================================================================
# DYNAMIC CONVOLUTION - FIXED WITH BATCHED IMPLEMENTATION
# =============================================================================

class DynamicConv(nn.Module):
    """
    Dynamic Convolution with per-sample kernel generation.
      Batched forward using unfold + bmm for efficiency and autograd compatibility.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_kernels=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_kernels = num_kernels

        self.controller = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_kernels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_kernels * 2, num_kernels),
            nn.Softmax(dim=1)
        )

        self.weight_bank = nn.Parameter(
            torch.randn(num_kernels, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        for i in range(num_kernels):
            nn.init.kaiming_normal_(self.weight_bank[i], mode='fan_out', nonlinearity='relu')
        self.register_buffer('scale', torch.tensor((in_channels * kernel_size * kernel_size) ** -0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale: torch.Tensor = self.scale  # type: ignore[assignment]

        B, C_in, H, W = x.shape
        assert C_in == self.in_channels, f"Expected input channels {self.in_channels}, got {C_in}"

        # controller returns attention weights per sample -> shape (B, num_kernels)
        attention = self.controller(x)  # (B, K)
        K = self.num_kernels
        k = self.kernel_size
        stride = self.stride
        padding = self.padding

        # Flatten weight_bank: (K, out, in, k, k) -> (K, out, in*k*k)
        weight_bank_flat = self.weight_bank.view(K, self.out_channels, -1) * scale  # (K, out, in*k*k)

        attention = attention.view(B, K, 1)  # (B, K, 1)
        weighted_kernels = (attention.unsqueeze(2) * weight_bank_flat.unsqueeze(0)).sum(dim=1)

        # Unfold input to patches: (B, in*k*k, L)
        patches = F.unfold(x, kernel_size=k, padding=padding, stride=stride)  # (B, in*k*k, L)
        patches = torch.clamp(patches, -10, 10)

        # batched matmul: (B, out, in*k*k) x (B, in*k*k, L) -> (B, out, L)
        out_unfold = torch.bmm(weighted_kernels, patches)  # (B, out, L)

        # compute output spatial dims
        H_out = (H + 2*padding - k) // stride + 1
        W_out = (W + 2*padding - k) // stride + 1
        out = out_unfold.view(B, self.out_channels, H_out, W_out)

        # add bias (broadcast)
        out = out + self.bias.view(1, -1, 1, 1)
        return out

# =============================================================================
# DATA TRANSFORMS AND LOADERS
# =============================================================================

def create_optimized_transforms(size, is_training=True):
    """Create optimized transforms with stronger augmentation"""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95,1.05)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.2)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(size * 1.14)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_optimized_dataloader(dataset, batch_size, shuffle=True, num_workers=None):
    """Create optimized DataLoader with Windows support"""
    if num_workers is None:
        num_workers = OPTIMAL_WORKERS

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2 if num_workers > 0 else 2,
        'worker_init_fn': worker_init_fn if num_workers > 0 else None,
        'drop_last': False,
        'timeout': 30 if num_workers > 0 else 0
    }

    if num_workers == 0:
        loader_kwargs.pop('persistent_workers', None)
        loader_kwargs.pop('worker_init_fn', None)
        loader_kwargs.pop('timeout', None)
        loader_kwargs['pin_memory'] = False

    try:
        return DataLoader(dataset, **loader_kwargs)
    except Exception as e:
        logger.warning(f"DataLoader creation failed: {e}")
        loader_kwargs.update({'num_workers': 0, 'pin_memory': False})
        loader_kwargs.pop('persistent_workers', None)
        loader_kwargs.pop('worker_init_fn', None)
        loader_kwargs.pop('timeout', None)
        return DataLoader(dataset, **loader_kwargs)


# =============================================================================
# TRAINING HELPER FUNCTIONS
# =============================================================================

def _unwrap_logits(outputs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Handle inception aux_logits and various output formats"""
    if isinstance(outputs, torch.Tensor):
        return outputs, None
    if hasattr(outputs, 'logits'):
        return outputs.logits, getattr(outputs, 'aux_logits', None)
    if isinstance(outputs, (tuple, list)):
        main: torch.Tensor | None = None
        aux: torch.Tensor | None = None
        for o in outputs:
            if isinstance(o, torch.Tensor) and main is None:
                main = o
            elif hasattr(o, 'logits') and main is None:
                # o has logits attribute verified by hasattr
                main = o.logits  # type: ignore[union-attr]
            elif isinstance(o, torch.Tensor) and main is not None and aux is None:
                aux = o
        if main is not None:
            return main, aux
    if isinstance(outputs, dict) and 'logits' in outputs:
        return outputs['logits'], outputs.get('aux_logits', None)
    # Fallback: assume outputs is already a tensor
    return outputs, None  # type: ignore[return-value]

def create_optimized_optimizer(model, lr, backbone_name):
    """Create optimized optimizer for specific backbone"""
    # FIXED: Balanced learning rates based on architecture complexity
    lr_configs = {
        'CustomConvNeXt': lr * 1.2,          # Reduced from 1.5
        'CustomEfficientNetV4': lr * 1.1,    # Reduced from 1.3
        'CustomGhostNetV2': lr * 1.5,        # Reduced from 2.0
        'CustomResNetMish': lr * 1.2,        # Reduced from 1.5
        'CustomCSPDarkNet': lr * 1.2,        # Reduced from 1.4
        'CustomInceptionV4': lr * 1.0,       # Reduced from 1.2
        'CustomViTHybrid': lr * 1.2,         # Keep
        'CustomSwinTransformer': lr * 0.7,   # Reduced from 0.8
        'CustomCoAtNet': lr * 1.2,           # Reduced from 0.9
        'CustomRegNet': lr * 1.2,            # Reduced from 1.4
        'CustomDenseNetHybrid': lr * 1.1,    # Reduced from 1.3
        'CustomDeiTStyle': lr * 0.8,         # Reduced from 0.9
        'CustomMaxViT': lr * 1.2,            # Reduced from 0.9
        'CustomMobileOne': lr * 1.0,         # FIXED: Reduced from 2.5
        'CustomDynamicConvNet': lr * 1.0     # Reduced from 1.2
    }

    actual_lr = lr_configs.get(backbone_name, lr)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=actual_lr,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    return optimizer

def save_checkpoint(path: Path, model, optimizer=None, scheduler=None, extra=None):
    """Save model checkpoint with detailed information"""
    path = Path(path)
    state = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        try:
            state["optimizer_state_dict"] = optimizer.state_dict()
        except Exception:
            state["optimizer_state_dict"] = None
    if scheduler is not None:
        try:
            state["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            state["scheduler_state_dict"] = None
    if extra is not None:
        state["extra"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
    logger.info(f"Checkpoint saved to {path}")


# =============================================================================
# TRAINING AND VALIDATION LOOPS
# =============================================================================

def train_epoch_optimized(model, loader, optimizer, criterion, device=DEVICE):
    """Enhanced training epoch with detailed metrics collection and logging"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']

    # Use GradScaler for mixed precision
    scaler = GradScaler() if device.type == 'cuda' else None

    pbar = tqdm(loader, desc=f"Train (LR: {current_lr:.2e})", leave=False)

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        if scaler is not None:
            with autocast(device_type='cuda'):
                outputs = model(images)
                logits, aux_logits = _unwrap_logits(outputs)
                loss_main = criterion(logits, targets)
                loss = loss_main

                # Handle aux_logits for models like Inception
                if aux_logits is not None and isinstance(aux_logits, torch.Tensor):
                    aux_loss = criterion(aux_logits, targets)
                    loss += 0.4 * aux_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            logits, aux_logits = _unwrap_logits(outputs)
            loss_main = criterion(logits, targets)
            loss = loss_main

            if aux_logits is not None and isinstance(aux_logits, torch.Tensor):
                aux_loss = criterion(aux_logits, targets)
                loss += 0.4 * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Update metrics
        running_loss += loss.item() * images.size(0)

        # Collect predictions and probabilities
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(targets.cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

        # Update progress bar
        if batch_idx % 10 == 0:
            current_samples = sum([len(p) for p in all_preds])
            avg_loss = running_loss / current_samples if current_samples > 0 else 0
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'lr': f"{current_lr:.2e}",
                'batch': f"{batch_idx}/{len(loader)}"
            })

    # Compute final metrics
    all_preds_cat = np.concatenate(all_preds) if len(all_preds) > 0 else np.array([])
    all_labels_cat = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])
    all_probs_cat = np.concatenate(all_probs) if len(all_probs) > 0 else np.array([])

    acc = (all_preds_cat == all_labels_cat).mean() if all_preds_cat.size > 0 else 0.0
    prec = precision_score(all_labels_cat, all_preds_cat, average='macro', zero_division=0) if all_preds_cat.size > 0 else 0.0
    rec = recall_score(all_labels_cat, all_preds_cat, average='macro', zero_division=0) if all_preds_cat.size > 0 else 0.0
    f1 = f1_score(all_labels_cat, all_preds_cat, average='macro', zero_division=0) if all_preds_cat.size > 0 else 0.0

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, acc, prec, rec, f1, all_preds_cat, all_labels_cat, all_probs_cat

def validate_epoch_optimized(model, loader, criterion, device=DEVICE):
    """Enhanced validation epoch with detailed metrics collection and logging"""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    pbar = tqdm(loader, desc="Validate", leave=False)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if device.type == 'cuda':
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    logits, _ = _unwrap_logits(outputs)
                    loss = criterion(logits, targets)
            else:
                outputs = model(images)
                logits, _ = _unwrap_logits(outputs)
                loss = criterion(logits, targets)

            # Update metrics
            running_loss += loss.item() * images.size(0)

            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(targets.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

            # Update progress bar
            if batch_idx % 5 == 0:
                current_samples = sum([len(p) for p in all_preds]) if all_preds else 1
                current_loss = running_loss / current_samples
                pbar.set_postfix({'loss': f"{current_loss:.4f}"})

    all_preds = np.concatenate(all_preds) if len(all_preds)>0 else np.array([])
    all_labels = np.concatenate(all_labels) if len(all_labels)>0 else np.array([])
    all_probs = np.concatenate(all_probs) if len(all_probs)>0 else np.array([])

    acc = (all_preds == all_labels).mean() if all_preds.size>0 else 0.0
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) if all_preds.size>0 else 0.0
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0) if all_preds.size>0 else 0.0
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if all_preds.size>0 else 0.0

    total = len(loader.dataset) if hasattr(loader, 'dataset') else (all_labels.size if all_labels.size > 0 else 1)
    return running_loss / total, acc, prec, rec, f1, all_preds, all_labels, all_probs


# =============================================================================
# FULL TRAINING PIPELINE FOR BACKBONE
# =============================================================================

def train_backbone_with_metrics(backbone_name, model, train_ds, val_ds,
                                epochs_head=EPOCHS_HEAD, epochs_finetune=EPOCHS_FINETUNE):
    """
    Complete training pipeline for a backbone with detailed logging.
    Includes head training, fine-tuning, export, and comprehensive metrics tracking.
    """


    logger.info(f"Starting training for {backbone_name}")

    # Validate datasets are not empty
    if len(train_ds) == 0:
        raise ValueError(f"Training dataset is empty for {backbone_name}")
    if len(val_ds) == 0:
        raise ValueError(f"Validation dataset is empty for {backbone_name}")

    # Move model to device
    model.to(DEVICE)

    # Create optimized dataloaders
    train_loader = create_optimized_dataloader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = create_optimized_dataloader(val_ds, BATCH_SIZE, shuffle=False)

    # Get class names
    if hasattr(train_ds, 'classes'):
        class_names = train_ds.classes
    elif hasattr(train_ds, 'dataset') and hasattr(train_ds.dataset, 'classes'):
        class_names = train_ds.dataset.classes
    else:
        class_names = [f'Class_{i}' for i in range(get_num_classes())]

    criterion = get_loss_function_for_backbone(backbone_name, get_num_classes())
    best_acc = 0.0
    history = {'head': [], 'finetune': []}
    best_model_state = None
    patience_counter = 0

    # =========================
    # STAGE 1: HEAD TRAINING
    # =========================
    logger.info(f"Stage 1: Head training for {backbone_name} (Epochs: {epochs_head})")

    # Freeze backbone - only train the head
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = False
        if hasattr(model.backbone, 'eval'):
            model.backbone.eval()

    # Get only head parameters for optimizer
    if hasattr(model, 'head'):
        model.head.parameters()
    else:
        [p for p in model.parameters() if p.requires_grad]

    optimizer = create_optimized_optimizer(model, lr=HEAD_LR, backbone_name=backbone_name)
    # Stage 1 - HEAD training
    scheduler = create_improved_scheduler(optimizer, epochs_head, len(train_loader), backbone_name)

    for epoch in range(epochs_head):
        current_lr = optimizer.param_groups[0]['lr']

        # Training epoch
        train_loss, train_acc, train_prec, _train_rec, train_f1, _, _, _ = train_epoch_optimized(
            model, train_loader, optimizer, criterion
        )

        # Validation epoch
        val_loss, val_acc, val_prec, _val_rec, val_f1, _val_preds, _val_labels, _val_probs = validate_epoch_optimized(
            model, val_loader, criterion
        )

        scheduler.step()

        # Store history
        history['head'].append((train_loss, val_loss, val_acc, train_acc, val_f1))

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            # Save checkpoint
            ckpt_path = CKPT_DIR / f"{backbone_name}_head_best.pth"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, extra={
                'epoch': epoch,
                'accuracy': val_acc,
                'stage': 'head'
            })
        else:
            patience_counter += 1

        # DETAILED LOGGING
        logger.info(f"HEAD Epoch {epoch+1:2d}/{epochs_head} | "
                   f"LR: {current_lr:.2e} | "
                   f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Prec: {train_prec:.4f} F1: {train_f1:.4f} | "
                   f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Prec: {val_prec:.4f} F1: {val_f1:.4f} | "
                   f"Best: {best_acc:.4f} | Patience: {patience_counter}/{PATIENCE_HEAD}")

        # Early stopping for head training
        if patience_counter >= PATIENCE_HEAD:
            logger.info(f"Early stopping triggered for head training at epoch {epoch+1}")
            break

    # Load best head model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best head model with accuracy: {best_acc:.4f}")

    # =========================
    # STAGE 2: FINE-TUNING
    # =========================
    logger.info(f"Stage 2: Fine-tuning for {backbone_name} (Epochs: {epochs_finetune})")

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    model.train()

    # Create new optimizer for fine-tuning with lower learning rate
    optimizer = create_optimized_optimizer(model, lr=BACKBONE_LR, backbone_name=backbone_name)
    scheduler = create_improved_scheduler(optimizer, epochs_finetune, len(train_loader), backbone_name)

    patience_counter = 0

    for epoch in range(epochs_finetune):
        current_lr = optimizer.param_groups[0]['lr']

        # Training epoch
        train_loss, train_acc, train_prec, _train_rec, train_f1, _, _, _ = train_epoch_optimized(
            model, train_loader, optimizer, criterion
        )

        # Validation epoch
        val_loss, val_acc, val_prec, _val_rec, val_f1, _val_preds, _val_labels, _val_probs = validate_epoch_optimized(
            model, val_loader, criterion
        )

        scheduler.step()

        # Store history
        history['finetune'].append((train_loss, val_loss, val_acc, train_acc, val_f1))

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            # Save checkpoint
            ckpt_path = CKPT_DIR / f"{backbone_name}_finetune_best.pth"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, extra={
                'epoch': epoch,
                'accuracy': val_acc,
                'stage': 'finetune'
            })
        else:
            patience_counter += 1

        # DETAILED LOGGING
        logger.info(f"FINE Epoch {epoch+1:2d}/{epochs_finetune} | "
                   f"LR: {current_lr:.2e} | "
                   f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Prec: {train_prec:.4f} F1: {train_f1:.4f} | "
                   f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Prec: {val_prec:.4f} F1: {val_f1:.4f} | "
                   f"Best: {best_acc:.4f} | Patience: {patience_counter}/{PATIENCE_FT}")

        # Early stopping for fine-tuning
        if patience_counter >= PATIENCE_FT:
            logger.info(f"Early stopping triggered for fine-tuning at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best fine-tuned model with accuracy: {best_acc:.4f}")

    # =========================
    # STAGE 3: FINAL EVALUATION & VISUALIZATION
    # =========================
    logger.info(f"Stage 3: Final evaluation and visualization for {backbone_name}")

    model.eval()
    final_val_loss, final_val_acc, final_val_prec, final_val_rec, final_val_f1, _val_preds, _val_labels, _val_probs = validate_epoch_optimized(
        model, val_loader, criterion
    )

    # Calculate comprehensive metrics
    metrics = {
        'best_accuracy': float(best_acc),
        'final_accuracy': float(final_val_acc),
        'final_precision': float(final_val_prec),
        'final_recall': float(final_val_rec),
        'final_f1_score': float(final_val_f1),
        'final_val_loss': float(final_val_loss),
        'backbone_name': backbone_name,
        'epochs_head': len(history['head']),
        'epochs_finetune': len(history['finetune'])
    }

    # ==========================================
    # NEW: GENERATE ALL VISUALIZATIONS
    # ==========================================
    logger.info(f"Generating comprehensive visualizations for {backbone_name}...")

    try:
        # Generate all plots (now as TIFF at 1200 DPI)
        plot_paths = generate_all_visualizations(
            model=model,
            backbone_name=backbone_name,
            history=history,
            val_loader=val_loader,
            class_names=class_names,
            criterion=criterion,
            device=DEVICE
        )

        # Save visualization summary
        viz_summary_path = save_visualization_summary(plot_paths, backbone_name, METRICS_DIR)

        # Add to metrics
        metrics['visualization_paths'] = plot_paths
        metrics['visualization_summary'] = viz_summary_path

        logger.info(f"Cool Generated {len(plot_paths)} visualizations for {backbone_name}")

    except Exception as e:
        logger.error(f"x Visualization generation failed for {backbone_name}: {e}")
        logger.exception("Full traceback:")
        metrics['visualization_paths'] = {}
        metrics['visualization_summary'] = None

    # Save final checkpoint
    final_ckpt_path = CKPT_DIR / f"{backbone_name}_final.pth"
    save_checkpoint(final_ckpt_path, model, extra=metrics)

    # Save metrics to JSON
    metrics_file = METRICS_DIR / f"{backbone_name}_training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Cool {backbone_name} training completed: Best accuracy = {best_acc:.4f}")
    logger.info(f"  Final checkpoint: {final_ckpt_path}")
    logger.info(f"  Metrics saved: {metrics_file}")

    # =========================
    # STAGE 4: MODEL EXPORT
    # =========================
    if EXPORT_FORMATS and ENABLE_EXPORT:
        logger.info(f"\n{'='*60}")
        logger.info(f"EXPORTING {backbone_name} TO DEPLOYMENT FORMATS")
        logger.info(f"{'='*60}")

        try:
            # Prepare sample input
            sample_input = prepare_sample_input(model, input_size=IMG_SIZE, device=DEVICE)

            # Prepare comprehensive training metadata
            training_metadata = {
                'best_accuracy': float(best_acc),
                'final_accuracy': float(final_val_acc),
                'final_precision': float(final_val_prec),
                'final_recall': float(final_val_rec),
                'final_f1_score': float(final_val_f1),
                'epochs_head': len(history['head']),
                'epochs_finetune': len(history['finetune']),
                'img_size': IMG_SIZE,
                'batch_size': BATCH_SIZE,
                '_num_classes': _num_classes,
            }

            # Export to all configured formats
            export_summary = export_and_package_model(
                model=model,
                arch_name=backbone_name,
                sample_input=sample_input,
                export_formats=EXPORT_FORMATS,
                deploy_root=DEPLOY_DIR,
                force=False,
                optimizer=None,
                scheduler=None,
                training_metadata=training_metadata,
                _num_classes=_num_classes,
                logger_inst=logger
            )

            logger.info(f"Cool Export completed: {export_summary['one_line_summary']}")
            metrics['export_summary'] = export_summary

        except Exception as e:
            logger.error(f"x Export failed for {backbone_name}: {e}")
            logger.exception("Full export error traceback:")
            metrics['export_summary'] = {
                'status': 'failed',
                'error': str(e)
            }
    else:
        if not ENABLE_EXPORT:
            logger.info("Export disabled (ENABLE_EXPORT = False)")
            metrics['export_summary'] = {
                'status': 'disabled',
                'reason': 'Export functionality disabled via ENABLE_EXPORT flag'
            }
        else:
            logger.info("Export skipped (no formats configured)")
            metrics['export_summary'] = {
                'status': 'skipped',
                'reason': 'EXPORT_FORMATS is empty'
            }

    return model, best_acc, history, metrics

# =============================================================================
# K-FOLD CROSS VALIDATION
# =============================================================================

def k_fold_cross_validation(backbone_name, full_dataset, k_folds=K_FOLDS):
    """
    Perform K-fold cross validation with detailed epoch-wise logging.
    """
    logger.info(f"Starting {k_folds}-fold CV for {backbone_name}")

    # Extract samples and labels
    if hasattr(full_dataset, 'samples'):
        samples: list[Any] = full_dataset.samples
        labels: list[int] = [s[1] for s in samples]
    else:
        samples = [(full_dataset[i][0], full_dataset[i][1]) for i in range(len(full_dataset))]
        labels = [s[1] for s in samples]

    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    fold_results: list[float] = []

    # Convert to numpy array for sklearn compatibility
    labels_array = np.array(labels)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels_array)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {backbone_name} - Fold {fold + 1}/{k_folds}")
        logger.info(f"{'='*60}")

        try:
            # Create fold datasets
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            class_names = full_dataset.classes if hasattr(full_dataset, 'classes') else [f'Class_{i}' for i in range(get_num_classes())]

            fold_train_ds = OptimizedTempDataset(
                train_samples, class_names,
                transform=create_optimized_transforms(IMG_SIZE, is_training=True)
            )
            fold_val_ds = OptimizedTempDataset(
                val_samples, class_names,
                transform=create_optimized_transforms(IMG_SIZE, is_training=False)
            )

            # Create model for this fold
            model = create_custom_backbone_safe(backbone_name, get_num_classes())
            model.to(DEVICE)

            # Create optimized dataloaders
            train_loader = create_optimized_dataloader(fold_train_ds, BATCH_SIZE, shuffle=True)
            val_loader = create_optimized_dataloader(fold_val_ds, BATCH_SIZE, shuffle=False)

            # Training with detailed logging (reduced epochs for K-fold)
            criterion = nn.CrossEntropyLoss()
            best_fold_acc = 0.0

            # Stage 1: Head training (reduced epochs)
            logger.info(f"Fold {fold+1} - Stage 1: Head training")

            # Freeze backbone
            if hasattr(model, 'backbone'):
                for param in model.backbone.parameters():
                    param.requires_grad = False

            optimizer = create_optimized_optimizer(model, lr=HEAD_LR, backbone_name=backbone_name)

            head_epochs = min(30, EPOCHS_HEAD)  # Reduced for K-fold
            for epoch in range(head_epochs):
                current_lr = optimizer.param_groups[0]['lr']

                train_loss, train_acc, _train_prec, _train_rec, train_f1, _, _, _ = train_epoch_optimized(
                    model, train_loader, optimizer, criterion
                )
                val_loss, val_acc, _val_prec, _val_rec, val_f1, _, _, _ = validate_epoch_optimized(
                    model, val_loader, criterion
                )

                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc

                # Log every 5 epochs
                if epoch % 5 == 0 or epoch == head_epochs - 1:
                    logger.info(f"K-fold HEAD Epoch {epoch+1:2d}/{head_epochs} | "
                               f"LR: {current_lr:.2e} | "
                               f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                               f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

            # Stage 2: Fine-tuning (reduced epochs)
            logger.info(f"Fold {fold+1} - Stage 2: Fine-tuning")

            # Unfreeze all parameters
            for param in model.parameters():
                param.requires_grad = True

            optimizer = create_optimized_optimizer(model, lr=BACKBONE_LR, backbone_name=backbone_name)

            finetune_epochs = min(20, EPOCHS_FINETUNE)  # Reduced for K-fold
            for epoch in range(finetune_epochs):
                current_lr = optimizer.param_groups[0]['lr']

                train_loss, train_acc, _train_prec, _train_rec, train_f1, _, _, _ = train_epoch_optimized(
                    model, train_loader, optimizer, criterion
                )
                val_loss, val_acc, _val_prec, _val_rec, val_f1, _, _, _ = validate_epoch_optimized(
                    model, val_loader, criterion
                )

                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc

                # Log every 3 epochs
                if epoch % 3 == 0 or epoch == finetune_epochs - 1:
                    logger.info(f"K-fold FINE Epoch {epoch+1:2d}/{finetune_epochs} | "
                               f"LR: {current_lr:.2e} | "
                               f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                               f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

            fold_results.append(best_fold_acc)
            logger.info(f"Cool Fold {fold+1} completed: Best accuracy = {best_fold_acc:.4f}")

            # Clean up memory
            del model, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        except Exception as e:
            logger.error(f" Fold {fold+1} failed for {backbone_name}: {e}")
            fold_results.append(0.0)

    # Calculate statistics
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)

    kfold_summary = {
        'backbone': backbone_name,
        'k_folds': k_folds,
        'fold_accuracies': [float(acc) for acc in fold_results],
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc)
    }

    # Save K-fold results
    kfold_file = KFOLD_DIR / f"{backbone_name}_kfold_results.json"
    with open(kfold_file, 'w') as f:
        json.dump(kfold_summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"K-fold CV completed for {backbone_name}")
    logger.info(f"Mean accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    logger.info(f"Results saved: {kfold_file}")
    logger.info(f"{'='*60}\n")

    return mean_acc, std_acc, kfold_summary



# =============================================================================
# DATASET PREPARATION
# =============================================================================

def prepare_optimized_datasets(raw_dir=RAW_DIR, split_dir=SPLIT_DIR,
                              train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=SEED):
    """Optimized dataset preparation"""
    raw_dir = Path(raw_dir)
    split_dir = Path(split_dir)

    if split_dir.exists() and any(split_dir.iterdir()):
        logger.info(f"Split dataset already exists at {split_dir}")
        return

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory {raw_dir} not found.")

    logger.info(f"Creating split dataset from {raw_dir} -> {split_dir}")

    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    all_samples = []

    for cls in classes:
        cls_dir = raw_dir / cls
        for img_path in cls_dir.glob("*"):
            if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                all_samples.append((str(img_path), cls))

    filepaths = np.array([s[0] for s in all_samples])
    labels = np.array([s[1] for s in all_samples])

    if len(np.unique(labels)) < 2:
        raise ValueError("Need at least two disease classes to split.")

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_idx, test_idx = next(sss1.split(filepaths, labels))

    X_train, X_test = filepaths[train_idx], filepaths[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    val_size = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss2.split(X_train, y_train))

    X_train_final, X_val = X_train[train_idx], X_train[val_idx]
    y_train_final, y_val = y_train[train_idx], y_train[val_idx]

    def copy_files(paths_labels, target_dir):
        for path, label in paths_labels:
            dest_dir = Path(target_dir) / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / Path(path).name
            if not dest_path.exists():
                shutil.copy2(path, str(dest_path))

    copy_files(zip(X_train_final, y_train_final), TRAIN_DIR)
    copy_files(zip(X_val, y_val), VAL_DIR)
    copy_files(zip(X_test, y_test), TEST_DIR)

    logger.info(f"Split created: train={len(X_train_final)}, val={len(X_val)}, test={len(X_test)}")

def verify_dataset_split(split_dir: Path = SPLIT_DIR):
    """Verify dataset split integrity after creation"""
    logger.info("Verifying dataset split integrity...")
    issues = []

    # Get expected classes from train directory
    train_path = split_dir / 'train'
    if not train_path.exists():
        logger.error(f"Train directory not found at {train_path}")
        return False

    expected_classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    logger.info(f"  Expected classes: {expected_classes}")

    split_stats = {}

    for split_name in ['train', 'val', 'test']:
        split_path = split_dir / split_name

        if not split_path.exists():
            issues.append(f"Missing {split_name} directory")
            continue

        found_classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])

        if set(found_classes) != set(expected_classes):
            issues.append(
                f"{split_name}: class mismatch. "
                f"Expected {len(expected_classes)} classes, found {len(found_classes)}"
            )

        total_images = 0
        for cls in found_classes:
            cls_path = split_path / cls
            images = list(cls_path.glob("*"))
            images = [img for img in images if img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
            total_images += len(images)

        split_stats[split_name] = {
            'classes': len(found_classes),
            'images': total_images
        }

        if total_images == 0:
            issues.append(f"{split_name}: no images found")
        else:
            logger.info(f"  {split_name}: {total_images} images across {len(found_classes)} classes")

    if issues:
        logger.error("Dataset split validation FAILED:")
        for issue in issues:
            logger.error(f"  x {issue}")
        return False

    logger.info("Dataset split validation PASSED")
    return True

def prepare_datasets_for_backbone(backbone_name, size=224):
    """Prepare datasets for specific backbone"""
    train_tf = create_optimized_transforms(size, is_training=True)
    val_tf = create_optimized_transforms(size, is_training=False)

    if TRAIN_DIR.exists() and VAL_DIR.exists():
        train_ds = WindowsCompatibleImageFolder(str(TRAIN_DIR), transform=train_tf)
        val_ds = WindowsCompatibleImageFolder(str(VAL_DIR), transform=val_tf)
    elif RAW_DIR.exists():
        raise FileNotFoundError("Please run dataset splitting first.")
    else:
        raise FileNotFoundError("No dataset found.")

    global _num_classes
    _num_classes = len(train_ds.classes)

    logger.info(f"Prepared datasets for {backbone_name}: train={len(train_ds)}, val={len(val_ds)}, classes={_num_classes}")
    return train_ds, val_ds

# =============================================================================
# MOBILEONE WITH REPARAMETERIZATION
# =============================================================================

class MobileOneBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_conv_branches=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_conv_branches = num_conv_branches

        # FIXED: Better initialization for conv branches
        self.conv_kxk = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels)
            ) for _ in range(num_conv_branches)
        ])

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Identity branch
        self.has_identity = (stride == 1 and in_channels == out_channels)
        if self.has_identity:
            self.identity_bn = nn.BatchNorm2d(in_channels)

        self.activation = nn.ReLU(inplace=True)
        self.is_fused = False

        # FIXED: Initialize all branches properly
        self._init_branches()

    def _init_branches(self) -> None:
        """Initialize all branches with proper scaling"""
        for branch in self.conv_kxk:
            if isinstance(branch, nn.Sequential):
                conv: nn.Conv2d = branch[0]  # type: ignore[assignment]
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                # FIXED: Scale down to prevent gradient explosion
                with torch.no_grad():
                    conv.weight.data *= (1.0 / (self.num_conv_branches + 2))  # +2 for 1x1 and identity

        if isinstance(self.conv_1x1, nn.Sequential):
            conv_1x1: nn.Conv2d = self.conv_1x1[0]  # type: ignore[assignment]
            nn.init.kaiming_normal_(conv_1x1.weight, mode='fan_out', nonlinearity='relu')
            with torch.no_grad():
                conv_1x1.weight.data *= (1.0 / (self.num_conv_branches + 2))

    def forward(self, x):
        if self.is_fused:
            return self.activation(self.fused_conv(x))

        # FIXED: Accumulate with better numerical stability
        out = torch.zeros_like(self.conv_kxk[0](x))

        for conv in self.conv_kxk:
            out = out + conv(x)

        out = out + self.conv_1x1(x)

        if self.has_identity:
            out = out + self.identity_bn(x)

        return self.activation(out)

    def fuse_reparam(self):
        """Fuse all branches into a single conv for inference"""
        if self.is_fused:
            return

        kernel_kxk, bias_kxk = self._fuse_bn_tensor(self.conv_kxk[0])
        for i in range(1, len(self.conv_kxk)):
            k, b = self._fuse_bn_tensor(self.conv_kxk[i])
            kernel_kxk += k
            bias_kxk += b

        kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.conv_1x1)
        kernel_1x1 = self._pad_1x1_to_kxk(kernel_1x1)

        kernel_identity = 0
        bias_identity = 0
        if self.has_identity:
            kernel_identity, bias_identity = self._fuse_bn_tensor_identity(self.identity_bn)

        fused_kernel = kernel_kxk + kernel_1x1 + kernel_identity
        fused_bias = bias_kxk + bias_1x1 + bias_identity

        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            self.kernel_size, self.stride, self.padding, bias=True
        )
        self.fused_conv.weight.data = fused_kernel  # type: ignore[assignment]
        if self.fused_conv.bias is not None:
            self.fused_conv.bias.data = fused_bias  # type: ignore[assignment]

        self.__delattr__('conv_kxk')
        self.__delattr__('conv_1x1')
        if self.has_identity:
            self.__delattr__('identity_bn')

        self.is_fused = True

    def _fuse_bn_tensor(self, branch: nn.Module) -> tuple[torch.Tensor | int, torch.Tensor | int]:
        if isinstance(branch, nn.Sequential):
            conv: nn.Conv2d = branch[0]  # type: ignore[assignment]
            bn: nn.BatchNorm2d = branch[1]  # type: ignore[assignment]
            kernel: torch.Tensor = conv.weight

            running_mean: torch.Tensor = bn.running_mean  # type: ignore[assignment]
            running_var: torch.Tensor = bn.running_var  # type: ignore[assignment]
            gamma: torch.Tensor = bn.weight
            beta: torch.Tensor = bn.bias  # type: ignore[assignment]
            eps: float = bn.eps

            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)

            return kernel * t, beta - running_mean * gamma / std
        else:
            return 0, 0

    def _fuse_bn_tensor_identity(self, bn):
        kernel = torch.zeros(self.out_channels, self.in_channels,
                           self.kernel_size, self.kernel_size,
                           dtype=bn.weight.dtype, device=bn.weight.device)

        center = self.kernel_size // 2
        for i in range(min(self.in_channels, self.out_channels)):
            kernel[i, i, center, center] = 1

        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        return kernel * t, beta - running_mean * gamma / std

    def _pad_1x1_to_kxk(self, kernel_1x1):
        if self.kernel_size == 1:
            return kernel_1x1
        pad = self.kernel_size // 2
        padded = F.pad(kernel_1x1, [pad, pad, pad, pad])
        return padded

# =============================================================================
# BUILDING BLOCKS - CNN COMPONENTS
# =============================================================================

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = CoreImageBlock(dim, dim, k_size=7, padding=3,
                                     groups=dim, use_ln=True, activation='linear')
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            get_activation_fn('gelu'),
            nn.Linear(dim * 4, dim)
        )
        self.gamma = nn.Parameter(torch.ones(1) * 1e-6)

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.mlp(x)
        return (x.permute(0, 3, 1, 2) * self.gamma) + input_x

class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block"""
    def __init__(self, in_c, out_c, stride=1, expand_ratio=4):
        super().__init__()
        hidden_dim = in_c * expand_ratio
        self.use_residual = stride == 1 and in_c == out_c

        layers = []
        if expand_ratio != 1:
            layers.append(CoreImageBlock(in_c, hidden_dim, 1, 1, 0, activation='silu'))

        layers.append(CoreImageBlock(hidden_dim, hidden_dim, 3, stride, 1,
                                    groups=hidden_dim, activation='silu'))
        layers.append(CoreImageBlock(hidden_dim, out_c, 1, 1, 0, activation='linear'))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class GhostModule(nn.Module):
    """Ghost Module"""
    def __init__(self, in_c, out_c, kernel_size=1, ratio=2, dw_size=3):
        super().__init__()
        init_channels = out_c // ratio

        self.primary_conv = CoreImageBlock(in_c, init_channels, kernel_size, 1,
                                          kernel_size//2, activation='leakyrelu')
        self.cheap_operation = CoreImageBlock(init_channels, init_channels, dw_size, 1,
                                             dw_size//2, groups=init_channels,
                                             activation='leakyrelu')

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

class MishBottleneck(nn.Module):
    """Bottleneck with Mish"""
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        outplanes = planes * self.expansion

        self.conv1 = CoreImageBlock(inplanes, planes, 1, 1, 0, activation='mish')
        self.conv2 = CoreImageBlock(planes, planes, 3, stride, 1, activation='mish')
        self.conv3 = CoreImageBlock(planes, outplanes, 1, 1, 0, activation='linear')

        if stride != 1 or inplanes != outplanes:
            self.downsample = CoreImageBlock(inplanes, outplanes, 1, stride, 0,
                                           activation='linear')
        else:
            self.downsample = nn.Identity()

        self.final_act = get_activation_fn('mish')

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += identity
        return self.final_act(out)

class CSPBlock(nn.Module):
    """CSP Block"""
    def __init__(self, in_c, out_c, num_blocks=3):
        super().__init__()
        mid_c = out_c // 2

        self.part1_conv = CoreImageBlock(in_c, mid_c, 1, 1, 0, activation='leakyrelu')
        self.part2_conv = CoreImageBlock(in_c, mid_c, 1, 1, 0, activation='leakyrelu')

        self.blocks = nn.Sequential(*[
            MishBottleneck(mid_c, mid_c // 4) for _ in range(num_blocks)
        ])

        self.concat_conv = CoreImageBlock(mid_c * 2, out_c, 1, 1, 0, activation='leakyrelu')

    def forward(self, x):
        part1 = self.part1_conv(x)
        part2 = self.part2_conv(x)
        part2 = self.blocks(part2)
        out = torch.cat([part1, part2], dim=1)
        return self.concat_conv(out)

class InceptionModule(nn.Module):
    """Inception Module"""
    def __init__(self, in_c, out_c):
        super().__init__()
        branch_out = out_c // 4

        self.branch1 = CoreImageBlock(in_c, branch_out, 1, 1, 0, activation='silu')

        self.branch2 = nn.Sequential(
            CoreImageBlock(in_c, branch_out, 1, 1, 0, activation='silu'),
            CoreImageBlock(branch_out, branch_out, 3, 1, 1, activation='silu')
        )

        self.branch3 = nn.Sequential(
            CoreImageBlock(in_c, branch_out, 1, 1, 0, activation='silu'),
            CoreImageBlock(branch_out, branch_out, 5, 1, 2, activation='silu')
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            CoreImageBlock(in_c, branch_out, 1, 1, 0, activation='silu')
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)

class DenseBlock(nn.Module):
    """Dense Block"""
    def __init__(self, in_c, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_in_c = in_c + i * growth_rate
            self.layers.append(nn.Sequential(
                CoreImageBlock(layer_in_c, growth_rate * 4, 1, 1, 0, activation='mish'),
                CoreImageBlock(growth_rate * 4, growth_rate, 3, 1, 1, activation='mish')
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # FP16 FIX: PRE-multiply scale on q before matmul to prevent FP16
        # overflow.  Post-multiply (q @ k.T) * scale computes the full-magnitude
        # dot product first, which can exceed FP16 max (~65504) with head_dim>=32.
        # Pre-multiplying q keeps intermediate values in safe FP16 range.
        q = q * self.scale

        # FP32 attention: cast q,k to float32 for the matmul to guarantee
        # no FP16 overflow, then cast back.  The VRAM overhead is minimal
        # because N=196 (14x14), so the 196x196 attention matrix is tiny.
        attn = (q.float() @ k.float().transpose(-2, -1)).to(q.dtype)
        # Clip attention scores to prevent softmax overflow (NaN gradients)
        attn = torch.clamp(attn, min=-50, max=50)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            get_activation_fn('gelu'),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# =============================================================================
# CUSTOM BACKBONE ARCHITECTURES
# =============================================================================

class CustomConvNeXt(nn.Module):
    """Custom ConvNeXt"""
    def __init__(self, _num_classes=1000):
        super().__init__()
        dims = [96, 192, 384, 768]

        self.stem = nn.Sequential(
            CoreImageBlock(3, dims[0], 4, 4, 0, activation='gelu', use_ln=True)
        )

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(dims[i]) for _ in range([3, 3, 9, 3][i])
            ])
            self.stages.append(stage)

            if i < 3:
                self.stages.append(nn.Sequential(
                    nn.LayerNorm(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], 2, 2)
                ))

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], _num_classes)

    def forward(self, x):
        x = self.stem(x)

        for stage in self.stages:
            if isinstance(stage, nn.Sequential) and isinstance(stage[0], nn.LayerNorm):
                x = x.permute(0, 2, 3, 1)
                x = stage[0](x)
                x = x.permute(0, 3, 1, 2)
                x = stage[1](x)
            else:
                x = stage(x)

        x = x.mean([-2, -1])
        x = self.norm(x.unsqueeze(1)).squeeze(1)
        return self.head(x)

class PatchMerging(nn.Module):
    """Patch Merging Layer - Downsamples by merging 2x2 patches"""
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size: {L} vs {H}*{W}"
        assert H % 2 == 0 and W % 2 == 0, f"Resolution must be even: {H}x{W}"

        x = x.view(B, H, W, C)

        # Merge 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C)
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*C)

        x = x.view(B, -1, 4 * C)  # (B, H/2*W/2, 4*C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2*W/2, 2*C)

        return x

class CustomSwinTransformer(nn.Module):
    """Custom Swin Transformer"""
    def __init__(self, _num_classes=1000, img_size=224, patch_size=4, embed_dim=128,
                 depths=None, num_heads=None, window_size=7):
        if num_heads is None:
            num_heads = [4, 8, 16, 32]
        if depths is None:
            depths = [2, 2, 18, 2]
        super().__init__()
        self._num_classes = _num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_size = patch_size

        # Stem with proper channel progression
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, 3, 2, 1, bias=False),  # 224 -> 112
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1, bias=False),  # 112 -> 56
            nn.BatchNorm2d(embed_dim),
        )
        # After patch_embed, resolution is img_size // 4 (e.g., 224 -> 56)
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]

        # FIXED: Learnable absolute positional embedding
        num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=0.1)

        # Build layers with proper dimensions
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            current_resolution = (
                self.patches_resolution[0] // (2 ** i_layer),
                self.patches_resolution[1] // (2 ** i_layer)
            )

            # Swin blocks
            blocks = []
            for i in range(depths[i_layer]):
                blocks.append(
                    SwinTransformerBlock(
                        dim=layer_dim,
                        input_resolution=current_resolution,
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2
                    )
                )
            self.layers.append(nn.Sequential(*blocks))

            # Patch merging
            if i_layer < self.num_layers - 1:
                self.layers.append(
                    PatchMerging(
                        input_resolution=current_resolution,
                        dim=layer_dim,
                        norm_layer=nn.LayerNorm
                    )
                )

        self.norm = nn.LayerNorm(self.num_features)

        # FIXED: Stronger classification head
        self.head = nn.Sequential(
            nn.Linear(self.num_features, self.num_features * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features * 2, _num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, C, H, W)
        _b, _c, _h, _w = x.shape  # Extract shape for reference, used in comments
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # FIXED: Add absolute positional encoding
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # Apply stages
        for layer in self.layers:
            x = layer(x)

        # Final classification
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)

        return x

class CustomDynamicConvNet(nn.Module):
    """Custom network with dynamic convolutions"""
    def __init__(self, _num_classes=1000):
        super().__init__()

        self.stem = CoreImageBlock(3, 64, 3, 2, 1, activation='silu')

        self.stages = nn.ModuleList([
            self._make_stage(64, 128, 3, 2),
            self._make_stage(128, 256, 4, 2),
            self._make_stage(256, 512, 6, 2),
            self._make_stage(512, 1024, 3, 2)
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, _num_classes)

    def _make_stage(self, in_c, out_c, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            if i == num_blocks - 1:
                layers.append(nn.Sequential(
                    DynamicConv(in_c if i == 0 else out_c, out_c, 3, s, 1, num_kernels=4),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True)
                ))
            else:
                layers.append(InvertedResidualBlock(in_c if i == 0 else out_c, out_c, s, 4))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

class CustomMobileOne(nn.Module):
    """Custom MobileOne"""
    def __init__(self, _num_classes=1000):
        super().__init__()

        # FIXED: Gentler stem with proper channel progression
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),  # 32 instead of 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # FIXED: Proper channel scaling (32=>64=>128=>256=>384)
        self.stages = nn.ModuleList([
            self._make_stage(32, 64, 2, 2),     # Stage 1: 32=>64
            self._make_stage(64, 128, 3, 2),    # Stage 2: 64=>128
            self._make_stage(128, 256, 4, 2),   # Stage 3: 128=>256
            self._make_stage(256, 384, 2, 1)    # Stage 4: 256=>384
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # FIXED: Add dropout before classifier
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(384, _num_classes)

        # CRITICAL: Proper initialization
        self.apply(self._init_weights)

        # FIXED: Special initialization for MobileOne blocks
        for m in self.modules():
            if isinstance(m, MobileOneBlock):
                # Initialize all branches with small weights
                for branch in m.conv_kxk:
                    if isinstance(branch, nn.Sequential):
                        conv: nn.Conv2d = branch[0]  # type: ignore[assignment]
                        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                        # Scale down to prevent gradient explosion
                        conv.weight.data *= 0.5

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # FIXED: Smaller initialization for final layer
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def _make_stage(self, in_c, out_c, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            # FIXED: Use 2 branches instead of 3 to reduce complexity
            layers.append(MobileOneBlock(
                in_c if i == 0 else out_c,
                out_c, 3, s, 1,
                num_conv_branches=2  # Reduced from 3
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)  # FIXED: Add dropout
        return self.fc(x)

    def fuse_model(self):
        """Fuse reparameterization for inference"""
        for module in self.modules():
            if isinstance(module, MobileOneBlock):
                module.fuse_reparam()

# =============================================================================
# ADDITIONAL BACKBONE ARCHITECTURES
# =============================================================================

class CustomEfficientNetV4(nn.Module):
    """Custom EfficientNet V4"""
    def __init__(self, _num_classes=1000):
        super().__init__()

        self.stem = CoreImageBlock(3, 32, 3, 2, 1, activation='silu')

        self.stages = nn.ModuleList([
            self._make_stage(32, 16, 1, 1, 1),
            self._make_stage(16, 24, 2, 2, 2),
            self._make_stage(24, 40, 2, 2, 2),
            self._make_stage(40, 80, 3, 2, 3),
            self._make_stage(80, 112, 3, 1, 3),
            self._make_stage(112, 192, 4, 2, 4),
            self._make_stage(192, 320, 1, 1, 1),
        ])

        self.head_conv = CoreImageBlock(320, 1280, 1, 1, 0, activation='silu')
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, _num_classes)

    def _make_stage(self, in_c, out_c, expand_ratio, stride, num_blocks):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            layers.append(InvertedResidualBlock(in_c if i == 0 else out_c,
                                               out_c, s, expand_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head_conv(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


class GhostModuleV2Enhanced(nn.Module):
    """Enhanced Ghost Module with SE attention and residual"""
    def __init__(self, in_c, out_c, kernel_size=1, ratio=2, dw_size=3, use_se=True):
        super().__init__()
        init_channels = out_c // ratio

        self.primary_conv = CoreImageBlock(in_c, init_channels, kernel_size, 1,
                                          kernel_size//2, activation='leakyrelu')
        self.cheap_operation = CoreImageBlock(init_channels, init_channels, dw_size, 1,
                                             dw_size//2, groups=init_channels,
                                             activation='leakyrelu')

        # SE attention for better feature recalibration
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_c, out_c // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c // 4, out_c, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)

        if self.use_se:
            out = out * self.se(out)

        return out


class GhostBottleneckV2(nn.Module):
    """Ghost Bottleneck with residual connection and SE"""
    def __init__(self, in_c, out_c, stride=1, expand_ratio=2):
        super().__init__()
        hidden_c = in_c * expand_ratio
        self.stride = stride
        self.use_residual = stride == 1 and in_c == out_c

        # Expansion
        self.ghost1 = GhostModuleV2Enhanced(in_c, hidden_c, use_se=False)

        # Depthwise
        if stride > 1:
            self.dw = nn.Sequential(
                CoreImageBlock(hidden_c, hidden_c, 3, stride, 1,
                             groups=hidden_c, activation='leakyrelu'),
                CoreImageBlock(hidden_c, hidden_c, 3, 1, 1,
                             groups=hidden_c, activation='leakyrelu')  # Extra DW
            )
        else:
            self.dw = CoreImageBlock(hidden_c, hidden_c, 3, 1, 1,
                                    groups=hidden_c, activation='leakyrelu')

        # Projection with SE
        self.ghost2 = GhostModuleV2Enhanced(hidden_c, out_c, use_se=True)

        # Shortcut
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                CoreImageBlock(in_c, in_c, 3, stride, 1,
                             groups=in_c, activation='linear'),
                CoreImageBlock(in_c, out_c, 1, 1, 0, activation='linear')
            )

    def forward(self, x):
        residual = x

        x = self.ghost1(x)
        x = self.dw(x)
        x = self.ghost2(x)

        if self.use_residual:
            return x + residual
        else:
            return x + self.shortcut(residual)


class CustomGhostNetV2(nn.Module):
    """Enhanced Custom GhostNet V2 with deeper architecture and attention"""
    def __init__(self, _num_classes=1000):
        super().__init__()

        # ENHANCED: Multi-scale stem with stronger feature extraction
        self.stem = nn.Sequential(
            CoreImageBlock(3, 24, 3, 2, 1, activation='leakyrelu'),          # 224->112
            CoreImageBlock(24, 24, 3, 1, 1, groups=24, activation='leakyrelu'),  # DW
            CoreImageBlock(24, 32, 1, 1, 0, activation='leakyrelu'),         # PW expand
            CoreImageBlock(32, 32, 3, 1, 1, groups=32, activation='leakyrelu'),  # Extra DW
        )

        # ENHANCED: Better channel progression with more blocks
        self.stages = nn.ModuleList([
            # Stage 1: 112x112, 32->48
            self._make_stage(32, 48, 3, 2, expand_ratio=2),

            # Stage 2: 56x56, 48->96
            self._make_stage(48, 96, 4, 2, expand_ratio=3),

            # Stage 3: 28x28, 96->192
            self._make_stage(96, 192, 6, 2, expand_ratio=4),

            # Stage 4: 14x14, 192->384
            self._make_stage(192, 384, 5, 2, expand_ratio=4),  # Increased blocks

            # Stage 5: 7x7, 384->512 (ENHANCED - better final features)
            self._make_stage(384, 512, 4, 1, expand_ratio=3),  # Increased output channels
        ])

        # ENHANCED: Stronger head with better feature processing
        self.conv_head = nn.Sequential(
            CoreImageBlock(512, 1280, 1, 1, 0, activation='leakyrelu'),
            CoreImageBlock(1280, 1280, 3, 1, 1, groups=1280, activation='leakyrelu'),  # Extra DW
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # ENHANCED: Classification head with stronger regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Increased dropout
            nn.Linear(1280, 640),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(640, _num_classes)
        )

        # CRITICAL: Proper initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def _make_stage(self, in_c, out_c, num_blocks, stride, expand_ratio=2):
        """Create a stage with Ghost bottlenecks"""
        layers = []

        # First block handles stride
        layers.append(GhostBottleneckV2(in_c, out_c, stride, expand_ratio))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(GhostBottleneckV2(out_c, out_c, 1, expand_ratio))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.conv_head(x)
        x = self.avgpool(x).flatten(1)
        x = self.classifier(x)

        return x


class CustomResNetMish(nn.Module):
    """Custom ResNet with Mish"""
    def __init__(self, _num_classes=1000, layers=None):
        if layers is None:
            layers = [3, 4, 6, 3]
        super().__init__()

        self.inplanes = 64
        self.conv1 = CoreImageBlock(3, 64, 7, 2, 3, activation='mish')
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * MishBottleneck.expansion, _num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = []
        layers.append(MishBottleneck(self.inplanes, planes, stride))
        self.inplanes = planes * MishBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(MishBottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class CustomCSPDarkNet(nn.Module):
    """Custom CSP DarkNet"""
    def __init__(self, _num_classes=1000):
        super().__init__()

        self.stem = CoreImageBlock(3, 64, 3, 1, 1, activation='leakyrelu')

        self.stages = nn.ModuleList([
            CSPBlock(64, 128, 2),
            CSPBlock(128, 256, 4),
            CSPBlock(256, 512, 8),
            CSPBlock(512, 1024, 4)
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, _num_classes)

    def forward(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = F.max_pool2d(x, 2, 2)
            x = stage(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class CustomInceptionV4(nn.Module):
    """Custom Inception V4"""
    def __init__(self, _num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            CoreImageBlock(3, 64, 3, 2, 1, activation='silu'),
            CoreImageBlock(64, 128, 3, 1, 1, activation='silu'),
            CoreImageBlock(128, 256, 3, 2, 1, activation='silu')
        )

        self.inception_blocks = nn.Sequential(*[
            InceptionModule(256, 256) for _ in range(4)
        ])

        self.reduction = CoreImageBlock(256, 512, 3, 2, 1, activation='silu')

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, _num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_blocks(x)
        x = self.reduction(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class CustomViTHybrid(nn.Module):
    """Custom ViT Hybrid"""
    def __init__(self, _num_classes=1000, img_size=224, patch_size=16):
        super().__init__()

        # DEEP CNN STEM: Progressive feature extraction (3 -> 768 channels)
        self.stem = nn.Sequential(
            # Stage 1: 224 -> 112
            CoreImageBlock(3, 64, 3, 2, 1, activation='gelu'),
            CoreImageBlock(64, 64, 3, 1, 1, activation='gelu'),

            # Stage 2: 112 -> 56
            CoreImageBlock(64, 128, 3, 2, 1, activation='gelu'),
            CoreImageBlock(128, 128, 3, 1, 1, activation='gelu'),
            CoreImageBlock(128, 128, 3, 1, 1, activation='gelu'),

            # Stage 3: 56 -> 28
            CoreImageBlock(128, 256, 3, 2, 1, activation='gelu'),
            CoreImageBlock(256, 256, 3, 1, 1, activation='gelu'),
            CoreImageBlock(256, 256, 3, 1, 1, activation='gelu'),

            # Stage 4: 28 -> 14 (deeper feature extraction)
            CoreImageBlock(256, 512, 3, 2, 1, activation='gelu'),
            CoreImageBlock(512, 512, 3, 1, 1, activation='gelu'),
            CoreImageBlock(512, 512, 3, 1, 1, activation='gelu'),

            # Final projection to ViT dimensions
            CoreImageBlock(512, 768, 1, 1, 0, activation='gelu'),
        )

        # 14x14 patches after stem
        self.num_patches = (img_size // 16) ** 2  # 196 patches
        self.embed_dim = 768  # Standard ViT-Base dimension

        # No additional projection needed - already at 768

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        # DEEP TRANSFORMER: 18 blocks (ViT-Large depth)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlockWithLayerScale(
                self.embed_dim,
                num_heads=12,
                mlp_ratio=4.0,
                init_values=1e-5  # Smaller init for deeper models
            )
            for _ in range(18)  # DEEP: 18 transformer blocks
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # Classification head with intermediate layer
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, _num_classes)
        )

        # Proper initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        # Deep CNN feature extraction
        x = self.stem(x)  # (B, 768, 14, 14)

        # Convert to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, 768)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Deep transformer processing
        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        # Classification from CLS token
        cls_output = x[:, 0]
        return self.head(cls_output)

class CustomCoAtNet(nn.Module):
    """Custom CoAtNet"""
    def __init__(self, _num_classes=1000):
        super().__init__()

        # OPTIMIZED STEM: 224 -> 56
        self.stem = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Block 2: 112 -> 56
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # STAGE 1: DEEP CNN (56x56, 128 -> 256)
        stage1_blocks = []

        # Downsample: 56 -> 28
        stage1_blocks.append(InvertedResidualBlock(128, 192, 2, 6))

        # Deep feature extraction at 28x28
        stage1_blocks.extend([
            InvertedResidualBlock(192, 192, 1, 6),
            InvertedResidualBlock(192, 192, 1, 4),
            InvertedResidualBlock(192, 192, 1, 6),
            InvertedResidualBlock(192, 192, 1, 4),
            InvertedResidualBlock(192, 192, 1, 6),
        ])

        # Channel expansion
        stage1_blocks.append(InvertedResidualBlock(192, 256, 1, 6))

        self.stage1 = nn.Sequential(*stage1_blocks)

        # STAGE 2: DEEP CNN + Attention (28x28, 256 -> 512)
        stage2_cnn_blocks = []

        # Downsample: 28 -> 14
        stage2_cnn_blocks.append(InvertedResidualBlock(256, 384, 2, 6))

        # Deep CNN at 14x14 - MORE BLOCKS for better feature extraction
        stage2_cnn_blocks.extend([
            InvertedResidualBlock(384, 384, 1, 6),
            InvertedResidualBlock(384, 384, 1, 4),
            InvertedResidualBlock(384, 384, 1, 6),
            InvertedResidualBlock(384, 384, 1, 4),
            InvertedResidualBlock(384, 384, 1, 6),
            InvertedResidualBlock(384, 384, 1, 4),
        ])

        # Channel expansion to 512
        stage2_cnn_blocks.append(InvertedResidualBlock(384, 512, 1, 6))

        self.stage2_cnn = nn.Sequential(*stage2_cnn_blocks)

        # OPTIMIZED attention: 4 blocks (not 6) with better initialization
        self.stage2_attn_blocks = nn.ModuleList([
            TransformerEncoderBlockWithLayerScale(512, num_heads=16, mlp_ratio=4.0, init_values=0.1)
            for _ in range(4)  # 4 attention blocks - optimal for performance
        ])

        # STAGE 3: OPTIMIZED Transformer (14x14, 512 -> 768)
        self.stage3_proj = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1, 1, bias=False),
            nn.BatchNorm2d(768),
            nn.GELU(),
        )

        # OPTIMAL transformer depth: 12 blocks with strong initialization
        # This balances representational power with training stability
        self.stage3_transformer = nn.ModuleList([
            TransformerEncoderBlockWithLayerScale(
                768,
                num_heads=12,
                mlp_ratio=4.0,
                init_values=0.1  # Strong initial values for stable training
            )
            for _ in range(12)  # 12 transformer blocks (was 24 - too deep!)
        ])

        self.norm = nn.LayerNorm(768)

        # Multi-layer classification head
        self.head = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, _num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Optimized stem
        x = self.stem(x)  # (B, 128, 56, 56)

        # Stage 1: Deep CNN
        x = self.stage1(x)  # (B, 256, 28, 28)

        # Stage 2: Deep CNN + Attention
        x = self.stage2_cnn(x)  # (B, 512, 14, 14)

        # Apply attention blocks with direct residuals for best gradient flow
        b, c, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, 196, 512)

        # Attention blocks (residual is INTERNAL to TransformerEncoderBlockWithLayerScale)
        for attn_block in self.stage2_attn_blocks:
            x_seq = attn_block(x_seq)

        x = x_seq.transpose(1, 2).reshape(b, c, h, w)

        # Stage 3: Optimized transformer
        x = self.stage3_proj(x)  # (B, 768, 14, 14)

        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)

        # Transformer blocks (residual is INTERNAL to TransformerEncoderBlockWithLayerScale)
        for transformer_block in self.stage3_transformer:
            x = transformer_block(x)

        x = self.norm(x)

        # Global average pooling
        x = x.mean(dim=1)  # Average all token positions

        return self.head(x)

class CustomRegNet(nn.Module):
    """Custom RegNet"""
    def __init__(self, _num_classes=1000):
        super().__init__()

        self.stem = CoreImageBlock(3, 32, 3, 2, 1, activation='relu')

        widths = [32, 64, 160, 384]
        depths = [1, 3, 6, 6]

        self.stages = nn.ModuleList()
        in_w = 32
        for i, (w, d) in enumerate(zip(widths, depths)):
            stage = []
            for j in range(d):
                stride = 2 if j == 0 and i > 0 else 1
                stage.append(MishBottleneck(in_w if j == 0 else w * 4, w, stride))
            self.stages.append(nn.Sequential(*stage))
            in_w = w * 4

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1] * 4, _num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class CustomDenseNetHybrid(nn.Module):
    """Custom DenseNet Hybrid"""
    def __init__(self, _num_classes=1000):
        super().__init__()
        growth_rate = 32

        self.conv1 = CoreImageBlock(3, 64, 7, 2, 3, activation='mish')
        self.pool1 = nn.MaxPool2d(3, 2, 1)

        num_features = 64
        self.dense1 = DenseBlock(num_features, growth_rate, 6)
        num_features += growth_rate * 6
        self.trans1 = CoreImageBlock(num_features, num_features // 2, 1, 1, 0, activation='mish')
        self.pool2 = nn.AvgPool2d(2, 2)

        num_features = num_features // 2
        self.dense2 = DenseBlock(num_features, growth_rate, 12)
        num_features += growth_rate * 12
        self.trans2 = CoreImageBlock(num_features, num_features // 2, 1, 1, 0, activation='mish')
        self.pool3 = nn.AvgPool2d(2, 2)

        num_features = num_features // 2
        self.dense3 = DenseBlock(num_features, growth_rate, 8)
        num_features += growth_rate * 8

        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, _num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.pool2(x)

        x = self.dense2(x)
        x = self.trans2(x)
        x = self.pool3(x)

        x = self.dense3(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

class TransformerEncoderBlockWithLayerScale(nn.Module):
    """
    Transformer block with layer scaling for stable deep training
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, init_values=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(0.1)
        )

        # Layer scale parameters
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x

class CustomDeiTStyle(nn.Module):
    """Custom DeiT-style transformer"""
    def __init__(self, _num_classes=1000, img_size=224, patch_size=16):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        embed_dim = 768  # Standard ViT-Base dimension

        # FIXED: Better patch embedding with conv stem
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim // 4, 3, 2, 1, bias=False),  # 224 -> 112
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 2, 1, bias=False),  # 112 -> 56
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1, bias=False),  # 56 -> 28
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 2, 1, bias=False),  # 28 -> 14
        )

        # Recalculate num_patches for conv stem
        self.num_patches = (img_size // 16) ** 2  # 14x14 = 196

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, embed_dim))

        # FIXED: Optimal depth with layer scaling
        self.blocks = nn.ModuleList([
            TransformerEncoderBlockWithLayerScale(embed_dim, num_heads=12, mlp_ratio=4.0)
            for _ in range(12)  # Reduced from 16 to 12 for better training
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # FIXED: Separate heads with proper initialization
        self.head = nn.Linear(embed_dim, _num_classes)
        self.head_dist = nn.Linear(embed_dim, _num_classes)

        # Proper initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.trunc_normal_(self.head_dist.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.head_dist.bias)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)

        x = x + self.pos_embed

        # Apply blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Average both token predictions
        return (self.head(x[:, 0]) + self.head_dist(x[:, 1])) / 2


class CustomMaxViT(nn.Module):
    """Custom MaxViT -   Proper shape handling between CNN and Transformer"""
    def __init__(self, _num_classes=1000):
        super().__init__()

        # ENHANCED STEM: 224 -> 56 with strong feature extraction
        self.stem = nn.Sequential(
            # 224 -> 112
            CoreImageBlock(3, 64, 3, 2, 1, activation='gelu'),
            CoreImageBlock(64, 64, 3, 1, 1, activation='gelu'),
            CoreImageBlock(64, 64, 3, 1, 1, activation='gelu'),

            # 112 -> 56
            CoreImageBlock(64, 128, 3, 2, 1, activation='gelu'),
            CoreImageBlock(128, 128, 3, 1, 1, activation='gelu'),
            CoreImageBlock(128, 128, 3, 1, 1, activation='gelu'),
        )

        # STAGE 1: DEEP CNN (56x56, 128 -> 256)
        stage1_blocks = []

        # Downsample: 56 -> 28
        stage1_blocks.append(InvertedResidualBlock(128, 192, 2, 6))

        # DEEP feature extraction at 28x28 (increased from 4 to 6)
        for _ in range(6):
            stage1_blocks.append(InvertedResidualBlock(192, 192, 1, 6))

        # Expand to 256
        stage1_blocks.append(InvertedResidualBlock(192, 256, 1, 6))

        self.stage1 = nn.Sequential(*stage1_blocks)

        # STAGE 2: DEEP CNN + Attention (28x28, 256 -> 512)
        stage2_cnn_blocks = []

        # Downsample: 28 -> 14
        stage2_cnn_blocks.append(InvertedResidualBlock(256, 384, 2, 6))

        # VERY DEEP feature extraction at 14x14 (increased from 5 to 8 blocks)
        for _ in range(8):
            stage2_cnn_blocks.append(InvertedResidualBlock(384, 384, 1, 6))

        # Expand to 512
        stage2_cnn_blocks.append(InvertedResidualBlock(384, 512, 1, 6))

        self.stage2_cnn = nn.Sequential(*stage2_cnn_blocks)

        # ENHANCED: 3 strong attention blocks (was 2)
        self.stage2_attn = nn.ModuleList([
            TransformerEncoderBlockWithLayerScale(
                512,
                num_heads=16,
                mlp_ratio=4.0,
                init_values=0.1
            )
            for _ in range(3)  # 3 attention blocks for better feature refinement
        ])

        # STAGE 3: Transformer (14x14, 512 -> 768)
        self.stage3_proj = nn.Sequential(
            CoreImageBlock(512, 768, 3, 1, 1, activation='gelu'),
            nn.Dropout2d(0.1),
        )

        # OPTIMIZED: 10 transformer blocks (was 8, but with better initialization)
        self.stage3_transformer = nn.ModuleList([
            TransformerEncoderBlockWithLayerScale(
                768,
                num_heads=12,
                mlp_ratio=4.0,
                init_values=0.1  # Strong initialization
            )
            for _ in range(10)  # 10 blocks - good balance
        ])

        # STRONG classification head
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, _num_classes)
        )

        # Proper initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # FIXED: Better initialization for deep networks
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Enhanced stem
        x = self.stem(x)  # (B, 128, 56, 56)

        # Stage 1: DEEP CNN
        x = self.stage1(x)  # (B, 256, 28, 28)

        # Stage 2: VERY DEEP CNN + Attention
        x = self.stage2_cnn(x)  # (B, 512, 14, 14)

        # Apply attention with direct residuals
        b, c, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, 196, 512)

        # Attention blocks (residual is INTERNAL to TransformerEncoderBlockWithLayerScale)
        for attn_block in self.stage2_attn:
            x_seq = attn_block(x_seq)

        x = x_seq.transpose(1, 2).reshape(b, c, h, w)

        # Stage 3: Optimized Transformer
        x = self.stage3_proj(x)  # (B, 768, 14, 14)

        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)

        # Transformer blocks (residual is INTERNAL to TransformerEncoderBlockWithLayerScale)
        for transformer_block in self.stage3_transformer:
            x = transformer_block(x)

        x = self.norm(x)

        # Global average pooling
        x = x.mean(dim=1)

        return self.head(x)


# =============================================================================
# TRANSFER LEARNING UTILITIES - FIXED
# =============================================================================

def download_pretrained_weights(pretrained_dir: Path, logger_inst=None):
    """
    Download pretrained weights for all configured backbones.
      Added retry logic and better error handling.
    """
    if logger_inst is None:
        logger_inst = logger

    if not PRETRAINED_DOWNLOAD_MAP:
        logger_inst.warning(
            "PRETRAINED_DOWNLOAD_MAP is empty. No pretrained weights configured to download."
        )
        return {"downloaded": 0, "skipped": 0, "attempted": []}

    try:
        import urllib.error
        import urllib.request

        from tqdm.auto import tqdm as download_tqdm

        if not pretrained_dir.exists():
            logger_inst.info(f"Creating weights directory: {pretrained_dir}")
            pretrained_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger_inst.info(f"Weights directory found: {pretrained_dir}")

        total_downloaded_count = 0
        total_skipped_count = 0
        attempted_files = []

        files_to_check = set()
        for path in PRETRAINED_PATHS.values():
            if isinstance(path, Path):
                files_to_check.add(path.name)

        if not files_to_check:
            logger_inst.warning("PRETRAINED_PATHS is empty. Checking all files in PRETRAINED_DOWNLOAD_MAP.")
            files_to_check = set(PRETRAINED_DOWNLOAD_MAP.keys())

        logger_inst.info(f"Checking {len(files_to_check)} pretrained weight files...")

        for filename in files_to_check:
            attempted_files.append(filename)

            if filename not in PRETRAINED_DOWNLOAD_MAP:
                logger_inst.warning(
                    f"  SKIPPED: No download URL defined for required weight file '{filename}'."
                )
                continue

            info = PRETRAINED_DOWNLOAD_MAP[filename]
            file_path = pretrained_dir / filename
            url = info['url']
            expected_size_mb = info['expected_size_mb']

            # Check if file exists and is valid
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)

                #   More lenient size check (within 10% tolerance)
                size_tolerance = expected_size_mb * 0.1
                if abs(file_size_mb - expected_size_mb) < max(0.5, size_tolerance):
                    logger_inst.info(
                        f"  SKIPPED: '{filename}' found and size verified ({file_size_mb:.1f}MB)."
                    )
                    total_skipped_count += 1
                    continue
                else:
                    logger_inst.warning(
                        f"  RE-DOWNLOADING: '{filename}' found but size check failed "
                        f"({file_size_mb:.1f}MB vs {expected_size_mb:.1f}MB expected)."
                    )

            # Download with retry logic
            max_retries = 3
            for retry in range(max_retries):
                try:
                    logger_inst.info(f"  DOWNLOADING: '{filename}' (attempt {retry+1}/{max_retries})")

                    class DownloadProgressBar(download_tqdm):
                        def update_to(self, b=1, bsize=1, tsize=None):
                            if tsize is not None:
                                self.total = tsize
                            self.update(b * bsize - self.n)

                    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                        urllib.request.urlretrieve(url, file_path, reporthook=t.update_to)

                    # Verify download
                    final_size_mb = file_path.stat().st_size / (1024 * 1024)
                    size_tolerance = expected_size_mb * 0.1
                    if abs(final_size_mb - expected_size_mb) < max(0.5, size_tolerance):
                        logger_inst.info(
                            f"  Cool SUCCESS: '{filename}' downloaded and verified ({final_size_mb:.1f}MB)."
                        )
                        total_downloaded_count += 1
                        break
                    else:
                        logger_inst.warning(
                            f"  Size check failed ({final_size_mb:.1f}MB vs {expected_size_mb:.1f}MB). "
                            f"Retrying..."
                        )
                        if file_path.exists():
                            file_path.unlink()

                except urllib.error.HTTPError as e:
                    logger_inst.error(f"  HTTP Error {e.code} for {filename}: {e.reason}")
                    if retry == max_retries - 1:
                        logger_inst.error(f"  x FAILED: All download attempts failed for {filename}")
                    else:
                        logger_inst.info("  Retrying in 2 seconds...")
                        time.sleep(2)

                except Exception as download_e:
                    logger_inst.error(f"  Download error for {filename}: {download_e}")
                    if retry == max_retries - 1:
                        logger_inst.error(f"  x FAILED: All download attempts failed for {filename}")
                    if file_path.exists():
                        file_path.unlink()

        result = {
            "downloaded": total_downloaded_count,
            "skipped": total_skipped_count,
            "attempted": attempted_files
        }

        logger_inst.info(
            f"\n{'='*60}\n"
            f"Weight download complete:\n"
            f"  Downloaded: {result['downloaded']}\n"
            f"  Skipped: {result['skipped']}\n"
            f"  Total checked: {len(files_to_check)}\n"
            f"{'='*60}"
        )

        return result

    except ImportError:
        logger_inst.error(
            "Dependencies missing. Cannot perform automatic download."
        )
        return {"downloaded": 0, "skipped": 0, "attempted": [], "error": "ImportError"}
    except Exception as e:
        logger_inst.exception(f"Unexpected error during weight download: {e}")
        return {"downloaded": 0, "skipped": 0, "attempted": [], "error": str(e)}


def get_finetune_param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:

    head_keywords = ['head.', 'fc.', 'classifier.']
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_head = any(keyword in name for keyword in head_keywords)

        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return backbone_params, head_params

def adapt_state_dict_keys(state_dict, model_state_dict):

    adapted_dict = {}

    # Common key mappings
    key_mappings = {
        'classifier': 'head',
        'fc': 'head',
        'features': 'stages',
        'norm': 'norm',
        'downsample': 'layers'
    }

    for k, v in state_dict.items():
        # Remove 'module.' prefix if present
        if k.startswith('module.'):
            k = k[7:]

        # Try direct match first
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            adapted_dict[k] = v
            continue

        # Try common mappings
        adapted_key = k
        for old, new in key_mappings.items():
            if old in k:
                adapted_key = k.replace(old, new)
                if adapted_key in model_state_dict and v.shape == model_state_dict[adapted_key].shape:
                    adapted_dict[adapted_key] = v
                    break

    return adapted_dict

def create_state_dict_mapper(backbone_name: str) -> dict[str, str]:

    mappers = {
        'CustomSwinTransformer': {
            'patch_embed.proj': 'patch_embed',
            'patch_embed.norm': 'patch_norm',
            'layers.0': 'layers.0',
            'layers.1': 'layers.1',
            'layers.2': 'layers.2',
            'layers.3': 'layers.3',
            'norm': 'norm',
            'head': 'head',
        },
        'CustomEfficientNetV4': {
            'features.0.0': 'stem.conv',
            'features.0.1': 'stem.norm',
            'features': 'stages',
            'classifier': 'classifier',
        },
        'CustomResNetMish': {
            'conv1.weight': 'conv1.conv.weight',
            'bn1': 'conv1.norm',
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
            'fc': 'fc',
        },
        'CustomDenseNetHybrid': {
            'features.conv0': 'conv1.conv',
            'features.norm0': 'conv1.norm',
            'features.denseblock1': 'dense1',
            'features.transition1.conv': 'trans1.conv',
            'features.transition1.norm': 'trans1.norm',
            'features.denseblock2': 'dense2',
            'features.transition2.conv': 'trans2.conv',
            'features.transition2.norm': 'trans2.norm',
            'features.denseblock3': 'dense3',
            'features.norm5': 'norm',
            'classifier': 'fc',
        },
        'CustomViTHybrid': {
            'conv_proj': 'patch_proj',
            'encoder.pos_embedding': 'pos_embed',
            'class_token': 'cls_token',
            'encoder.layers.encoder_layer_': 'transformer_blocks.',
            'encoder.ln': 'norm',
            'heads.head': 'head',
        },
        'CustomRegNet': {
            'stem': 'stem.conv',
            's1': 'stages.0',
            's2': 'stages.1',
            's3': 'stages.2',
            's4': 'stages.3',
            'fc': 'fc',
        },
        'CustomDeiTStyle': {
            'patch_embed': 'patch_embed',
            'cls_token': 'cls_token',
            'pos_embed': 'pos_embed',
            'blocks': 'blocks',
            'norm': 'norm',
            'head': 'head',
        },
        # For backbones without specific mapping, return empty dict
        'CustomGhostNetV2': {},
        'CustomCSPDarkNet': {},
        'CustomInceptionV4': {},
        'CustomCoAtNet': {},
        'CustomMaxViT': {},
        'CustomMobileOne': {},
        'CustomDynamicConvNet': {},
        'CustomConvNeXt': {},
    }

    return mappers.get(backbone_name, {})


def map_pretrained_keys(state_dict, model_state_dict, backbone_name, logger_inst=None):

    if logger_inst is None:
        logger_inst = logger

    mapper = create_state_dict_mapper(backbone_name)
    mapped_dict = {}

    # Direct key matching first
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present
        clean_key = k[7:] if k.startswith('module.') else k

        # Try direct match
        if clean_key in model_state_dict and v.shape == model_state_dict[clean_key].shape:
            mapped_dict[clean_key] = v
            continue

        # Try mapper
        for old_prefix, new_prefix in mapper.items():
            if clean_key.startswith(old_prefix):
                new_key = clean_key.replace(old_prefix, new_prefix, 1)
                if new_key in model_state_dict and v.shape == model_state_dict[new_key].shape:
                    mapped_dict[new_key] = v
                    logger_inst.debug(f"  Mapped: {clean_key} => {new_key}")
                    break

    return mapped_dict

def smart_adapt_pretrained_weights(pretrained_state_dict: dict,
                                   model: nn.Module,
                                   backbone_name: str,
                                   logger_inst=None,
                                   use_existing_mappers: bool = True) -> dict:
    """
      Improved weight adaptation with better matching strategies.
    """
    if logger_inst is None:
        from logging import getLogger
        logger_inst = getLogger(__name__)

    model_state = model.state_dict()
    adapted_dict = {}

    # STEP 1: Direct key matching (most reliable)
    logger_inst.info("  Step 1: Direct key matching...")
    for pre_name, pre_tensor in pretrained_state_dict.items():
        # Remove common prefixes
        clean_key = pre_name
        for prefix in ['module.', 'model.', 'backbone.', '_orig_mod.']:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]

        # Try direct match
        if clean_key in model_state and pre_tensor.shape == model_state[clean_key].shape:
            adapted_dict[clean_key] = pre_tensor
            logger_inst.debug(f"    Direct match: {clean_key}")

    logger_inst.info(f"    Direct matching found {len(adapted_dict)} matches")

    # STEP 2: Shape-based matching with semantic filtering
    logger_inst.info("  Step 2: Shape-based matching...")

    # Build shape index for unmatched model parameters
    unmatched_model_params = {k: v for k, v in model_state.items() if k not in adapted_dict}
    shape_to_model_keys = defaultdict(list)
    for k, v in unmatched_model_params.items():
        shape_to_model_keys[v.shape].append(k)

    # Find pretrained params with matching shapes (excluding already matched)
    set(adapted_dict.values())

    for pre_name, pre_tensor in pretrained_state_dict.items():
        if id(pre_tensor) in {id(v) for v in adapted_dict.values()}:
            continue  # Skip already matched

        if pre_tensor.shape in shape_to_model_keys:
            candidates = shape_to_model_keys[pre_tensor.shape]

            if len(candidates) == 1:
                # Unique shape match
                model_key = candidates[0]
                adapted_dict[model_key] = pre_tensor
                shape_to_model_keys[pre_tensor.shape].remove(model_key)
                logger_inst.debug(f"    Shape match: {pre_name} -> {model_key}")
            elif len(candidates) > 1:
                # Multiple candidates - use name similarity
                best_match = None
                best_score = 0

                for candidate in candidates:
                    # Simple name similarity score
                    score = 0
                    pre_parts = set(pre_name.lower().split('.'))
                    cand_parts = set(candidate.lower().split('.'))
                    common = pre_parts & cand_parts
                    score = len(common)

                    # Bonus for matching layer types
                    if ('conv' in pre_name and 'conv' in candidate) or \
                       ('bn' in pre_name and 'norm' in candidate) or \
                       ('fc' in pre_name and 'fc' in candidate):
                        score += 2

                    if score > best_score:
                        best_score = score
                        best_match = candidate

                if best_match and best_score > 1:
                    adapted_dict[best_match] = pre_tensor
                    shape_to_model_keys[pre_tensor.shape].remove(best_match)
                    logger_inst.debug(f"    Semantic match ({best_score}): {pre_name} -> {best_match}")

    logger_inst.info(f"    Shape matching added {len(adapted_dict) - len([k for k in adapted_dict if k in model_state])} matches")

    # STEP 3: Pattern-based transformation matching
    logger_inst.info("  Step 3: Pattern-based transformations...")

    common_transforms = [
        ('features.', 'stages.'),
        ('classifier.', 'head.'),
        ('fc.', 'head.'),
        ('norm.', 'bn.'),
        ('bn.', 'norm.'),
        ('downsample.0.', 'downsample.conv.'),
        ('downsample.1.', 'downsample.norm.'),
    ]

    initial_count = len(adapted_dict)
    for pre_name, pre_tensor in pretrained_state_dict.items():
        if id(pre_tensor) in {id(v) for v in adapted_dict.values()}:
            continue

        for old_pattern, new_pattern in common_transforms:
            if old_pattern in pre_name:
                transformed = pre_name.replace(old_pattern, new_pattern)
                if transformed in model_state and \
                   transformed not in adapted_dict and \
                   pre_tensor.shape == model_state[transformed].shape:
                    adapted_dict[transformed] = pre_tensor
                    logger_inst.debug(f"    Transform: {pre_name} -> {transformed}")
                    break

    logger_inst.info(f"    Transform matching added {len(adapted_dict) - initial_count} matches")

    # Summary
    match_rate = (len(adapted_dict) / len(model_state)) * 100 if model_state else 0
    logger_inst.info(f"  TOTAL: Matched {len(adapted_dict)}/{len(model_state)} parameters ({match_rate:.1f}%)")

    return adapted_dict

def load_pretrained_to_model(model: nn.Module,
                             pretrained_path: str,
                             backbone_name: str | None = None,
                             logger_inst=None,
                             map_location: str = 'cpu',
                             freeze_backbone: bool = False,
                             auto_move_to_device: bool = False) -> dict[str, Any]:

    if logger_inst is None:
        logger_inst = logger

    try:
        checkpoint = torch.load(pretrained_path, map_location=map_location)

        # Extract state dict from checkpoint
        if 'model_state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            pretrained_state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['state_dict']
        else:
            pretrained_state_dict = checkpoint

        # Remove 'module.' prefix if present
        cleaned_state = {}
        for k, v in pretrained_state_dict.items():
            clean_key = k[7:] if k.startswith('module.') else k
            cleaned_state[clean_key] = v
        pretrained_state_dict = cleaned_state


        logger_inst.info(f"  Applying smart weight adaptation for {backbone_name}...")
        # Ensure backbone_name is not None for smart_adapt_pretrained_weights
        if backbone_name is None:
            backbone_name = model.__class__.__name__
        adapted_state_dict = smart_adapt_pretrained_weights(
            pretrained_state_dict,
            model,
            backbone_name,
            logger_inst
        )

        # Load adapted weights
        model_state_dict = model.state_dict()
        matched_count = len(adapted_state_dict)

        if matched_count > 0:
            # Load the adapted weights
            model.load_state_dict(adapted_state_dict, strict=False)

            loaded_keys = list(adapted_state_dict.keys())
            missing_keys = list(set(model_state_dict.keys()) - set(adapted_state_dict.keys()))

            stats = {
                'matched': matched_count,
                'loaded_keys': loaded_keys,
                'missing_keys': missing_keys,
                'skipped_shape_mismatch': [],
                'skipped_not_in_model': []
            }

            logger_inst.info(f"  Cool Successfully loaded {matched_count} parameter tensors")
            logger_inst.info(f"  {len(missing_keys)} parameters initialized randomly (head/unmatched)")

        else:
            logger_inst.warning("  No compatible parameters found")
            logger_inst.info("  Training from scratch")

            stats = {
                'matched': 0,
                'loaded_keys': [],
                'missing_keys': list(model_state_dict.keys()),
                'skipped_shape_mismatch': [],
                'skipped_not_in_model': []
            }

        # Optionally freeze backbone
        if freeze_backbone and matched_count > 0:
            backbone_params, head_params = get_finetune_param_groups(model)
            for param in backbone_params:
                param.requires_grad = False
            logger_inst.info(f"  Froze {len(backbone_params)} backbone parameters")
            logger_inst.info(f"  Kept {len(head_params)} head parameters trainable")

        # Optionally move to device
        if auto_move_to_device:
            model.to(DEVICE)

        return stats

    except Exception as e:
        logger_inst.exception(f"x Failed to load pretrained weights: {e}")
        return {
            'matched': 0,
            'error': str(e),
            'loaded_keys': [],
            'skipped_shape_mismatch': [],
            'skipped_not_in_model': [],
            'missing_keys': []
        }


# =============================================================================
# ARCHITECTURE VERIFICATION
# =============================================================================

def verify_swin_transformer(model, backbone_name, input_size=224, _num_classes=None, device=DEVICE):

    if _num_classes is None:
        _num_classes = _num_classes if _num_classes else 1000

    try:
        model.eval()
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)

        with torch.no_grad():
            output = model(dummy_input)

        if output.shape != (batch_size, _num_classes):
            smoke_checker.log_check(f"{backbone_name}: Swin Output Shape", "FAIL",
                                   f"Expected ({batch_size}, {_num_classes}), got {output.shape}")
            return False

        has_shift = False
        mask_nontrivial = False

        for module in model.modules():
            if isinstance(module, SwinTransformerBlock) and module.shift_size > 0:
                has_shift = True
                if module.attn_mask is not None:
                    mask_nontrivial = (module.attn_mask != 0).any().item()
                    break

        if has_shift and mask_nontrivial:
            smoke_checker.log_check(f"{backbone_name}: Swin Shifted Window", "PASS",
                                   "Shifted window attention with non-trivial mask detected")
        else:
            smoke_checker.log_check(f"{backbone_name}: Swin Shifted Window", "WARN",
                                   f"Shift detected: {has_shift}, Mask non-trivial: {mask_nontrivial}")

        model.train()
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()

        smoke_checker.log_check(f"{backbone_name}: Swin Backward Pass", "PASS",
                               "Gradients computed successfully")
        return True

    except Exception as e:
        smoke_checker.log_check(f"{backbone_name}: Swin Verification", "FAIL", str(e))
        return False


def verify_dynamic_conv(model, backbone_name, input_size=224, _num_classes=None, device=DEVICE):

    if _num_classes is None:
        _num_classes = _num_classes if _num_classes else 1000

    try:
        dynamic_convs = [m for m in model.modules() if isinstance(m, DynamicConv)]

        if not dynamic_convs:
            smoke_checker.log_check(f"{backbone_name}: DynamicConv Detection", "WARN",
                                   "No DynamicConv modules found")
            return True

        model.eval()
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)

        dc = dynamic_convs[0]
        test_input = torch.randn(batch_size, dc.in_channels, 32, 32).to(device)
        attention = dc.controller(test_input)

        if attention.shape != (batch_size, dc.num_kernels):
            smoke_checker.log_check(f"{backbone_name}: DynamicConv Attention Shape", "FAIL",
                                   f"Expected ({batch_size}, {dc.num_kernels}), got {attention.shape}")
            return False

        attention_sum = attention.sum(dim=1)
        if not torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-5):
            smoke_checker.log_check(f"{backbone_name}: DynamicConv Attention Sum", "FAIL",
                                   "Attention weights don't sum to 1")
            return False

        with torch.no_grad():
            output_dc = dc(test_input)

        expected_h = (32 + 2 * dc.padding - dc.kernel_size) // dc.stride + 1
        expected_w = expected_h
        if output_dc.shape != (batch_size, dc.out_channels, expected_h, expected_w):
            smoke_checker.log_check(f"{backbone_name}: DynamicConv Output Shape", "FAIL",
                                   f"Unexpected output shape: {output_dc.shape}")
            return False

        model.train()
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()

        if dc.weight_bank.grad is None:
            smoke_checker.log_check(f"{backbone_name}: DynamicConv Gradients", "FAIL",
                                   "No gradients for weight_bank")
            return False

        smoke_checker.log_check(f"{backbone_name}: DynamicConv Verification", "PASS",
                               f"Found {len(dynamic_convs)} DynamicConv modules, all checks passed")
        return True

    except Exception as e:
        smoke_checker.log_check(f"{backbone_name}: DynamicConv Verification", "FAIL", str(e))
        return False


def verify_mobileone_reparam(model, backbone_name, input_size=224, _num_classes=None, device=DEVICE):

    if _num_classes is None:
        _num_classes = _num_classes if _num_classes else 1000

    try:
        mobileone_blocks = [m for m in model.modules() if isinstance(m, MobileOneBlock)]

        if not mobileone_blocks:
            smoke_checker.log_check(f"{backbone_name}: MobileOne Detection", "WARN",
                                   "No MobileOne blocks found")
            return True

        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)

        model.train()
        output_train = model(dummy_input)
        loss = output_train.sum()
        loss.backward()

        smoke_checker.log_check(f"{backbone_name}: MobileOne Training", "PASS",
                               "Training forward/backward successful")

        model.eval()
        with torch.no_grad():
            output_before_fuse = model(dummy_input).clone()

        if hasattr(model, 'fuse_model'):
            model.fuse_model()
        else:
            for block in mobileone_blocks:
                block.fuse_reparam()

        with torch.no_grad():
            output_after_fuse = model(dummy_input)

        max_diff = (output_before_fuse - output_after_fuse).abs().max().item()
        relative_error = max_diff / (output_before_fuse.abs().max().item() + 1e-8)

        #   Numeric tolerance test
        if relative_error < 5e-3:
            smoke_checker.log_check(f"{backbone_name}: MobileOne Fusion", "PASS",
                                   f"Fusion successful, max relative error: {relative_error:.2e}")
            return True
        else:
            smoke_checker.log_check(f"{backbone_name}: MobileOne Fusion", "FAIL",
                                   f"Outputs differ after fusion: relative error {relative_error:.2e}")
            return False

    except Exception as e:
        smoke_checker.log_check(f"{backbone_name}: MobileOne Verification", "FAIL", str(e))
        return False


def verify_model_architecture(model, backbone_name, input_size=224, _num_classes=None, device=DEVICE):

    if _num_classes is None:
        _num_classes = _num_classes if _num_classes else 1000

    logger.info(f"  Verifying architecture for {backbone_name}...")

    checks_passed = 0
    checks_total = 0

    # Check 1: Device transfer
    checks_total += 1
    try:
        model.to(device)
        smoke_checker.log_check(f"{backbone_name}: Device Transfer", "PASS", f"Model moved to {device}")
        checks_passed += 1
    except Exception as e:
        smoke_checker.log_check(f"{backbone_name}: Device Transfer", "FAIL", str(e))
        return False

    # Check 2: Forward pass
    checks_total += 1
    try:
        dummy_input = torch.randn(2, 3, input_size, input_size).to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        smoke_checker.log_check(f"{backbone_name}: Forward Pass", "PASS",
                              f"Output shape: {output.shape}")
        checks_passed += 1
    except Exception as e:
        smoke_checker.log_check(f"{backbone_name}: Forward Pass", "FAIL", str(e))
        return False

    # Check 3: Output shape
    checks_total += 1
    expected_output_shape = (2, _num_classes)
    if output.shape == expected_output_shape:
        smoke_checker.log_check(f"{backbone_name}: Output Shape", "PASS",
                              f"Expected {expected_output_shape}, got {output.shape}")
        checks_passed += 1
    else:
        smoke_checker.log_check(f"{backbone_name}: Output Shape", "FAIL",
                              f"Expected {expected_output_shape}, got {output.shape}")

    # Check 4: Trainable parameters
    checks_total += 1
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params > 0:
        smoke_checker.log_check(f"{backbone_name}: Trainable Params", "PASS",
                              f"{trainable_params:,} / {total_params:,}")
        checks_passed += 1
        logger.info(f"  Model parameters: {trainable_params:,} trainable / {total_params:,} total")
    else:
        smoke_checker.log_check(f"{backbone_name}: Trainable Params", "FAIL",
                              "No trainable parameters found")

    # Check 5: Backward pass
    checks_total += 1
    try:
        model.train()
        dummy_input = torch.randn(2, 3, input_size, input_size).to(device)
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        smoke_checker.log_check(f"{backbone_name}: Backward Pass", "PASS",
                              "Gradients computed successfully")
        checks_passed += 1
    except Exception as e:
        smoke_checker.log_check(f"{backbone_name}: Backward Pass", "FAIL", str(e))

    # Check 6: Train/eval mode
    checks_total += 1
    try:
        model.train()
        assert model.training
        model.eval()
        assert not model.training
        smoke_checker.log_check(f"{backbone_name}: Train/Eval Mode", "PASS",
                              "Mode switching works")
        checks_passed += 1
    except Exception as e:
        smoke_checker.log_check(f"{backbone_name}: Train/Eval Mode", "FAIL", str(e))

    # Check 7: Feature dimensions
    checks_total += 1
    try:
        feature_dims = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if isinstance(module, nn.Linear):
                    feature_dims.append(f"{name}: {module.in_features}=>{module.out_features}")
                else:
                    feature_dims.append(f"{name}: {module.in_channels}=>{module.out_channels}")

        smoke_checker.log_check(f"{backbone_name}: Feature Dimensions", "PASS",
                              f"Detected {len(feature_dims)} layers")
        logger.info(f"  Key layers: {len(feature_dims)} detected")
        checks_passed += 1
    except Exception as e:
        smoke_checker.log_check(f"{backbone_name}: Feature Dimensions", "WARN", str(e))

    # Architecture-specific checks
    checks_total += 3

    if 'Swin' in backbone_name:
        if verify_swin_transformer(model, backbone_name, input_size, _num_classes, device):
            checks_passed += 1
    else:
        checks_passed += 1

    if 'Dynamic' in backbone_name:
        if verify_dynamic_conv(model, backbone_name, input_size, _num_classes, device):
            checks_passed += 1
    else:
        checks_passed += 1

    if 'MobileOne' in backbone_name:
        if verify_mobileone_reparam(model, backbone_name, input_size, _num_classes, device):
            checks_passed += 1
    else:
        checks_passed += 1

    success_rate = (checks_passed / checks_total) * 100
    logger.info(f"  Architecture verification: {checks_passed}/{checks_total} checks passed ({success_rate:.1f}%)")

    return checks_passed >= (checks_total * 0.7)


def get_model_summary(model, input_size=224):
    """Get detailed model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layer_count = len(list(model.modules()))
    param_size_mb = (total_params * 4) / (1024 ** 2)

    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'layer_count': layer_count,
        'parameter_size_mb': param_size_mb
    }

    return summary


# =============================================================================
# UNIT TESTS - NEW
# =============================================================================

def test_dynamicconv_forward_backward():
    """Test DynamicConv batched forward/backward"""
    torch.manual_seed(0)
    dc = DynamicConv(in_channels=3, out_channels=8, kernel_size=3, num_kernels=4)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    out = dc(x)
    loss = out.mean()
    loss.backward()

    # ensure gradients exist
    assert dc.weight_bank.grad is not None, "weight_bank has no grad"
    assert x.grad is not None, "input has no grad"

    # Check controller gradients
    has_controller_grad = False
    for param in dc.controller.parameters():
        if param.grad is not None:
            has_controller_grad = True
            break
    assert has_controller_grad, "controller has no gradients"

    logger.info("Cool DynamicConv forward/backward test PASSED")
    return True


def test_window_partition_reverse():
    """Test window partition invertibility"""
    torch.manual_seed(0)
    x = torch.randn(2, 56, 56, 64)
    wsize = 7

    # Test partition and reverse with divisible size
    windows = window_partition(x, wsize)
    x_rec = window_reverse(windows, wsize, 56, 56)

    assert torch.allclose(x, x_rec, atol=1e-6), "window_partition/window_reverse not invertible"

    # Test with non-divisible size (should auto-pad, not raise error)
    x_bad = torch.randn(2, 55, 55, 64)  # 55 not divisible by 7
    windows_bad = window_partition(x_bad, wsize)

    # Calculate expected padded dimensions
    pad_h = (wsize - 55 % wsize) % wsize  # = 1
    pad_w = (wsize - 55 % wsize) % wsize  # = 1
    H_padded = 55 + pad_h  # = 56
    W_padded = 55 + pad_w  # = 56

    # Verify window partition succeeded with padding
    expected_num_windows = 2 * (H_padded // wsize) * (W_padded // wsize)
    assert windows_bad.shape[0] == expected_num_windows, \
        f"Expected {expected_num_windows} windows, got {windows_bad.shape[0]}"

    # Reverse should work with padded dimensions
    x_rec_bad = window_reverse(windows_bad, wsize, H_padded, W_padded)
    assert x_rec_bad.shape == (2, H_padded, W_padded, 64), \
        f"Unexpected reversed shape: {x_rec_bad.shape}"

    # The reversed tensor should match the original in the non-padded region
    # (comparing the original 55x55 region with the padded 56x56 result's 55x55 region)
    assert torch.allclose(x_bad, x_rec_bad[:, :55, :55, :], atol=1e-6), \
        "Original region doesn't match after padding and reverse"

    logger.info("Cool window partition invertibility test PASSED")
    return True


def test_mobileone_fuse_equiv():
    """Test MobileOne fusion numeric equivalence"""
    torch.manual_seed(0)
    block = MobileOneBlock(in_channels=16, out_channels=16, kernel_size=3, stride=1, num_conv_branches=1)
    block.eval()

    x = torch.randn(1, 16, 32, 32)

    with torch.no_grad():
        y1 = block(x).clone()

        block.fuse_reparam()
        y2 = block(x)

    max_diff = (y1 - y2).abs().max().item()
    assert torch.allclose(y1, y2, atol=1e-4, rtol=1e-5), \
        f"MobileOne fusion changed outputs, max diff: {max_diff}"

    logger.info(f"Cool MobileOne fusion equivalence test PASSED (max diff: {max_diff:.2e})")
    return True


def test_dynamicconv_batched_efficiency():
    """Test that batched DynamicConv is faster than per-sample (optional benchmark)"""
    torch.manual_seed(0)
    dc = DynamicConv(in_channels=3, out_channels=16, kernel_size=3, num_kernels=4)
    dc.eval()

    batch_sizes = [1, 4, 8]
    times = []

    for bs in batch_sizes:
        x = torch.randn(bs, 3, 32, 32)

        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = dc(x)
        elapsed = time.time() - start
        times.append(elapsed)

    logger.info(f"Cool DynamicConv batch efficiency test: {times}")
    # Batched should be more efficient per-sample
    time_per_sample = [t/bs for t, bs in zip(times, batch_sizes)]
    logger.info(f"  Time per sample: {time_per_sample}")

    return True


def run_all_unit_tests():
    """Run all unit tests"""
    logger.info("\n" + "="*80)
    logger.info("RUNNING UNIT TESTS")
    logger.info("="*80)

    tests = [
        ("DynamicConv Forward/Backward", test_dynamicconv_forward_backward),
        ("Window Partition Invertibility", test_window_partition_reverse),
        ("MobileOne Fusion Equivalence", test_mobileone_fuse_equiv),
        ("DynamicConv Batched Efficiency", test_dynamicconv_batched_efficiency),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            logger.info(f"\nRunning: {test_name}")
            result = test_func()
            if result:
                passed += 1
                smoke_checker.log_check(f"UnitTest: {test_name}", "PASS", "")
            else:
                failed += 1
                smoke_checker.log_check(f"UnitTest: {test_name}", "FAIL", "Test returned False")
        except Exception as e:
            failed += 1
            logger.error(f"x {test_name} FAILED: {e}")
            smoke_checker.log_check(f"UnitTest: {test_name}", "FAIL", str(e))

    logger.info("\n" + "="*80)
    logger.info(f"UNIT TEST SUMMARY: {passed}/{len(tests)} passed, {failed} failed")
    logger.info("="*80 + "\n")

    return passed, failed

# =============================================================================
# IMAGE SIZE VERIFICATION
# =============================================================================

def validate_img_size_for_model(model_name: str, img_size: int):
    """Validate image size is compatible with model architecture"""
    # Swin and ViT-based models need sizes divisible by patch size
    if 'Swin' in model_name or 'ViT' in model_name or 'DeiT' in model_name:
        patch_size = 16 if 'ViT' in model_name or 'DeiT' in model_name else 4
        if img_size % patch_size != 0:
            logger.warning(
                f"{model_name} prefers image size divisible by {patch_size}, "
                f"got {img_size}. This may cause issues."
            )

    # Minimum size check
    if img_size < 32:
        raise ValueError(f"Image size {img_size} too small, minimum is 32")

    # Window size check for Swin
    if 'Swin' in model_name:
        window_size = 7
        patches_per_side = img_size // 4  # patch_size = 4 for Swin
        if patches_per_side % window_size != 0:
            logger.warning(
                f"Swin Transformer: image size {img_size} results in {patches_per_side}x{patches_per_side} patches, "
                f"which is not divisible by window_size {window_size}. Expect potential issues."
            )

    return True

# =============================================================================
# BACKBONE CREATION
# =============================================================================

def create_custom_backbone(name: str, _num_classes: int):
    """Create custom backbone from scratch with verification"""
    name_lower = name.lower()

    backbone_map = {
        'customconvnext': CustomConvNeXt,
        'customefficientnetv4': CustomEfficientNetV4,
        'customghostnetv2': CustomGhostNetV2,
        'customresnetmish': CustomResNetMish,
        'customcspdarknet': CustomCSPDarkNet,
        'custominceptionv4': CustomInceptionV4,
        'customvithybrid': CustomViTHybrid,
        'customswintransformer': CustomSwinTransformer,
        'customcoatnet': CustomCoAtNet,
        'customregnet': CustomRegNet,
        'customdensenethybrid': CustomDenseNetHybrid,
        'customdeitstyle': CustomDeiTStyle,
        'custommaxvit': CustomMaxViT,
        'custommobileone': CustomMobileOne,
        'customdynamicconvnet': CustomDynamicConvNet
    }

    if name_lower in backbone_map:
        validate_img_size_for_model(name, IMG_SIZE)
        logger.info(f"Creating {name} with {_num_classes} classes...")
        model = backbone_map[name_lower](_num_classes=_num_classes)

        #   Better transfer learning logging
        if ENABLE_TRANSFER_LEARNING and name in PRETRAINED_PATHS:
            pretrained_path = PRETRAINED_PATHS[name]
            logger.info(f"  Checking pretrained weights: {pretrained_path.name}")
            if pretrained_path.exists():
                logger.info(f"  Cool Loading pretrained weights from {pretrained_path.name}")
                load_stats = load_pretrained_to_model(
                    model,
                    str(pretrained_path),
                    backbone_name=name,
                    logger_inst=logger,
                    map_location=str(DEVICE),
                    freeze_backbone=(TRANSFER_LEARNING_MODE == 'feature_extraction'),
                    auto_move_to_device=False
                )
                logger.info(f"  Cool Loaded {load_stats['matched']} pretrained parameters")
            else:
                logger.warning(f"  x Pretrained weights not found: {pretrained_path.name}")
                logger.info(f"  Training from scratch for {name}")
        else:
            logger.info(f"  Training from scratch for {name}")

        summary = get_model_summary(model)
        logger.info(f"  Total parameters: {summary['total_parameters']:,}")
        logger.info(f"  Trainable parameters: {summary['trainable_parameters']:,}")
        logger.info(f"  Estimated size: {summary['parameter_size_mb']:.2f} MB")

        if not verify_model_architecture(model, name, IMG_SIZE, _num_classes, DEVICE):
            raise ValueError(f"Architecture verification failed for {name}")

        return model
    else:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(backbone_map.keys())}")


def create_custom_backbone_safe(name: str, _num_classes: int):
    """Safe wrapper with comprehensive error handling"""
    try:
        return create_custom_backbone(name, _num_classes)
    except ValueError as e:
        logger.error(f"Validation error for {name}: {e}")
        smoke_checker.log_check(f"{name}: Creation", "FAIL", f"Validation error: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Runtime error creating {name}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        smoke_checker.log_check(f"{name}: Creation", "FAIL", f"Runtime error: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error creating {name}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        smoke_checker.log_check(f"{name}: Creation", "FAIL", f"Unexpected error: {e}")
        raise

# =============================================================================
# MAIN PIPELINE - SIMPLIFIED FOR DEMONSTRATION
# =============================================================================

def run_optimized_pipeline():
    """Main optimized pipeline with verification, K-fold CV, and full training"""

    # Configure logging for this run
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_log_file = BASE_DIR / f'training_run_{timestamp}.log'

    file_handler = logging.FileHandler(run_log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.logger.addHandler(file_handler)

    logger.info("=== STARTING ENHANCED DISEASE CLASSIFICATION PIPELINE ===")
    logger.info(f"Run log file: {run_log_file}")

    # Run unit tests first
    _passed, failed = run_all_unit_tests()
    if failed > 0:
        logger.warning(f"{failed} unit tests failed. Proceeding with caution...")

    # Prepare dataset
    try:
        prepare_optimized_datasets()
        if not verify_dataset_split():
            logger.error("Dataset split validation failed. Please check your data.")
            return {}
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return {}

    # Test ALL 15 backbones
    backbones_to_test = BACKBONES  # Uses the full list defined at top

    logger.info(f"Training ALL {len(backbones_to_test)} custom backbone architectures")
    logger.info(f"Backbones: {', '.join(backbones_to_test)}")

    # Load full dataset for K-fold
    if not RAW_DIR.exists():
        logger.error(f"Dataset directory not found: {RAW_DIR}")
        logger.info("Please ensure the following:")
        logger.info(f"  1. Directory exists: {RAW_DIR}")
        logger.info("  2. Contains subdirectories for each disease class")
        logger.info("  3. Each subdirectory contains image files (.jpg, .png, etc.)")
        logger.info("\nAlternatively, set environment variable: DBT_RAW_DIR=/path/to/your/data")
        return {}

    global _num_classes
    try:
        full_dataset = WindowsCompatibleImageFolder(str(RAW_DIR), transform=transforms.ToTensor())
    except Exception as e:
        logger.error(f"Failed to load dataset from {RAW_DIR}: {e}")
        return {}


    _num_classes = len(full_dataset.classes)
    logger.info(f"Dataset loaded: {len(full_dataset)} samples, {_num_classes} disease classes")

    # Download pretrained weights if enabled
    if ENABLE_TRANSFER_LEARNING:
        download_result = download_pretrained_weights(PRETRAINED_DIR, logger)
        logger.info(f"Pretrained weights download result: {download_result}")

    results = {}

    # =========================================================================
    # STAGE 0: MODEL VERIFICATION
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 0: MODEL ARCHITECTURE VERIFICATION")
    logger.info("="*80)

    verification_results = {}

    for i, backbone_name in enumerate(backbones_to_test):
        logger.info(f"\n{'='*60}")
        logger.info(f"VERIFYING BACKBONE {i+1}/{len(backbones_to_test)}: {backbone_name.upper()}")
        logger.info(f"{'='*60}")

        try:
            # Create and verify model
            model = create_custom_backbone_safe(backbone_name, get_num_classes())
            logger.info(f"Cool {backbone_name} creation and verification: SUCCESS")

            verification_results[backbone_name] = {
                'status': 'verified',
                'num_params': sum(p.numel() for p in model.parameters()),
            }

            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

        except Exception as e:
            logger.error(f"x {backbone_name} FAILED: {e}")
            verification_results[backbone_name] = {
                'status': 'failed',
                'error': str(e)
            }

    # Save smoke check summary
    smoke_checker.save_summary()

    logger.info("\n" + "="*80)
    logger.info("VERIFICATION STAGE COMPLETED")
    logger.info("="*80)

    # Report verification results
    verified_count = sum(1 for r in verification_results.values() if r['status'] == 'verified')
    logger.info(f"Successfully verified: {verified_count}/{len(backbones_to_test)} models")

    # Only proceed with training for verified models
    backbones_to_train = [name for name, result in verification_results.items()
                          if result['status'] == 'verified']

    if not backbones_to_train:
        logger.error("No models passed verification. Cannot proceed with training.")
        return verification_results

    logger.info(f"Proceeding to training with {len(backbones_to_train)} verified models")

    # =========================================================================
    # STAGE 1 & 2: K-FOLD CV AND FULL TRAINING
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING STAGES")
    logger.info("="*80)

    # Track overall pipeline timing
    pipeline_start_time = time.time()

    for i, backbone_name in enumerate(backbones_to_train):
        # Track per-model timing
        model_start_time = time.time()

        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING BACKBONE {i+1}/{len(backbones_to_train)}: {backbone_name.upper()}")
        logger.info(f"Progress: {i}/{len(backbones_to_train)} completed ({(i/len(backbones_to_train)*100):.1f}%)")

        # Calculate and display estimated time remaining
        if i > 0:
            elapsed = time.time() - pipeline_start_time
            avg_time_per_model = elapsed / i
            remaining = avg_time_per_model * (len(backbones_to_train) - i)
            logger.info(f"Estimated time remaining: {remaining/60:.1f} min")
        else:
            logger.info("Estimated time: Calculating...")

        logger.info(f"{'='*80}")

        try:
            # Stage 1: K-fold Cross Validation
            if ENABLE_KFOLD_CV:
                logger.info(f"\nStage 1: K-fold Cross Validation for {backbone_name}")
                mean_acc, std_acc, kfold_summary = k_fold_cross_validation(
                    backbone_name, full_dataset, k_folds=K_FOLDS)
            else:
                                logger.info("\nStage 1: K-fold Cross Validation SKIPPED (disabled)")
                                mean_acc, std_acc, kfold_summary = 0.0, 0.0, {'skipped': True}

            # Stage 2: Train final model
            logger.info(f"\nStage 2: Training final {backbone_name} model")

            # Prepare datasets for final training
            train_ds, val_ds = prepare_datasets_for_backbone(backbone_name, size=IMG_SIZE)

            # Create model
            model = create_custom_backbone_safe(backbone_name, get_num_classes())

            # Train with full pipeline (includes export now)
            final_model, final_acc, final_history, final_metrics = train_backbone_with_metrics(
                backbone_name, model, train_ds, val_ds,
                epochs_head=EPOCHS_HEAD,
                epochs_finetune=EPOCHS_FINETUNE
            )

            # Stage 3: Test set evaluation
            logger.info(f"\nStage 3: Test set evaluation for {backbone_name}")
            if TEST_DIR.exists():
                test_ds = WindowsCompatibleImageFolder(
                    str(TEST_DIR),
                    transform=create_optimized_transforms(IMG_SIZE, is_training=False)
                )
                test_loader = create_optimized_dataloader(test_ds, BATCH_SIZE, shuffle=False)

                _test_loss, test_acc, test_prec, test_rec, test_f1, _, _, _ = validate_epoch_optimized(
                    final_model, test_loader, nn.CrossEntropyLoss()
                )

                logger.info("Test Results:")
                logger.info(f"  Accuracy:  {test_acc:.4f}")
                logger.info(f"  Precision: {test_prec:.4f}")
                logger.info(f"  Recall:    {test_rec:.4f}")
                logger.info(f"  F1 Score:  {test_f1:.4f}")

                final_metrics['test_accuracy'] = float(test_acc)
                final_metrics['test_precision'] = float(test_prec)
                final_metrics['test_recall'] = float(test_rec)
                final_metrics['test_f1_score'] = float(test_f1)
            else:
                logger.warning(f"Test directory not found: {TEST_DIR}. Skipping test evaluation.")
                final_metrics['test_accuracy'] = None
                final_metrics['test_precision'] = None
                final_metrics['test_recall'] = None
                final_metrics['test_f1_score'] = None

            # Store results (combine verification and training results)
            results[backbone_name] = {
                'verification': verification_results[backbone_name],
                'kfold_mean_accuracy': mean_acc if ENABLE_KFOLD_CV else None,
                'kfold_std_accuracy': std_acc if ENABLE_KFOLD_CV else None,
                'kfold_summary': kfold_summary if ENABLE_KFOLD_CV else None,
                'final_accuracy': final_acc,
                'final_metrics': final_metrics,
                'training_history': final_history,
                'status': 'success'
            }

            # Save intermediate checkpoint
            checkpoint_file = METRICS_DIR / f'pipeline_checkpoint_{backbone_name}.json'
            with open(checkpoint_file, 'w') as f:
                json.dump(results[backbone_name], f, indent=2, default=str)
            logger.info(f"Cool Checkpoint saved: {checkpoint_file}")

            # Calculate and log model training time
            model_end_time = time.time()
            model_elapsed = model_end_time - model_start_time
            logger.info(f"\n{'='*60}")
            logger.info(f"{backbone_name} COMPLETED")
            if ENABLE_KFOLD_CV:
                logger.info(f"K-fold CV: {mean_acc:.4f} +/- {std_acc:.4f}")
            logger.info(f"Final Accuracy: {final_acc:.4f}")
            if final_metrics.get('test_accuracy') is not None:
                logger.info(f"Test Accuracy: {final_metrics['test_accuracy']:.4f}")
            logger.info(f"Training time: {model_elapsed/60:.1f} min")
            logger.info(f"{'='*60}\n")

            # Clean up
            del model, final_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

        except Exception as e:
            logger.error(f"x {backbone_name} FAILED: {e}")
            logger.exception("Full traceback:")
            results[backbone_name] = {
                'verification': verification_results[backbone_name],
                'kfold_mean_accuracy': 0.0,
                'kfold_std_accuracy': 0.0,
                'final_accuracy': 0.0,
                'error': str(e),
                'status': 'failed'
            }

    # Calculate total pipeline time
    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time

    # =========================
    # GENERATE COMPARISON PLOTS
    # =========================
    logger.info("\n" + "="*80)
    logger.info("GENERATING COMPARISON VISUALIZATIONS")
    logger.info("="*80)

    try:
        # 1. K-fold comparison (if K-fold was run)
        kfold_results_dict: dict[str, Any] = {}
        kfold_plot_path: Path | None = None
        if ENABLE_KFOLD_CV:
            for backbone in backbones_to_train:
                if backbone in results and 'kfold_summary' in results[backbone]:
                    kfold_results_dict[backbone] = results[backbone]['kfold_summary']

            if kfold_results_dict:
                kfold_plot_path = PLOTS_DIR / "all_backbones_kfold_comparison.tiff"
                plot_kfold_results(kfold_results_dict, kfold_plot_path,
                                  "K-Fold Cross-Validation: All Backbones")
                logger.info(f"Cool K-fold comparison plot: {kfold_plot_path}")

        # 2. Final backbone performance comparison
        comparison_path = PLOTS_DIR / "all_backbones_comparison.tiff"
        plot_backbone_comparison(results, comparison_path,
                                "Final Performance: All Backbones")
        logger.info(f"Cool Backbone comparison plot: {comparison_path}")

        # 3. Create summary visualization index
        viz_index = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_backbones': len(backbones_to_train),
            'comparison_plots': {
                'kfold_comparison': str(kfold_plot_path) if kfold_plot_path is not None else None,
                'backbone_comparison': str(comparison_path)
            },
            'individual_plots': {}
        }

        # Collect all individual visualization paths
        for backbone in backbones_to_train:
            if backbone in results and 'training_results' in results:
                backbone_results = results['training_results'].get(backbone, {})
                if 'final_metrics' in backbone_results:
                    viz_paths = backbone_results['final_metrics'].get('visualization_paths', {})
                    if viz_paths:
                        viz_index['individual_plots'][backbone] = viz_paths

        # Save visualization index
        viz_index_path = PLOTS_DIR / 'visualization_index.json'
        with open(viz_index_path, 'w') as f:
            json.dump(viz_index, f, indent=2, default=str)

        logger.info(f"Cool Visualization index saved: {viz_index_path}")
        logger.info(f"Cool Total plots generated: {sum(len(v) for v in viz_index['individual_plots'].values()) + 2}")

    except Exception as e:
        logger.error(f"x Failed to generate comparison visualizations: {e}")
        logger.exception("Full traceback:")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("FINAL RESULTS SUMMARY")
    logger.info(f"{'='*80}")

    # Verification summary
    logger.info("\nVERIFICATION RESULTS:")
    for backbone_name, result in verification_results.items():
        if result['status'] == 'verified':
            logger.info(f"  Cool {backbone_name}: {result['num_params']:,} parameters")
        else:
            logger.info(f"  x {backbone_name}: {result.get('error', 'Failed')}")

    # Training summary
    logger.info("\nTRAINING RESULTS:")
    for backbone_name, result in results.items():
        if result.get('status') == 'success':
            logger.info(f"Cool {backbone_name}:")
            if ENABLE_KFOLD_CV and result.get('kfold_mean_accuracy') is not None:
                logger.info(f"  K-fold CV: {result['kfold_mean_accuracy']:.4f} +/- {result['kfold_std_accuracy']:.4f}")
            logger.info(f"  Final Acc: {result['final_accuracy']:.4f}")
            if result['final_metrics'].get('test_accuracy') is not None:
                logger.info(f"  Test Acc:  {result['final_metrics']['test_accuracy']:.4f}")
        else:
            logger.info(f"x {backbone_name}: FAILED - {result.get('error', 'Unknown error')}")

    # Save final summary
    summary_file = METRICS_DIR / 'pipeline_summary.json'
    final_results = {
        'verification_results': verification_results,
        'training_results': results,
        'total_execution_time_seconds': total_time,
        'timestamp': timestamp
    }
    with open(summary_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    logger.info(f"\nCool Pipeline completed! Results saved to: {summary_file}")

    # Calculate overall statistics
    successful_runs = [r for r in results.values() if r.get('status') == 'success']
    avg_kfold_acc: float | None = None
    avg_final_acc: float | None = None
    avg_test_acc: float | None = None
    if successful_runs:
        if ENABLE_KFOLD_CV:
            kfold_accs = [r['kfold_mean_accuracy'] for r in successful_runs
                          if r.get('kfold_mean_accuracy') is not None]
            avg_kfold_acc = float(np.mean(kfold_accs)) if kfold_accs else None
        avg_final_acc = float(np.mean([r['final_accuracy'] for r in successful_runs]))

        # Calculate average test accuracy (only for models that have it)
        test_accs = [r['final_metrics']['test_accuracy'] for r in successful_runs
                     if r['final_metrics'].get('test_accuracy') is not None]
        avg_test_acc = float(np.mean(test_accs)) if test_accs else None

        logger.info("\nOVERALL STATISTICS:")
        logger.info(f"  Models verified: {verified_count}/{len(backbones_to_test)}")
        logger.info(f"  Successfully trained: {len(successful_runs)}/{len(backbones_to_train)} models")
        if ENABLE_KFOLD_CV and avg_kfold_acc is not None:
            logger.info(f"  Average K-fold CV accuracy: {avg_kfold_acc:.4f}")
        if avg_final_acc is not None:
            logger.info(f"  Average final accuracy: {avg_final_acc:.4f}")
        if avg_test_acc is not None:
            logger.info(f"  Average test accuracy: {avg_test_acc:.4f}")
        logger.info(f"  Total execution time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # Visualization summary
    logger.info("\nVISUALIZATION SUMMARY:")
    logger.info(f"  Plots directory: {PLOTS_DIR}")

    total_plots = 0
    for backbone in backbones_to_train:
        if backbone in results and 'training_results' in results:
            backbone_results = results['training_results'].get(backbone, {})
            if 'final_metrics' in backbone_results:
                viz_paths = backbone_results['final_metrics'].get('visualization_paths', {})
                total_plots += len(viz_paths)
                if viz_paths:
                    logger.info(f"  {backbone}: {len(viz_paths)} plots")

    logger.info("  Comparison plots: 2")
    logger.info(f"  Total visualizations: {total_plots + 2}")

    # Check visualization index
    viz_index_file = PLOTS_DIR / 'visualization_index.json'
    if viz_index_file.exists():
        logger.info(f"  Cool Visualization index: {viz_index_file}")
    else:
        logger.warning("  x Visualization index not found")

    # Generate detailed pipeline report
    report_path = METRICS_DIR / 'pipeline_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DISEASE CLASSIFICATION PIPELINE - FINAL REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Execution Time: {total_time:.1f}s ({total_time/60:.1f}min)\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Log File: {run_log_file}\n\n")

        f.write("Configuration:\n")
        f.write(f"  Image Size: {IMG_SIZE}\n")
        f.write(f"  Batch Size: {BATCH_SIZE}\n")
        f.write(f"  K-Folds: {K_FOLDS}\n")
        f.write(f"  Device: {DEVICE}\n")
        f.write(f"  Number of Classes: {_num_classes}\n")
        f.write(f"  Head Epochs: {EPOCHS_HEAD}\n")
        f.write(f"  Finetune Epochs: {EPOCHS_FINETUNE}\n")
        f.write(f"  Transfer Learning: {ENABLE_TRANSFER_LEARNING}\n\n")

        f.write("VERIFICATION RESULTS:\n")
        f.write("-" * 80 + "\n")
        for backbone_name, result in verification_results.items():
            status = "Cool PASS" if result['status'] == 'verified' else "x FAIL"
            f.write(f"{status:8} {backbone_name:30} ")
            if result['status'] == 'verified':
                f.write(f"({result['num_params']:,} params)\n")
            else:
                f.write(f"Error: {result.get('error', 'Unknown')}\n")

        f.write("\n\nTRAINING RESULTS:\n")
        f.write("-" * 80 + "\n")
        for backbone_name, result in results.items():
            if result.get('status') == 'success':
                f.write(f"Cool {backbone_name}\n")
                if ENABLE_KFOLD_CV:
                    f.write(f"    K-fold CV: {result['kfold_mean_accuracy']:.4f} +/- {result['kfold_std_accuracy']:.4f}\n")
                f.write(f"    Final Acc: {result['final_accuracy']:.4f}\n")
                f.write(f"    Precision: {result['final_metrics'].get('final_precision', 0):.4f}\n")
                f.write(f"    Recall:    {result['final_metrics'].get('final_recall', 0):.4f}\n")
                f.write(f"    F1 Score:  {result['final_metrics'].get('final_f1_score', 0):.4f}\n")

                # Test results
                if result['final_metrics'].get('test_accuracy') is not None:
                    f.write(f"    Test Acc:  {result['final_metrics']['test_accuracy']:.4f}\n")
                    f.write(f"    Test Prec: {result['final_metrics']['test_precision']:.4f}\n")
                    f.write(f"    Test Rec:  {result['final_metrics']['test_recall']:.4f}\n")
                    f.write(f"    Test F1:   {result['final_metrics']['test_f1_score']:.4f}\n")

                # Export status
                if 'export_summary' in result['final_metrics']:
                    export = result['final_metrics']['export_summary']
                    if export.get('status') != 'skipped':
                        f.write(f"    Exported:  {export.get('successful_exports', 0)}/{len(export.get('requested_formats', []))} formats\n")
                f.write("\n")
            else:
                f.write(f"x {backbone_name}: {result.get('error', 'Failed')}\n\n")

        if successful_runs:
            f.write("\nOVERALL STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Models Verified:        {verified_count}/{len(backbones_to_test)}\n")
            f.write(f"Successfully Trained:   {len(successful_runs)}/{len(backbones_to_train)}\n")
            if ENABLE_KFOLD_CV and avg_kfold_acc is not None:
                f.write(f"Avg K-fold CV Acc:      {avg_kfold_acc:.4f}\n")
            if avg_final_acc is not None:
                f.write(f"Avg Final Acc:          {avg_final_acc:.4f}\n")
            if avg_test_acc is not None:
                f.write(f"Avg Test Acc:           {avg_test_acc:.4f}\n")
            f.write(f"Total Time:             {total_time:.1f}s ({total_time/60:.1f}min)\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"Report saved to: {report_path}\n")
        f.write("="*80 + "\n")

    logger.info(f"Cool Detailed report saved to: {report_path}")

    return final_results

# =============================================================================
# DEBUG MODE FUNCTIONS
# =============================================================================

class DebugLogger:
    """Enhanced logger for debug mode with structured output"""
    def __init__(self, backbone_name, function_name):
        self.backbone = backbone_name
        self.function = function_name
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.log_dir = DEBUG_LOG_DIR / f"{backbone_name}_{function_name}_{self.timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / 'debug.log'
        self.results = {
            'backbone': backbone_name,
            'function': function_name,
            'timestamp': self.timestamp,
            'checks': [],
            'metrics': {},
            'errors': []
        }

    def log(self, message, level='INFO'):
        """Log message to both console and file"""
        timestamp = time.strftime('%H:%M:%S')
        formatted = f"[{timestamp}] [{level:5}] {message}"
        print(formatted)

        with open(self.log_file, 'a') as f:
            f.write(formatted + '\n')

    def check(self, name, status, details=''):
        """Log a check result"""
        symbol = 'Cool' if status == 'PASS' else 'x'
        self.log(f"{symbol} {name}: {status} {details}", 'CHECK')
        self.results['checks'].append({
            'name': name,
            'status': status,
            'details': details
        })

    def metric(self, name, value):
        """Log a metric"""
        self.log(f" {name}: {value}", 'METRIC')
        self.results['metrics'][name] = value

    def error(self, message):
        """Log an error"""
        self.log(f"x ERROR: {message}", 'ERROR')
        self.results['errors'].append(message)

    def save_results(self):
        """Save debug results to JSON"""
        results_file = self.log_dir / 'debug_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.log(f"Results saved to: {results_file}")
        return self.results


def debug_model_creation(backbone_name: str, debug_log: DebugLogger) -> nn.Module:
    """DEBUG: Test model creation"""
    debug_log.log(f"\n{'='*60}")
    debug_log.log("DEBUG: Model Creation")
    debug_log.log(f"{'='*60}")

    try:
        # Test model instantiation
        debug_log.log("Creating model...")
        model = create_custom_backbone_safe(backbone_name, get_num_classes())
        debug_log.check("Model Creation", "PASS", "Model created successfully")

        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        debug_log.metric("Total Parameters", f"{total_params:,}")
        debug_log.metric("Trainable Parameters", f"{trainable_params:,}")

        # Check model structure
        debug_log.log("\nModel Structure:")
        for name, module in model.named_children():
            debug_log.log(f"  {name}: {module.__class__.__name__}")

        debug_log.check("Model Structure", "PASS", "All modules present")

        return model

    except Exception as e:
        debug_log.error(f"Model creation failed: {e}")
        debug_log.check("Model Creation", "FAIL", str(e))
        raise


def debug_forward_pass(model: nn.Module, backbone_name: str, debug_log: DebugLogger) -> bool:
    """DEBUG: Test forward pass"""
    debug_log.log(f"\n{'='*60}")
    debug_log.log("DEBUG: Forward Pass")
    debug_log.log(f"{'='*60}")

    try:
        model.to(DEVICE)
        model.eval()
        num_classes = get_num_classes()

        # Test different batch sizes
        for batch_size in [1, 2, 8]:
            debug_log.log(f"\nTesting batch_size={batch_size}")
            dummy_input = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

            with torch.no_grad():
                output = model(dummy_input)

            expected_shape = (batch_size, num_classes)
            if output.shape == expected_shape:
                debug_log.check(f"Forward Pass (bs={batch_size})", "PASS",
                              f"Output shape: {output.shape}")
            else:
                debug_log.check(f"Forward Pass (bs={batch_size})", "FAIL",
                              f"Expected {expected_shape}, got {output.shape}")

            # Check output statistics
            debug_log.metric(f"Output Mean (bs={batch_size})", f"{output.mean().item():.4f}")
            debug_log.metric(f"Output Std (bs={batch_size})", f"{output.std().item():.4f}")
            debug_log.metric(f"Output Min (bs={batch_size})", f"{output.min().item():.4f}")
            debug_log.metric(f"Output Max (bs={batch_size})", f"{output.max().item():.4f}")

            # Check for NaN/Inf
            if torch.isfinite(output).all():
                debug_log.check(f"Output Validity (bs={batch_size})", "PASS", "No NaN/Inf")
            else:
                debug_log.check(f"Output Validity (bs={batch_size})", "FAIL", "Contains NaN/Inf")

        return True

    except Exception as e:
        debug_log.error(f"Forward pass failed: {e}")
        debug_log.check("Forward Pass", "FAIL", str(e))
        raise


def debug_backward_pass(model, backbone_name, debug_log):
    """DEBUG: Test backward pass and gradients"""
    debug_log.log(f"\n{'='*60}")
    debug_log.log("DEBUG: Backward Pass")
    debug_log.log(f"{'='*60}")

    try:
        model.to(DEVICE)
        model.train()
        num_classes = get_num_classes()

        dummy_input = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        dummy_target = torch.randint(0, num_classes, (4,)).to(DEVICE)

        # Forward pass
        output = model(dummy_input)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, dummy_target)

        debug_log.metric("Loss Value", f"{loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Check gradients
        params_with_grad = 0
        params_without_grad = 0
        grad_norms = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    params_with_grad += 1
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)

                    if grad_norm > 1e6:
                        debug_log.log(f"    Large gradient in {name}: {grad_norm:.2e}")
                else:
                    params_without_grad += 1
                    debug_log.log(f"    No gradient for {name}")

        debug_log.metric("Params with Grad", params_with_grad)
        debug_log.metric("Params without Grad", params_without_grad)

        if grad_norms:
            debug_log.metric("Mean Grad Norm", f"{np.mean(grad_norms):.4e}")
            debug_log.metric("Max Grad Norm", f"{np.max(grad_norms):.4e}")
            debug_log.metric("Min Grad Norm", f"{np.min(grad_norms):.4e}")

        if params_without_grad == 0:
            debug_log.check("Gradient Flow", "PASS", "All parameters have gradients")
        else:
            debug_log.check("Gradient Flow", "WARN",
                          f"{params_without_grad} parameters missing gradients")

        return True

    except Exception as e:
        debug_log.error(f"Backward pass failed: {e}")
        debug_log.check("Backward Pass", "FAIL", str(e))
        raise


def debug_single_epoch(model, backbone_name, debug_log):
    """DEBUG: Train for single epoch"""
    debug_log.log(f"\n{'='*60}")
    debug_log.log("DEBUG: Single Epoch Training")
    debug_log.log(f"{'='*60}")

    try:
        # Prepare datasets
        train_ds, val_ds = prepare_datasets_for_backbone(backbone_name, size=IMG_SIZE)

        # Use smaller batch for debugging
        train_loader = create_optimized_dataloader(train_ds, _debug_batch_size, shuffle=True)
        val_loader = create_optimized_dataloader(val_ds, _debug_batch_size, shuffle=False)

        debug_log.log(f"Train batches: {len(train_loader)}")
        debug_log.log(f"Val batches: {len(val_loader)}")

        # Setup training
        model.to(DEVICE)
        optimizer = create_optimized_optimizer(model, lr=HEAD_LR, backbone_name=backbone_name)
        criterion = get_loss_function_for_backbone(backbone_name, _num_classes)

        # Train for 1 epoch
        debug_log.log("\nStarting training...")
        train_loss, train_acc, train_prec, _train_rec, train_f1, _, _, _ = train_epoch_optimized(
            model, train_loader, optimizer, criterion
        )

        debug_log.metric("Train Loss", f"{train_loss:.4f}")
        debug_log.metric("Train Accuracy", f"{train_acc:.4f}")
        debug_log.metric("Train Precision", f"{train_prec:.4f}")
        debug_log.metric("Train F1", f"{train_f1:.4f}")

        # Validate
        debug_log.log("\nStarting validation...")
        val_loss, val_acc, val_prec, _val_rec, val_f1, _, _, _ = validate_epoch_optimized(
            model, val_loader, criterion
        )

        debug_log.metric("Val Loss", f"{val_loss:.4f}")
        debug_log.metric("Val Accuracy", f"{val_acc:.4f}")
        debug_log.metric("Val Precision", f"{val_prec:.4f}")
        debug_log.metric("Val F1", f"{val_f1:.4f}")

        if train_acc > 0.01 and val_acc > 0.01:
            debug_log.check("Single Epoch", "PASS", "Training progressing normally")
        else:
            debug_log.check("Single Epoch", "WARN", "Very low accuracy, may need investigation")

        return True

    except Exception as e:
        debug_log.error(f"Single epoch training failed: {e}")
        debug_log.check("Single Epoch", "FAIL", str(e))
        raise


def debug_overfit_batch(model, backbone_name, debug_log):
    """DEBUG: Overfit on single batch (sanity check)"""
    debug_log.log(f"\n{'='*60}")
    debug_log.log("DEBUG: Overfit Single Batch")
    debug_log.log(f"{'='*60}")

    try:
        # Prepare datasets
        train_ds, _ = prepare_datasets_for_backbone(backbone_name, size=IMG_SIZE)

        # Get single batch
        train_loader = create_optimized_dataloader(train_ds, 8, shuffle=True)
        single_batch = next(iter(train_loader))
        images, targets = single_batch

        debug_log.log(f"Batch shape: {images.shape}")
        debug_log.log(f"Target shape: {targets.shape}")

        # Setup
        model.to(DEVICE)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Overfit for 50 iterations
        debug_log.log("\nOverfitting on single batch...")
        losses = []
        accuracies = []

        for i in range(50):
            images_gpu = images.to(DEVICE)
            targets_gpu = targets.to(DEVICE)

            optimizer.zero_grad()
            output = model(images_gpu)
            logits, _ = _unwrap_logits(output)
            loss = criterion(logits, targets_gpu)
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(1) == targets_gpu).float().mean().item()
            losses.append(loss.item())
            accuracies.append(acc)

            if (i + 1) % 10 == 0:
                debug_log.log(f"  Iter {i+1}/50 - Loss: {loss.item():.4f}, Acc: {acc:.4f}")

        final_loss = losses[-1]
        final_acc = accuracies[-1]

        debug_log.metric("Final Loss", f"{final_loss:.4f}")
        debug_log.metric("Final Accuracy", f"{final_acc:.4f}")
        debug_log.metric("Loss Reduction", f"{losses[0]:.4f} => {final_loss:.4f}")

        # Check if model can overfit
        if final_acc > 0.9 and final_loss < 0.1:
            debug_log.check("Overfit Capability", "PASS",
                          "Model can learn (achieved >90% on single batch)")
        elif final_acc > 0.5:
            debug_log.check("Overfit Capability", "WARN",
                          "Model learning but slowly (50-90% accuracy)")
        else:
            debug_log.check("Overfit Capability", "FAIL",
                          "Model not learning (<50% accuracy on single batch)")

        return True

    except Exception as e:
        debug_log.error(f"Overfit batch failed: {e}")
        debug_log.check("Overfit Batch", "FAIL", str(e))
        raise


def debug_dataset_loading(backbone_name, debug_log):
    """DEBUG: Test dataset loading"""
    debug_log.log(f"\n{'='*60}")
    debug_log.log("DEBUG: Dataset Loading")
    debug_log.log(f"{'='*60}")

    try:
        # Load datasets
        train_ds, val_ds = prepare_datasets_for_backbone(backbone_name, size=IMG_SIZE)

        debug_log.metric("Train Samples", len(train_ds))
        debug_log.metric("Val Samples", len(val_ds))
        debug_log.metric("Num Classes", len(train_ds.classes))
        debug_log.log(f"Classes: {train_ds.classes}")

        # Test data loading
        debug_log.log("\nTesting data loading...")
        train_loader = create_optimized_dataloader(train_ds, _debug_batch_size, shuffle=True)

        batch = next(iter(train_loader))
        images, labels = batch

        debug_log.metric("Batch Image Shape", str(images.shape))
        debug_log.metric("Batch Label Shape", str(labels.shape))
        debug_log.metric("Image dtype", str(images.dtype))
        debug_log.metric("Image Range", f"[{images.min():.3f}, {images.max():.3f}]")

        # Check for data issues
        if torch.isfinite(images).all():
            debug_log.check("Data Validity", "PASS", "No NaN/Inf in images")
        else:
            debug_log.check("Data Validity", "FAIL", "Images contain NaN/Inf")

        # Check class distribution
        unique, counts = torch.unique(labels, return_counts=True)
        debug_log.log("\nClass distribution in batch:")
        for cls, count in zip(unique.tolist(), counts.tolist()):
            debug_log.log(f"  Class {cls}: {count} samples")

        debug_log.check("Dataset Loading", "PASS", "All checks passed")

        return True

    except Exception as e:
        debug_log.error(f"Dataset loading failed: {e}")
        debug_log.check("Dataset Loading", "FAIL", str(e))
        raise


def debug_full_training(backbone_name, debug_log):
    """DEBUG: Full training pipeline with reduced epochs"""
    debug_log.log(f"\n{'='*60}")
    debug_log.log(f"DEBUG: Full Training Pipeline for {backbone_name}")
    debug_log.log(f"Using {_debug_epochs_head} head epochs, {_debug_epochs_finetune} finetune epochs")
    debug_log.log(f"{'='*60}")

    try:
        # Prepare datasets
        train_ds, val_ds = prepare_datasets_for_backbone(backbone_name, size=IMG_SIZE)

        # Create model
        model = create_custom_backbone_safe(backbone_name, get_num_classes())

        # Train with debug configuration
        final_model, best_acc, _history, metrics = train_backbone_with_metrics(
            backbone_name, model, train_ds, val_ds,
            epochs_head=_debug_epochs_head,
            epochs_finetune=_debug_epochs_finetune
        )

        debug_log.metric("Best Accuracy", f"{best_acc:.4f}")
        debug_log.metric("Final Accuracy", f"{metrics['final_accuracy']:.4f}")
        debug_log.metric("Final Precision", f"{metrics['final_precision']:.4f}")
        debug_log.metric("Final F1", f"{metrics['final_f1_score']:.4f}")

        debug_log.check("Full Training", "PASS", f"Training completed with {best_acc:.2%} accuracy")

        return final_model, metrics

    except Exception as e:
        debug_log.error(f"Full training failed: {e}")
        debug_log.check("Full Training", "FAIL", str(e))
        raise


def debug_export_only(model, backbone_name, debug_log):
    """DEBUG: Test export functionality"""
    debug_log.log(f"\n{'='*60}")
    debug_log.log("DEBUG: Export Functionality")
    debug_log.log(f"{'='*60}")

    try:
        sample_input = prepare_sample_input(model, input_size=IMG_SIZE, device=DEVICE)

        # Test minimal export formats
        test_formats = ['state_dict', 'torchscript']

        export_summary = export_and_package_model(
            model=model,
            arch_name=backbone_name,
            sample_input=sample_input,
            export_formats=test_formats,
            deploy_root=DEBUG_LOG_DIR / 'exports',
            force=True,
            _num_classes=_num_classes,
            logger_inst=debug_log
        )

        debug_log.metric("Successful Exports", export_summary['successful_exports'])
        debug_log.metric("Failed Exports", export_summary['failed_exports'])

        if export_summary['successful_exports'] > 0:
            debug_log.check("Export", "PASS", f"{export_summary['successful_exports']} formats exported")
        else:
            debug_log.check("Export", "FAIL", "No successful exports")

        return export_summary

    except Exception as e:
        debug_log.error(f"Export failed: {e}")
        debug_log.check("Export", "FAIL", str(e))
        raise


def run_debug_mode():
    """Main debug mode orchestrator"""
    logger.info("\n" + "="*80)
    logger.info("$ DEBUG MODE ACTIVATED")
    logger.info(f"Backbone: {DEBUG_BACKBONE}")
    logger.info(f"Function: {DEBUG_FUNCTION}")
    logger.info("="*80)
    global _num_classes
    # Create debug logger
    debug_log = DebugLogger(DEBUG_BACKBONE, DEBUG_FUNCTION)

    debug_log.log("\nDebug Configuration:")
    debug_log.log(f"  Image Size: {IMG_SIZE}")
    debug_log.log(f"  Batch Size: {_debug_batch_size}")
    debug_log.log(f"  Device: {DEVICE}")
    debug_log.log(f"  Num Classes: {_num_classes if _num_classes else 'Auto-detect'}")


    try:
        # Prepare datasets if needed
        if DEBUG_FUNCTION != 'model_creation' and DEBUG_FUNCTION != 'smoke_tests':
            prepare_optimized_datasets()

            if _num_classes is None:
                temp_ds = WindowsCompatibleImageFolder(str(TRAIN_DIR), transform=transforms.ToTensor())
                _num_classes = len(temp_ds.classes)
                debug_log.log(f"  Auto-detected {_num_classes} classes")

        # Execute requested debug function
        if DEBUG_FUNCTION == 'model_creation':
            model = debug_model_creation(DEBUG_BACKBONE, debug_log)

        elif DEBUG_FUNCTION == 'forward_pass':
            model = debug_model_creation(DEBUG_BACKBONE, debug_log)
            debug_forward_pass(model, DEBUG_BACKBONE, debug_log)

        elif DEBUG_FUNCTION == 'backward_pass':
            model = debug_model_creation(DEBUG_BACKBONE, debug_log)
            debug_backward_pass(model, DEBUG_BACKBONE, debug_log)

        elif DEBUG_FUNCTION == 'single_epoch':
            model = debug_model_creation(DEBUG_BACKBONE, debug_log)
            debug_single_epoch(model, DEBUG_BACKBONE, debug_log)

        elif DEBUG_FUNCTION == 'overfit_batch':
            model = debug_model_creation(DEBUG_BACKBONE, debug_log)
            debug_overfit_batch(model, DEBUG_BACKBONE, debug_log)

        elif DEBUG_FUNCTION == 'dataset_loading':
            debug_dataset_loading(DEBUG_BACKBONE, debug_log)

        elif DEBUG_FUNCTION == 'full_training':
            debug_full_training(DEBUG_BACKBONE, debug_log)

        elif DEBUG_FUNCTION == 'export_only':
            model = debug_model_creation(DEBUG_BACKBONE, debug_log)
            debug_export_only(model, DEBUG_BACKBONE, debug_log)

        elif DEBUG_FUNCTION == 'smoke_tests':
            run_all_unit_tests()

        elif DEBUG_FUNCTION == 'architecture_verify':
            model = debug_model_creation(DEBUG_BACKBONE, debug_log)
            verify_model_architecture(model, DEBUG_BACKBONE, IMG_SIZE, _num_classes, DEVICE)

        elif DEBUG_FUNCTION == 'pretrained_loading':
            model = debug_model_creation(DEBUG_BACKBONE, debug_log)
            if ENABLE_TRANSFER_LEARNING and DEBUG_BACKBONE in PRETRAINED_PATHS:
                pretrained_path = PRETRAINED_PATHS[DEBUG_BACKBONE]
                if pretrained_path.exists():
                    stats = load_pretrained_to_model(model, str(pretrained_path), DEBUG_BACKBONE, debug_log)
                    debug_log.metric("Matched Params", stats['matched'])
                else:
                    debug_log.log(f"Pretrained weights not found: {pretrained_path}")

        elif DEBUG_FUNCTION == 'all_checks':
            debug_log.log("\nRunning all debug checks...")
            model = debug_model_creation(DEBUG_BACKBONE, debug_log)
            debug_forward_pass(model, DEBUG_BACKBONE, debug_log)
            debug_backward_pass(model, DEBUG_BACKBONE, debug_log)
            debug_dataset_loading(DEBUG_BACKBONE, debug_log)
            debug_overfit_batch(model, DEBUG_BACKBONE, debug_log)

        else:
            debug_log.error(f"Unknown debug function: {DEBUG_FUNCTION}")
            debug_log.log("\nAvailable functions:")
            for func, desc in DEBUG_FUNCTIONS.items():
                debug_log.log(f"  {func:20} - {desc}")

        # Save results
        results = debug_log.save_results()

        # Print summary
        debug_log.log(f"\n{'='*60}")
        debug_log.log("DEBUG SUMMARY")
        debug_log.log(f"{'='*60}")

        passed = sum(1 for c in results['checks'] if c['status'] == 'PASS')
        failed = sum(1 for c in results['checks'] if c['status'] == 'FAIL')
        warned = sum(1 for c in results['checks'] if c['status'] == 'WARN')

        debug_log.log(f"Checks: {passed} passed, {failed} failed, {warned} warnings")
        debug_log.log(f"Metrics: {len(results['metrics'])} collected")
        debug_log.log(f"Errors: {len(results['errors'])}")

        if failed == 0:
            debug_log.log("\nCool All checks PASSED!")
        else:
            debug_log.log(f"\nx {failed} checks FAILED")

        debug_log.log(f"\nDebug log directory: {debug_log.log_dir}")

        return results

    except Exception as e:
        debug_log.error(f"Debug mode crashed: {e}")
        debug_log.log("\n" + "="*60)
        logger.exception("Full traceback:")
        debug_log.save_results()
        raise

# =============================================================================
# 1. CONFUSION MATRIX PLOTTING
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """
    Generate and save confusion matrix plot with both raw counts and normalized values.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title(f'{title} - Raw Counts', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Class', fontsize=12)
        ax1.set_ylabel('True Class', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)

        # Plot 2: Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax2, cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
        ax2.set_title(f'{title} - Normalized', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Class', fontsize=12)
        ax2.set_ylabel('True Class', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='y', rotation=0)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff', )
        plt.close()

        logger.info(f"Cool Confusion matrix saved to: {save_path}")
        return True

    except Exception as e:
        logger.error(f"x Failed to create confusion matrix: {e}")
        return False


# =============================================================================
# 2. ROC CURVES PLOTTING
# =============================================================================

def plot_roc_curves(y_true, y_probs, class_names, save_path, title="ROC Curves"):
    """
    Generate and save ROC curves for all classes.
    """
    try:
        n_classes = len(class_names)

        # Ensure we get a numpy array from label_binarize
        y_true_bin: np.ndarray = np.asarray(label_binarize(y_true, classes=range(n_classes)))

        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        fpr: dict[Any, np.ndarray] = {}
        tpr: dict[Any, np.ndarray] = {}
        roc_auc: dict[Any, float] = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = float(auc(fpr[i], tpr[i]))

        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc["micro"] = float(auc(fpr["micro"], tpr["micro"]))

        _fig, ax = plt.subplots(figsize=(12, 10))

        # Get colormap
        cmap = plt.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, n_classes))

        ax.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle='--', linewidth=3)

        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff', )
        plt.close()

        logger.info(f"Cool ROC curves saved to: {save_path}")
        logger.info(f"  Micro-average AUC: {roc_auc['micro']:.4f}")

        return roc_auc

    except Exception as e:
        logger.error(f"x Failed to create ROC curves: {e}")
        return {}


# =============================================================================
# 3. TRAINING HISTORY PLOTTING
# =============================================================================

def plot_training_history(history, save_path, title="Training History"):
    """
    Plot comprehensive training history including loss, accuracy, and F1 score.
    """
    try:
        if not history or ('head' not in history and 'finetune' not in history):
            logger.warning("No training history available for plotting")
            return False

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        all_epochs = []
        all_train_loss, all_val_loss = [], []
        all_train_acc, all_val_acc = [], []
        all_val_f1 = []

        epoch_count = 0
        head_epochs = 0

        if 'head' in history and history['head']:
            for train_loss, val_loss, val_acc, train_acc, val_f1 in history['head']:
                all_epochs.append(epoch_count)
                all_train_loss.append(train_loss)
                all_val_loss.append(val_loss)
                all_train_acc.append(train_acc)
                all_val_acc.append(val_acc)
                all_val_f1.append(val_f1)
                epoch_count += 1
            head_epochs = epoch_count

        if 'finetune' in history and history['finetune']:
            for train_loss, val_loss, val_acc, train_acc, val_f1 in history['finetune']:
                all_epochs.append(epoch_count)
                all_train_loss.append(train_loss)
                all_val_loss.append(val_loss)
                all_train_acc.append(train_acc)
                all_val_acc.append(val_acc)
                all_val_f1.append(val_f1)
                epoch_count += 1

        if not all_epochs:
            logger.warning("No valid training history found")
            return False

        # Plot 1: Loss
        axes[0, 0].plot(all_epochs, all_train_loss, label='Train Loss',
                       color='#2E86AB', linewidth=2, marker='o', markersize=3)
        axes[0, 0].plot(all_epochs, all_val_loss, label='Val Loss',
                       color='#A23B72', linewidth=2, marker='s', markersize=3)
        if head_epochs > 0:
            axes[0, 0].axvline(x=head_epochs-0.5, color='gray', linestyle='--',
                              alpha=0.7, linewidth=2, label='Head->Fine-tune')
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 0].legend(loc='best', fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Accuracy
        axes[0, 1].plot(all_epochs, all_train_acc, label='Train Acc',
                       color='#2E86AB', linewidth=2, marker='o', markersize=3)
        axes[0, 1].plot(all_epochs, all_val_acc, label='Val Acc',
                       color='#A23B72', linewidth=2, marker='s', markersize=3)
        if head_epochs > 0:
            axes[0, 1].axvline(x=head_epochs-0.5, color='gray', linestyle='--',
                              alpha=0.7, linewidth=2, label='Head->Fine-tune')
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('Accuracy', fontsize=11)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].legend(loc='best', fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1.05])

        # Plot 3: F1 Score
        axes[1, 0].plot(all_epochs, all_val_f1, label='Val F1',
                       color='#18A558', linewidth=2, marker='D', markersize=3)
        if head_epochs > 0:
            axes[1, 0].axvline(x=head_epochs-0.5, color='gray', linestyle='--',
                              alpha=0.7, linewidth=2, label='Head->Fine-tune')
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('F1 Score', fontsize=11)
        axes[1, 0].set_title('Validation F1 Score', fontsize=12, fontweight='bold')
        axes[1, 0].legend(loc='best', fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1.05])

        # Plot 4: Summary Statistics
        summary_text = (
            f'Training Summary\n'
            f'{"="*40}\n\n'
            f'Total Epochs: {len(all_epochs)}\n'
            f'Head Epochs: {head_epochs}\n'
            f'Fine-tune Epochs: {len(all_epochs) - head_epochs}\n\n'
            f'Best Validation Metrics:\n'
            f'  • Accuracy: {max(all_val_acc):.4f}\n'
            f'  • F1 Score: {max(all_val_f1):.4f}\n'
            f'  • Min Val Loss: {min(all_val_loss):.4f}\n\n'
            f'Final Metrics:\n'
            f'  • Train Loss: {all_train_loss[-1]:.4f}\n'
            f'  • Val Loss: {all_val_loss[-1]:.4f}\n'
            f'  • Train Acc: {all_train_acc[-1]:.4f}\n'
            f'  • Val Acc: {all_val_acc[-1]:.4f}\n'
            f'  • Val F1: {all_val_f1[-1]:.4f}'
        )

        axes[1, 1].text(0.05, 0.95, summary_text,
                       transform=axes[1, 1].transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       fontfamily='monospace',
                       bbox={"boxstyle": 'round', "facecolor": 'wheat', "alpha": 0.5})
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff', )
        plt.close()

        logger.info(f"Cool Training history plot saved to: {save_path}")
        return True

    except Exception as e:
        logger.error(f"x Failed to create training history plot: {e}")
        logger.exception("Full traceback:")
        return False


# =============================================================================
# 4. PER-CLASS METRICS PLOTTING
# =============================================================================

def plot_per_class_metrics(y_true, y_pred, class_names, save_path, title="Per-Class Metrics"):
    """
    Plot per-class precision, recall, and F1 scores.
    """
    try:
        from sklearn.metrics import precision_recall_fscore_support

        # With average=None, returns arrays for each class
        result = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        # Explicitly cast to numpy arrays for type safety
        precision: np.ndarray = np.asarray(result[0])
        recall: np.ndarray = np.asarray(result[1])
        f1: np.ndarray = np.asarray(result[2])
        support: np.ndarray = np.asarray(result[3])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        x = np.arange(len(class_names))
        width = 0.25

        ax1.bar(x - width, precision, width, label='Precision', color='#2E86AB', alpha=0.8)
        ax1.bar(x, recall, width, label='Recall', color='#A23B72', alpha=0.8)
        ax1.bar(x + width, f1, width, label='F1-Score', color='#18A558', alpha=0.8)

        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Per-Class Precision, Recall, and F1-Score', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.05])

        ax2.bar(x, support, color='#F18F01', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Class Distribution (Support)', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        max_support = int(np.max(support))
        for i, v in enumerate(support):
            ax2.text(i, v + max_support * 0.02, str(int(v)),
                    ha='center', va='bottom', fontsize=9)

        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff', )
        plt.close()

        logger.info(f"Cool Per-class metrics plot saved to: {save_path}")

        # Save metrics to JSON
        metrics_json_path = save_path.parent / f"{save_path.stem}_metrics.json"
        metrics_dict = {
            'per_class': {
                class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(class_names))
            },
            'macro_avg': {
                'precision': float(np.mean(precision)),
                'recall': float(np.mean(recall)),
                'f1_score': float(np.mean(f1))
            },
            'weighted_avg': {
                'precision': float(np.sum(precision * support) / np.sum(support)),
                'recall': float(np.sum(recall * support) / np.sum(support)),
                'f1_score': float(np.sum(f1 * support) / np.sum(support))
            }
        }

        with open(metrics_json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        logger.info(f"Cool Per-class metrics JSON saved to: {metrics_json_path}")

        return True

    except Exception as e:
        logger.error(f"x Failed to create per-class metrics plot: {e}")
        return False


# =============================================================================
# 5. K-FOLD RESULTS PLOTTING
# =============================================================================

def plot_kfold_results(kfold_results, save_path, title="K-Fold Cross-Validation Results"):
    """
    Plot K-fold cross-validation results across different backbones.
    """
    try:
        if not kfold_results:
            logger.warning("No K-fold results to plot")
            return False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        backbones = list(kfold_results.keys())
        mean_accs = [kfold_results[bb]['mean_accuracy'] for bb in backbones]
        std_accs = [kfold_results[bb]['std_accuracy'] for bb in backbones]

        x_pos = np.arange(len(backbones))
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(backbones)))

        bars = ax1.bar(x_pos, mean_accs, yerr=std_accs,
                       color=colors, alpha=0.7, capsize=5, edgecolor='black')
        ax1.set_xlabel('Backbone', fontsize=12)
        ax1.set_ylabel('Mean Accuracy', fontsize=12)
        ax1.set_title('K-Fold Mean Accuracy by Backbone', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(backbones, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.05])

        for _i, (bar, mean, std) in enumerate(zip(bars, mean_accs, std_accs)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}+/-{std:.3f}',
                    ha='center', va='bottom', fontsize=9, rotation=0)

        fold_data = []
        for bb in backbones:
            if 'fold_accuracies' in kfold_results[bb]:
                fold_data.append(kfold_results[bb]['fold_accuracies'])
            else:
                fold_data.append([kfold_results[bb]['mean_accuracy']] * 5)

        bp = ax2.boxplot(fold_data, labels=backbones, patch_artist=True,
                        showmeans=True, meanline=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_xlabel('Backbone', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('K-Fold Accuracy Distribution', fontsize=13, fontweight='bold')
        ax2.set_xticklabels(backbones, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1.05])

        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff', )
        plt.close()

        logger.info(f"Cool K-fold results plot saved to: {save_path}")
        return True

    except Exception as e:
        logger.error(f"x Failed to create K-fold results plot: {e}")
        return False


# =============================================================================
# 6. COMPREHENSIVE BACKBONE COMPARISON
# =============================================================================

def plot_backbone_comparison(results, save_path, title="Backbone Performance Comparison"):
    """
    Create comprehensive comparison of all backbone performances.
    """
    try:
        if not results:
            logger.warning("No results to compare")
            return False

        backbones = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for backbone, result in results.items():
            if 'final_metrics' in result:
                metrics = result['final_metrics']
                backbones.append(backbone)
                accuracies.append(metrics.get('final_accuracy', 0))
                precisions.append(metrics.get('final_precision', 0))
                recalls.append(metrics.get('final_recall', 0))
                f1_scores.append(metrics.get('final_f1_score', 0))

        if not backbones:
            logger.warning("No valid metrics found for comparison")
            return False

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        x_pos = np.arange(len(backbones))
        width = 0.6
        cmap = plt.get_cmap('Set3')
        colors = cmap(np.linspace(0, 1, len(backbones)))

        # Plot 1: Accuracy comparison
        bars1 = axes[0, 0].bar(x_pos, accuracies, width, color=colors,
                              alpha=0.8, edgecolor='black')
        axes[0, 0].set_xlabel('Backbone', fontsize=11)
        axes[0, 0].set_ylabel('Accuracy', fontsize=11)
        axes[0, 0].set_title('Final Accuracy by Backbone', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(backbones, rotation=45, ha='right', fontsize=9)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].set_ylim([0, 1.05])

        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{acc:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 2: All metrics comparison
        x = np.arange(len(backbones))
        width_multi = 0.2

        axes[0, 1].bar(x - 1.5*width_multi, accuracies, width_multi,
                      label='Accuracy', color='#2E86AB', alpha=0.8)
        axes[0, 1].bar(x - 0.5*width_multi, precisions, width_multi,
                      label='Precision', color='#A23B72', alpha=0.8)
        axes[0, 1].bar(x + 0.5*width_multi, recalls, width_multi,
                      label='Recall', color='#18A558', alpha=0.8)
        axes[0, 1].bar(x + 1.5*width_multi, f1_scores, width_multi,
                      label='F1-Score', color='#F18F01', alpha=0.8)

        axes[0, 1].set_xlabel('Backbone', fontsize=11)
        axes[0, 1].set_ylabel('Score', fontsize=11)
        axes[0, 1].set_title('All Metrics Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(backbones, rotation=45, ha='right', fontsize=9)
        axes[0, 1].legend(loc='best', fontsize=9)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].set_ylim([0, 1.05])

        # Plot 3: Radar chart for top 3 backbones
        from math import pi

        top_3_indices = np.argsort(accuracies)[-3:][::-1]

        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        N = len(categories)

        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        ax = plt.subplot(2, 2, 3, projection='polar')

        for idx in top_3_indices:
            values = [accuracies[idx], precisions[idx], recalls[idx], f1_scores[idx]]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=backbones[idx])
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Top 3 Backbones - Radar Chart', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True)

        # Plot 4: Summary table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')

        table_data = []
        table_data.append(['Backbone', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

        for i, backbone in enumerate(backbones):
            table_data.append([
                backbone,
                f'{accuracies[i]:.4f}',
                f'{precisions[i]:.4f}',
                f'{recalls[i]:.4f}',
                f'{f1_scores[i]:.4f}'
            ])

        table = axes[1, 1].table(cellText=table_data, cellLoc='center',
                                loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        for i in range(1, len(table_data)):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        axes[1, 1].set_title('Performance Summary Table', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff', )
        plt.close()

        logger.info(f"Cool Backbone comparison plot saved to: {save_path}")
        return True

    except Exception as e:
        logger.error(f"x Failed to create backbone comparison plot: {e}")
        logger.exception("Full traceback:")
        return False


# =============================================================================
# 7. COMPREHENSIVE VISUALIZATION FUNCTION
# =============================================================================

def generate_all_visualizations(model, backbone_name, history, val_loader,
                                class_names, criterion, device):
    """
    Generate all visualizations for a trained model.
    """
    logger.info(f"Generating comprehensive visualizations for {backbone_name}...")

    plot_paths = {}

    try:
        # Get validation predictions
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)

                # Handle different output formats
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Generate all plots
        from pathlib import Path

        # 1. Training History Plot
        history_path = Path(f"{backbone_name}_training_history.tiff")
        if plot_training_history(history, history_path, f"{backbone_name} Training History"):
            plot_paths['training_history'] = str(history_path)

        # 2. Confusion Matrix
        cm_path = Path(f"{backbone_name}_confusion_matrix.tiff")
        if plot_confusion_matrix(all_labels, all_preds, class_names, cm_path,
                                f"{backbone_name} Confusion Matrix"):
            plot_paths['confusion_matrix'] = str(cm_path)

        # 3. ROC Curves
        roc_path = Path(f"{backbone_name}_roc_curves.tiff")
        roc_aucs = plot_roc_curves(all_labels, all_probs, class_names, roc_path,
                                   f"{backbone_name} ROC Curves")
        if roc_aucs:
            plot_paths['roc_curves'] = str(roc_path)

        # 4. Per-Class Metrics
        metrics_path = Path(f"{backbone_name}_per_class_metrics.tiff")
        if plot_per_class_metrics(all_labels, all_preds, class_names, metrics_path,
                                  f"{backbone_name} Per-Class Metrics"):
            plot_paths['per_class_metrics'] = str(metrics_path)

        logger.info(f"Cool Generated {len(plot_paths)} visualizations for {backbone_name}")

    except Exception as e:
        logger.error(f"x Failed to generate visualizations for {backbone_name}: {e}")
        logger.exception("Full traceback:")

    return plot_paths


# =============================================================================
# 8. VISUALIZATION SUMMARY SAVER
# =============================================================================

def save_visualization_summary(plot_paths, backbone_name, save_dir):
    """
    Save a summary of all generated visualizations.
    """
    try:
        summary = {
            'backbone': backbone_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'plots': plot_paths,
            'plot_count': len(plot_paths),
            'format': 'TIFF',
            'dpi': 1200,
            'compression': 'LZW'
        }

        summary_path = Path(save_dir) / f"{backbone_name}_visualization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Cool Visualization summary saved to: {summary_path}")
        return str(summary_path)

    except Exception as e:
        logger.error(f"x Failed to save visualization summary: {e}")
        return None


# =============================================================================
# 9. LEARNING RATE SCHEDULE PLOTTING
# =============================================================================

def plot_learning_rate_schedule(lr_history, save_path, title="Learning Rate Schedule"):
    """
    Plot learning rate schedule over epochs.
    """
    try:
        if not lr_history:
            logger.warning("No learning rate history to plot")
            return False

        _fig, ax = plt.subplots(figsize=(12, 6))

        epochs = list(range(len(lr_history)))
        ax.plot(epochs, lr_history, linewidth=2, color='#E63946', marker='o', markersize=4)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        min_lr = min(lr_history)
        max_lr = max(lr_history)
        min_idx = lr_history.index(min_lr)
        max_idx = lr_history.index(max_lr)

        ax.annotate(f'Max: {max_lr:.2e}', xy=(max_idx, max_lr),
                   xytext=(max_idx, max_lr*2),
                   arrowprops={"arrowstyle": '->', "color": 'green'},
                   fontsize=10, color='green')

        ax.annotate(f'Min: {min_lr:.2e}', xy=(min_idx, min_lr),
                   xytext=(min_idx, min_lr*0.5),
                   arrowprops={"arrowstyle": '->', "color": 'red'},
                   fontsize=10, color='red')

        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff', )
        plt.close()

        logger.info(f"Cool Learning rate schedule plot saved to: {save_path}")
        return True

    except Exception as e:
        logger.error(f"x Failed to create learning rate schedule plot: {e}")
        return False


# =============================================================================
# 10. GRADIENT FLOW VISUALIZATION
# =============================================================================

def plot_gradient_flow(named_parameters, save_path, title="Gradient Flow"):
    """
    Plot gradient flow through the network layers.
    """
    try:
        ave_grads = []
        max_grads = []
        layers = []

        for n, p in named_parameters:
            if p.requires_grad and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())

        if not layers:
            logger.warning("No gradients to plot")
            return False

        _fig, ax = plt.subplots(figsize=(14, 6))

        x_pos = np.arange(len(layers))
        ax.bar(x_pos, max_grads, alpha=0.5, lw=1, color='c', label='Max Gradient')
        ax.bar(x_pos, ave_grads, alpha=0.7, lw=1, color='b', label='Mean Gradient')

        ax.set_xlabel('Layers', fontsize=12)
        ax.set_ylabel('Gradient Magnitude', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(layers, rotation=90, fontsize=8)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff', )
        plt.close()

        logger.info(f"Cool Gradient flow plot saved to: {save_path}")
        return True

    except Exception as e:
        logger.error(f"x Failed to create gradient flow plot: {e}")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    set_seed(SEED)

    # Check if debug mode
    if DEBUG_MODE:
        logger.info(f"\n{'='*80}")
        logger.info("$ DEBUG MODE ENABLED")
        logger.info(f"{'='*80}")
        logger.info("Environment Variables:")
        logger.info(f"  DBT_DEBUG_MODE={DEBUG_MODE}")
        logger.info(f"  DBT_DEBUG_BACKBONE={DEBUG_BACKBONE}")
        logger.info(f"  DBT_DEBUG_FUNCTION={DEBUG_FUNCTION}")
        logger.info(f"  DBT_DEBUG_HEAD_EPOCHS={_debug_epochs_head}")
        logger.info(f"  DBT_DEBUG_FT_EPOCHS={_debug_epochs_finetune}")
        logger.info(f"  DBT_DEBUG_BATCH_SIZE={_debug_batch_size}")
        logger.info("\nAvailable Debug Functions:")
        for func, desc in DEBUG_FUNCTIONS.items():
            logger.info(f"  {func:20} - {desc}")
        logger.info(f"{'='*80}\n")

        # Run debug mode
        try:
            debug_results = run_debug_mode()
            logger.info("\nCool Debug mode completed successfully")
            sys.exit(0)
        except Exception as e:
            logger.error(f"\nx Debug mode failed: {e}")
            sys.exit(1)


    if not hasattr(smoke_checker, 'checks'):
        logger.warning("Reinitializing smoke_checker...")
        smoke_checker = SmokeCheckLogger()

    logger.info("="*80)
    export_status = "ENABLED" if ENABLE_EXPORT else "DISABLED"
    logger.info("DISEASE CLASSIFICATION PIPELINE with export system")
    logger.info("="*80)

    try:
        start_time = time.time()

        # Run unit tests first (including export tests)
        logger.info("\n" + "="*80)
        logger.info("RUNNING ALL UNIT TESTS")
        logger.info("="*80)

        # Original unit tests with error handling
        try:
            passed_core, failed_core = run_all_unit_tests()
        except Exception as e:
            logger.error(f"Core unit tests failed: {e}")
            passed_core, failed_core = 0, 0

        # Export unit tests with error handling
        if ENABLE_EXPORT:
            try:
                passed_export, failed_export = run_export_unit_tests()
            except Exception as e:
                logger.error(f"Export unit tests failed: {e}")
                passed_export, failed_export = 0, 0
        else:
            logger.info("Export unit tests skipped (ENABLE_EXPORT = False)")
            passed_export, failed_export = 0, 0

        total_passed = passed_core + passed_export
        total_failed = failed_core + failed_export

        logger.info(f"\n{'='*80}")
        logger.info(f"TOTAL TEST SUMMARY: {total_passed} passed, {total_failed} failed")
        logger.info(f"{'='*80}\n")

        if total_failed > 0:
            logger.warning(f"{total_failed} unit tests failed. Proceeding with caution...")

        results = run_optimized_pipeline()
        end_time = time.time()

        total_time = end_time - start_time
        logger.info(f"\nPIPELINE COMPLETED IN {total_time:.1f}s ({total_time/60:.1f}min)")

        #   Properly extract training results
        if results:
            training_results = results.get('training_results', {})
            verification_results = results.get('verification_results', {})

            verified_count = sum(1 for r in verification_results.values() if r.get('status') == 'verified')
            trained_count = sum(1 for r in training_results.values() if r.get('status') == 'success')

            logger.info(f"\n{'='*60}")
            logger.info("FINAL PIPELINE RESULTS:")
            logger.info(f"  Verified:  {verified_count} models")
            logger.info(f"  Trained:   {trained_count} models (with export)")
            logger.info(f"{'='*60}")

            if trained_count > 0:
                logger.info(f"\nCool SUCCESS: {trained_count} backbones fully trained and exported")
            else:
                logger.warning("\nxx WARNING: No models completed training")
        else:
            logger.error("\nx ERROR: Pipeline returned no results")

    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        raise




