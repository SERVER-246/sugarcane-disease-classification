# Copilot Instructions for Disease Classification Backbone Framework

## Project Overview

This is a **PyTorch-based disease classification framework** with 15 custom-built neural network architectures. It trains models end-to-end on medical image datasets with transfer learning, K-fold cross-validation, and multi-format model export (PyTorch, ONNX, TorchScript, TensorRT, CoreML, etc.).

**Key Scope:**
- `Base_backbones.py`: 7,300+ line monolithic training script with all backbone definitions, training pipeline, and export system
- Single-file codebase (no modular structure)
- Windows-first design (multiprocessing, path handling)

## Architecture & Design Patterns

### Three-Stage Training Pipeline

All backbones follow this workflow in `train_backbone_with_metrics()`:

1. **Head Training** (frozen backbone, train classifier head only) - `EPOCHS_HEAD=40`
2. **Fine-tuning** (unfreeze backbone, full model training) - `EPOCHS_FINETUNE=25`
3. **Export** (convert to ONNX, TorchScript, TensorRT, CoreML, etc.)

**Learning Rates**: `BACKBONE_LR=1e-6` (finetune), `HEAD_LR=1e-3` (head training)

### Custom Backbone Architectures

All 15 backbones are built from scratch using modular blocks. Key patterns:

- **Core building block**: `CoreImageBlock` - wraps Conv2d + BN + activation + optional LayerNorm
- **Activation functions**: GELU, SiLU, Mish (via `get_activation_fn()`)
- **Custom layers**:
  - `MobileOneBlock` - reparameterizable blocks (multi-branch → single conv)
  - `DynamicConv` - multiple kernel sizes fused at runtime
  - `SwinTransformerBlock` - window-based attention + cyclic shifts
  - `WindowAttention` - multi-head self-attention with spatial partitioning
  - `PatchMerging` - 2x2 patch downsampling for transformer layers

**Backbone List** (in `BACKBONES` constant):
```python
CustomConvNeXt, CustomEfficientNetV4, CustomGhostNetV2, CustomResNetMish,
CustomCSPDarkNet, CustomInceptionV4, CustomViTHybrid, CustomSwinTransformer,
CustomCoAtNet, CustomRegNet, CustomDenseNetHybrid, CustomDeiTStyle,
CustomMaxViT, CustomMobileOne, CustomDynamicConvNet
```

## Critical Workflows

### Local Development & Debugging

**Debug Mode** - Set environment variables before running:
```bash
DBT_DEBUG_MODE=true
DBT_DEBUG_BACKBONE=CustomCoAtNet  # which backbone to test
DBT_DEBUG_FUNCTION=full_training  # see DEBUG_FUNCTIONS dict
DBT_DEBUG_HEAD_EPOCHS=15          # faster training
DBT_DEBUG_FT_EPOCHS=10
DBT_DEBUG_BATCH_SIZE=8
```

**Available Debug Functions** (from `DEBUG_FUNCTIONS` dict):
- `model_creation` - Just instantiate the model
- `forward_pass` - Test forward pass with dummy input
- `backward_pass` - Verify gradients flow
- `single_epoch` - One training epoch only
- `overfit_batch` - Overfit on single batch (sanity check)
- `dataset_loading` - Verify data pipeline
- `full_training` - Complete training with debug epochs
- `export_only` - Test export formats
- `smoke_tests` - Unit test suite
- `architecture_verify` - Validate model structure
- `pretrained_loading` - Check pretrained weight loading
- `all_checks` - Run all debug checks

### Dataset & Paths

**Directory Structure** (via env vars):
```
DBT_BASE_DIR/
├── Data/               (RAW_DIR) - original images organized by class
├── split_dataset/      (SPLIT_DIR)
│   ├── train/
│   ├── val/
│   └── test/
├── checkpoints/        (CKPT_DIR) - saved .pth files
├── plots_metrics/      (PLOTS_DIR) - training curves
├── kfold_results/      (KFOLD_DIR) - cross-validation results
├── deployment_models/  (DEPLOY_DIR) - exported formats
└── debug_logs/         - debug output
```

**Dataset Loading**:
- `prepare_optimized_datasets()` - Splits raw images into train/val/test (80/10/10)
- `WindowsCompatibleImageFolder` - Custom loader for Windows path handling
- K-fold CV via `k_fold_cross_validation()` with `StratifiedKFold(n_splits=K_FOLDS=5)`

### Model Creation & Training

**Key Functions**:
- `create_custom_backbone(name, num_classes)` - Factory function, maps backbone names to classes
- `train_backbone_with_metrics()` - Main training orchestrator (head→finetune stages)
- `train_epoch_optimized()` - Single epoch with mixed precision (AMP)
- `validate_epoch_optimized()` - Validation with metric collection

**Automatic Device Selection**:
```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**Loss Functions** - Per-backbone selection in `get_loss_function_for_backbone()`:
- Vision transformers (ViT, Swin, etc.) → CrossEntropyLoss
- Other convnets → CrossEntropyLoss (default)

**Optimizers** - Per-backbone in `create_optimized_optimizer()`:
- Most backbones → AdamW
- Special handling for specific architectures

### Model Export System

**Export Pipeline** in `export_and_package_model()`:
1. **State Dict** - Raw PyTorch weights (`.pth`)
2. **TorchScript** - Serialized computation graph (`.pt`)
3. **ONNX** - Portable format (`.onnx`)
4. **TensorRT** - NVIDIA GPU optimization (`.engine`)
5. **OpenVINO** - Intel hardware (`.xml`)
6. **CoreML** - Apple devices (`.mlmodel`)
7. **TFLite** - Mobile/embedded (`.tflite`)

Each format includes:
- Smoke tests (`smoke_check_*()` functions) - verify shape/dtype correctness
- Artifact packaging (metadata, config, normalization stats)

## Configuration & Environment Variables

**Essential Env Vars**:
```bash
DBT_BASE_DIR          # Project root (default: F:\\DBT-Base-DIr)
DBT_RAW_DIR           # Raw images directory
DBT_SPLIT_DIR         # Train/val/test split location
DBT_DEBUG_MODE        # Enable debug mode
DBT_DEBUG_BACKBONE    # Which backbone to debug
DBT_DEBUG_FUNCTION    # Debug function name
```

**Hyperparameters** (in CONFIGURATION section):
```python
IMG_SIZE = 224            # Image input dimension
BATCH_SIZE = 32           # Training batch size
EPOCHS_HEAD = 40          # Head training epochs
EPOCHS_FINETUNE = 25      # Fine-tuning epochs
PATIENCE_HEAD = 5         # Early stopping patience (head)
PATIENCE_FT = 5           # Early stopping patience (finetune)
WEIGHT_DECAY = 1e-4       # L2 regularization
K_FOLDS = 5               # Cross-validation folds
ENABLE_KFOLD_CV = True    # Enable K-fold validation
ENABLE_EXPORT = True      # Enable multi-format export
ENABLE_TRANSFER_LEARNING = True
TRANSFER_LEARNING_MODE = 'fine_tuning'  # vs 'feature_extraction'
```

## Code Patterns & Conventions

### Model Definition Template

All backbones follow this structure:
```python
class CustomArchName(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # 1. Stem: Conv layers to extract initial features
        self.stem = nn.Sequential(...)
        
        # 2. Main stages: Progressive downsampling + feature extraction
        self.stages = nn.ModuleList([...])
        
        # 3. Head: Classification layers
        self.classifier = nn.Linear(...)
    
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)
```

### Error Handling Pattern

Use this try-catch wrapper for robustness:
```python
try:
    # operation
except ValueError as e:
    logger.error(f"Validation: {e}")
    smoke_checker.log_check("component", "FAIL", str(e))
    raise
except RuntimeError as e:
    logger.error(f"Runtime: {e}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    raise
```

### GPU Memory Management

- Always call `torch.cuda.empty_cache()` after OOM errors
- Use AMP (mixed precision) in training: `from torch import amp`
- Checkpoint saving uses CPU transfer: `export_model_state_dict_cpu()`

### Logging Pattern

Use `logger` (not `print`):
```python
logger.info(f"Training {backbone_name}...")  # Info
logger.warning(f"x Potential issue...")       # Warning with 'x' prefix
logger.error(f"ERROR: {e}")                   # Error
logger.exception("Full traceback:")           # Full exception
```

## Performance & Optimization Techniques

- **Mixed Precision Training**: AMP enabled in `train_epoch_optimized()`
- **Data Loading**: `num_workers` auto-detected via `get_optimal_workers()`
- **Windows Compat**: Custom multiprocessing spawn method (required on Windows)
- **Early Stopping**: Per-stage patience counters (`PATIENCE_HEAD`, `PATIENCE_FT`)
- **Learning Rate Scheduling**: `create_improved_scheduler()` uses CosineAnnealingLR + warmup

## Testing & Validation

**Unit Test Suite** (`run_all_unit_tests()`):
- Model creation tests
- Forward/backward pass validation
- Export format tests
- Idempotency checks

**Smoke Checks** (`SmokeCheckLogger`):
- Verify exported models produce correct output shapes
- Check numerical stability (no NaN/Inf)
- Validate metadata in exported artifacts

**K-Fold Cross-Validation**:
- Automatic stratified splitting
- Per-fold training and aggregated metrics
- Results saved to `KFOLD_DIR`

## Important Gotchas & Fixes

1. **Transformer Positional Encoding** - Must use learnable `nn.Parameter()` for absolute position embeddings, not fixed positions
2. **MobileOne Initialization** - Scale down initial weights (×0.5) to prevent gradient explosion from multi-branch fusion
3. **Patch Embedding Resolution** - Track actual spatial resolution through layers; `CustomSwinTransformer` needs `actual_resolution = img_size // 4` for patch embedding
4. **Windows DataLoader** - Use `WindowsCompatibleImageFolder` instead of standard `ImageFolder` (path separator issues)
5. **Multiprocessing on Windows** - Call `multiprocessing.set_start_method('spawn', force=True)` before other imports (done at top of file)
6. **Export Failure Recovery** - If ONNX export fails, fallback to TorchScript (handled in orchestrator)

## When Modifying Code

- **Adding a new backbone**: Register in `BACKBONES` list, implement class, add to `backbone_map` in `create_custom_backbone()`
- **Changing training hyperparams**: Update `EPOCHS_HEAD`, `EPOCHS_FINETUNE`, `BATCH_SIZE` at top, or use debug env vars for testing
- **New export format**: Add smoke check function, register in `export_formats` dict, implement export wrapper
- **Dataset changes**: Modify `prepare_optimized_datasets()` split logic or `RAW_DIR` path
- **Debugging failures**: Set `DBT_DEBUG_MODE=true` + specific `DBT_DEBUG_FUNCTION` before running

## Run Commands

**Full Pipeline** (all 15 backbones):
```bash
python Base_backbones.py
```

**Debug Mode** (single backbone, fast):
```bash
set DBT_DEBUG_MODE=true
set DBT_DEBUG_BACKBONE=CustomCoAtNet
set DBT_DEBUG_FUNCTION=full_training
python Base_backbones.py
```

**Quick Sanity Check**:
```bash
set DBT_DEBUG_MODE=true
set DBT_DEBUG_FUNCTION=model_creation
python Base_backbones.py
```
