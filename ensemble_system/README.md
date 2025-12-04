# Ensemble Orchestration System

Complete ensemble system that mirrors the BASE-BACK training pipeline structure while orchestrating ensemble operations across all trained backbones.

## Overview

This ensemble system follows the **exact same sequence and structure** as the base training pipeline:
1. Smoke/Verification Stage
2. K-fold Cross-Validation (OOF) Stage  
3. Stage 1: Head Training
4. Stage 2: Fine-tuning
5. Stage 3: Test Evaluation
6. Export & Packaging Stage
7. Cleanup & Resource Management

But instead of training single backbones, it orchestrates ensemble operations (voting, stacking, fusion, model soups, MoE, distillation) across all available trained models.

## Directory Structure

```
ensemble_system/
├── ensemble_orchestrator.py    # Main CLI orchestrator
├── configs/
│   └── ensemble_config.yaml    # Configuration file
├── utils/
│   ├── ensemble_methods.py     # Score-level ensembles, model soups
│   ├── stacking_methods.py     # OOF stacking, meta-learners
│   ├── fusion_methods.py       # Feature-level fusion
│   ├── distillation_methods.py # Knowledge distillation
│   ├── moe_methods.py          # Mixture of Experts
│   └── debug_utils.py          # Debug/test utilities
└── tests/
    └── debug_tests.py          # Automated unit tests
```

## Installation & Setup

The ensemble system reuses all BASE-BACK utilities, so no additional dependencies are required beyond what's needed for base training.

```bash
# Navigate to ensemble directory
cd ensemble_system

# The system automatically imports from BASE-BACK/src
```

## Usage

### 1. Discover Available Models

Scan all trained backbones and build catalog:

```bash
python ensemble_orchestrator.py discover
```

This creates `ensembles/ensemble_catalog.json` with all available:
- Exported models (DEPLOY_DIR)
- Checkpoints (CKPT_DIR)
- K-fold OOF predictions (KFOLD_DIR)
- Training metrics (METRICS_DIR)

### 2. Build Score-Level Ensembles

```bash
python ensemble_orchestrator.py build-ensemble --config configs/ensemble_config.yaml
```

Creates:
- Soft voting ensemble
- Hard voting ensemble
- Logit averaging ensemble
- Weighted ensemble (optimized on validation set)
- Model soups (uniform & greedy)
- Snapshot ensembles

### 3. Train Stacking Ensemble

```bash
python ensemble_orchestrator.py stack --config configs/ensemble_config.yaml
```

Uses K-fold OOF predictions from base training to train meta-learners:
- Logistic Regression
- XGBoost
- Random Forest
- MLP (neural network)

Avoids data leakage by using only OOF predictions.

### 4. Train Feature Fusion

```bash
python ensemble_orchestrator.py fusion-train --config configs/ensemble_config.yaml
```

Two-stage fusion:
1. Extract embeddings from all backbones
2. Train fusion head (Concat+MLP, Attention, Bilinear, CCA)

Optionally supports end-to-end fine-tuning.

### 5. Train Meta-Fuser (Ensemble-of-Ensembles)

```bash
python ensemble_orchestrator.py meta-fuse --config configs/ensemble_config.yaml
```

Combines outputs from all ensemble families into a final meta-model.

### 6. Distill Ensemble into Student

```bash
python ensemble_orchestrator.py distill --config configs/ensemble_config.yaml
```

Knowledge distillation:
- Teacher: any ensemble or meta-fuser
- Student: smaller/faster model
- Loss: KL divergence + cross-entropy
- Temperature scaling supported

### 7. Evaluate Ensemble

```bash
python ensemble_orchestrator.py evaluate --config configs/ensemble_config.yaml
```

Comprehensive evaluation:
- Accuracy, F1, Precision, Recall
- Log-loss, ECE calibration
- Confusion matrix
- Per-class metrics
- Inference latency

### 8. Export Ensemble Model

```bash
python ensemble_orchestrator.py export --model-name soft_voting_ensemble
```

Uses the same `export_and_package_model()` function from base training to export to:
- State dict
- Checkpoint
- TorchScript
- ONNX
- TFLite
- TensorRT
- Custom formats

Creates identical directory structure as base training:
```
DEPLOY_DIR/soft_voting_ensemble/
├── state_dict/
├── checkpoint/
├── torchscript/
├── onnx/
├── tflite/
├── custom/
├── metadata.json
├── class_mapping.json
├── export_status.json
└── checksums.json
```

## Debug Mode

Test individual stages with minimal runtime:

```bash
# Test discovery
python ensemble_orchestrator.py debug --section discover --verbose

# Test smoke checks for specific backbone
python ensemble_orchestrator.py debug --section smoke --arch CustomViTHybrid --fast

# Test K-fold OOF loading
python ensemble_orchestrator.py debug --section kfold --fast

# Test embedding extraction
python ensemble_orchestrator.py debug --section extract_embeddings --arch CustomConvNeXt --fast

# Test ensemble building (dry-run)
python ensemble_orchestrator.py debug --section build_ensemble --dry-run

# Test stacking
python ensemble_orchestrator.py debug --section stack --fast

# Test fusion training
python ensemble_orchestrator.py debug --section fusion_train --fast

# Test distillation
python ensemble_orchestrator.py debug --section distill --fast

# Test BN recomputation
python ensemble_orchestrator.py debug --section recompute_bn --arch CustomViTHybrid

# Test export (dry-run)
python ensemble_orchestrator.py debug --section export --dry-run

# Run all debug checks
python ensemble_orchestrator.py debug --section all --fast
```

Debug features:
- `--dry-run`: Walk code paths without executing (validates logic & file paths)
- `--fast`: Use tiny synthetic dataset (16 samples, 1 epoch) for quick validation
- `--verbose`: Step-by-step logging with shapes, stats, file paths
- `--confirm`: Required for destructive debug actions

Debug outputs saved to `debug_tmp/` sandbox (never overwrites production models).

## Configuration File

`configs/ensemble_config.yaml`:

```yaml
project:
  name: "disease_classification_ensemble"
  version: "1.0"
  
device: "cuda"  # or "cpu"
seed: 42

dataset:
  img_size: 224
  batch_size: 32
  num_workers: 16
  
backbones:
  # All 15 custom backbones
  include:
    - CustomConvNeXt
    - CustomEfficientNetV4
    - CustomGhostNetV2
    - CustomResNetMish
    - CustomCSPDarkNet
    - CustomInceptionV4
    - CustomViTHybrid
    - CustomSwinTransformer
    - CustomCoAtNet
    - CustomRegNet
    - CustomDenseNetHybrid
    - CustomDeiTStyle
    - CustomMaxViT
    - CustomMobileOne
    - CustomDynamicConvNet
  
  # Optional: exclude specific backbones
  exclude: []

exports:
  enable: true
  formats:
    - state_dict
    - checkpoint
    - torchscript
    - onnx
    - tflite

runs:
  # Score-level ensembles
  score_ensemble:
    enable: true
    methods:
      - soft_voting
      - hard_voting
      - logit_averaging
      - weighted  # optimize weights on validation set
    optimize_weights: true
    optimization_metric: "f1_macro"  # or "accuracy", "log_loss"
  
  # Model soups
  model_soup:
    enable: true
    methods:
      - uniform  # simple weight averaging
      - greedy   # incremental selection
    recompute_bn: true  # REQUIRED for weight-space ensembles
  
  # Snapshot ensembles
  snapshot_ensemble:
    enable: false  # requires retraining
    n_snapshots: 5
    epochs_per_cycle: 10
  
  # Stacking
  stacking:
    enable: true
    meta_learners:
      - lr        # Logistic Regression
      - xgb       # XGBoost
      - mlp       # Neural network
    mlp_config:
      hidden_dims: [256, 128]
      epochs: 50
      lr: 0.001
      dropout: 0.3
    xgb_config:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
  
  # Feature fusion
  fusion:
    enable: true
    extract_embeddings: true
    fusion_methods:
      - concat_mlp
      - attention
      - bilinear
      - cca_aligned
    fusion_config:
      hidden_dims: [512, 256]
      epochs: 40
      lr: 0.001
      dropout: 0.3
    end_to_end: false  # if true, fine-tune backbones
  
  # Mixture of Experts
  moe:
    enable: false
    num_experts: 5  # subset of backbones
    top_k: 2        # sparse routing
    gating_hidden_dim: 128
    epochs: 30
  
  # Meta-fuser (ensemble-of-ensembles)
  meta_fuser:
    enable: true
    input_ensembles:
      - soft_voting
      - logit_averaging
      - stacking_xgb
      - fusion_concat_mlp
    fuser_type: "xgb"  # or "mlp", "attention"
    mlp_config:
      hidden_dims: [128, 64]
      epochs: 30
      lr: 0.001

pruning:
  enable: true
  methods:
    - greedy_forward       # start empty, add best
    - greedy_backward      # start full, remove worst
    - diversity_max        # maximize pairwise disagreement
    - clustering           # cluster + select representatives
  target_size: 7           # target ensemble size
  cost_budget:
    max_latency_ms: 100    # optional latency constraint
    max_size_mb: 500       # optional size constraint

distillation:
  enable: true
  teacher: "meta_fuser"    # or any ensemble name
  student: "CustomMobileOne"  # lightweight model
  temperature: 4.0
  alpha: 0.5               # weight for KL loss
  epochs: 50
  lr: 0.001
  batch_size: 32
```

## File & Export Structure

All ensemble outputs follow the **exact same structure** as base training:

### Metrics Directory
```
METRICS_DIR/ensembles/
├── soft_voting_ensemble/
│   └── metrics.json
├── stacking_xgb/
│   └── metrics.json
├── fusion_concat_mlp/
│   └── metrics.json
└── distilled_mobile/
    └── metrics.json
```

### Deployment Directory
```
DEPLOY_DIR/
├── soft_voting_ensemble/
│   ├── state_dict/
│   ├── checkpoint/
│   ├── torchscript/
│   ├── onnx/
│   ├── tflite/
│   ├── metadata.json
│   ├── class_mapping.json
│   ├── export_status.json
│   └── checksums.json
├── stacking_xgb/
│   └── ... (same structure)
└── fusion_concat_mlp/
    └── ... (same structure)
```

### Ensemble-Specific Directory
```
ENSEMBLE_DIR/
├── ensemble_catalog.json           # discovery results
├── fusion_embeddings/              # extracted embeddings
│   ├── CustomViTHybrid/
│   │   ├── train.npy
│   │   ├── val.npy
│   │   └── test.npy
│   └── CustomConvNeXt/
│       └── ...
└── oof_predictions/                # cached OOF from K-fold
```

## Relation to Base Training

| Base Training Stage | Ensemble System Equivalent |
|---------------------|----------------------------|
| Smoke/Verification | Verify each backbone loads correctly |
| K-fold CV | Load OOF predictions for stacking |
| Head Training | Train meta-learner/fusion head |
| Fine-tuning | End-to-end fusion fine-tuning |
| Test Evaluation | Evaluate ensemble on test set |
| Export & Package | Export ensemble using same exporter |
| Cleanup | Same memory management pattern |

### Reused Utilities from BASE-BACK

```python
# Configuration
from config.settings import (
    BACKBONES, NUM_CLASSES, IMG_SIZE, BATCH_SIZE,
    CKPT_DIR, METRICS_DIR, KFOLD_DIR, DEPLOY_DIR, ...
)

# Utils
from utils import (
    logger, smoke_checker, DEVICE, set_seed,
    compute_file_sha256
)

# Data
from utils.datasets import (
    WindowsCompatibleImageFolder, create_optimized_transforms,
    create_optimized_dataloader, prepare_optimized_datasets
)

# Models
from models import create_custom_backbone_safe, BACKBONE_MAP

# Training
from training import (
    train_epoch_optimized, validate_epoch_optimized,
    get_loss_function_for_backbone, create_optimized_optimizer,
    create_improved_scheduler
)
```

## Memory Management

Same cleanup pattern as base training:

```python
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Between every operation
safe_delete_model(model)
cleanup_memory()
```

## BatchNorm Recomputation

For weight-space ensembles (model soups, SWA):

```python
def recompute_bn_statistics(model, train_loader, num_batches=100):
    """Recompute BN stats after weight averaging"""
    model.train()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            _ = model(images)  # Updates BN running stats
    model.eval()
```

## Testing

Run automated unit tests:

```bash
cd tests
pytest debug_tests.py -v
```

Tests cover:
- Discovery
- Model loading
- Prediction extraction
- Ensemble building
- Stacking
- Fusion
- Export validation

## Logging

Uses the same logger as base training:

```python
from utils import logger

logger.info("Stage 1: Loading models...")
logger.warning("Model X not found, skipping")
logger.error("Failed to load checkpoint")
```

All logs follow the same format and verbosity as base training pipeline.

## Examples

### Quick Start: Build Simple Voting Ensemble

```bash
# 1. Discover models
python ensemble_orchestrator.py discover

# 2. Create minimal config
cat > configs/quick_ensemble.yaml << EOF
backbones:
  include:
    - CustomConvNeXt
    - CustomViTHybrid
    - CustomSwinTransformer
runs:
  score_ensemble:
    enable: true
    methods: [soft_voting, hard_voting]
EOF

# 3. Build ensemble
python ensemble_orchestrator.py build-ensemble --config configs/quick_ensemble.yaml

# 4. Export
python ensemble_orchestrator.py export --model-name soft_voting_ensemble
```

### Advanced: Full Ensemble Pipeline

```bash
# Build all ensemble types
python ensemble_orchestrator.py build-ensemble --config configs/ensemble_config.yaml
python ensemble_orchestrator.py stack --config configs/ensemble_config.yaml
python ensemble_orchestrator.py fusion-train --config configs/ensemble_config.yaml
python ensemble_orchestrator.py meta-fuse --config configs/ensemble_config.yaml

# Evaluate
python ensemble_orchestrator.py evaluate --config configs/ensemble_config.yaml

# Distill best ensemble
python ensemble_orchestrator.py distill --config configs/ensemble_config.yaml
```

## Troubleshooting

### Issue: "Catalog not found"
**Solution**: Run `python ensemble_orchestrator.py discover` first

### Issue: "No K-fold results for backbone X"
**Solution**: That backbone wasn't trained with K-fold CV. Either:
- Retrain with `ENABLE_KFOLD_CV=True`
- Exclude from stacking: add to `backbones.exclude` in config

### Issue: "OOM during fusion"
**Solution**: Use streaming inference (enabled by default) or reduce batch size in config

### Issue: "Export failed for ensemble"
**Solution**: Check `export_status.json` in deployment directory. May need to install export dependencies (ONNX, TFLite, etc.)

### Issue: "BN stats mismatch after soup"
**Solution**: Ensure `recompute_bn: true` in model_soup config

## Performance Tips

1. **Discovery**: Run once and reuse catalog (saved to disk)
2. **Streaming inference**: Load models one at a time for large ensembles
3. **Pruning**: Use diversity-based selection to reduce ensemble size
4. **Distillation**: Compress large ensembles into single fast model
5. **Caching**: OOF predictions and embeddings are cached to disk

## Architecture

The system is designed with strict separation:

- **BASE-BACK**: Base training pipeline (NEVER modified)
- **ensemble_system**: Ensemble operations (imports from BASE-BACK)
- **Shared utilities**: Reused via imports
- **Shared directories**: CKPT_DIR, METRICS_DIR, DEPLOY_DIR, etc.

This ensures:
- No disruption to base training
- Consistent file formats and structures
- Easy maintenance and updates
- Clear separation of concerns

## License

Same license as the base training framework.

## Citation

If using this ensemble system, please cite both the base framework and the ensemble orchestrator.
