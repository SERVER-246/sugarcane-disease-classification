# 🌿 Sugarcane Disease Classification Framework

## Comprehensive Project Documentation

**Version:** 1.0.0  
**Last Updated:** December 2025  
**Authors:** DBT Project Team  
**License:** Research/Academic Use

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Phase 1: Base Backbone Development](#phase-1-base-backbone-development)
4. [Phase 2: 15-COIN Ensemble Pipeline](#phase-2-15-coin-ensemble-pipeline)
5. [Results & Performance](#results--performance)
6. [Technical Architecture](#technical-architecture)
7. [Reproducibility Guide](#reproducibility-guide)
8. [Bug Fixes & Lessons Learned](#bug-fixes--lessons-learned)
9. [Future Work](#future-work)
10. [File Structure Reference](#file-structure-reference)

---

## Executive Summary

This project implements a **state-of-the-art disease classification system** for sugarcane crops using deep learning. The framework consists of two main phases:

### Phase 1: Custom Backbone Development
- **15 neural network architectures** built from scratch
- Implements CNNs, Vision Transformers, and hybrid models
- Two-stage training: head training → fine-tuning
- Multi-format model export (PyTorch, ONNX, TorchScript, etc.)

### Phase 2: 15-COIN Ensemble Pipeline  
A **7-stage hierarchical ensemble system** that combines all 15 backbones:
- Score-level fusion, stacking, feature fusion
- Mixture of Experts, Meta-ensemble
- Knowledge Distillation for deployment

### Key Results

| Metric | Value |
|--------|-------|
| **Best Individual Model** | CustomCSPDarkNet (96.04%) |
| **Best Ensemble (Stage 6)** | Meta-MLP (96.61%) |
| **Distilled Student** | 93.21% (6.2M params, 24MB) |
| **Dataset** | 13 disease classes, 10,607 images |
| **Training Time** | ~50 hours (backbones) + 4.7 hours (ensemble) |

---

## Project Overview

### Problem Statement

Sugarcane is a critical agricultural crop, and early disease detection is essential for preventing crop losses. This project addresses the challenge of **automated disease classification** from leaf images across 13 disease categories.

### Disease Classes

1. Black Stripe
2. Brown Spot  
3. Grassy Shoot Disease
4. Healthy
5. Leaf Flecking
6. Leaf Scorching
7. Mosaic
8. Pokkah Boeng
9. Red Rot
10. Ring Spot
11. Smut
12. Wilt
13. Yellow Leaf Disease

### Dataset Statistics

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 8,485 | 80% |
| Validation | 1,061 | 10% |
| Test | 1,061 | 10% |
| **Total** | **10,607** | 100% |

---

## Phase 1: Base Backbone Development

### Overview

Phase 1 develops **15 custom neural network architectures** from scratch, each designed with modern deep learning principles. All models are defined in `Base_backbones.py` (7,300+ lines) and the modularized `BASE-BACK/` directory.

### The 15 Backbone Architectures

| # | Architecture | Type | Embedding Dim | Key Innovation |
|---|--------------|------|---------------|----------------|
| 1 | **CustomConvNeXt** | CNN | 768 | Modernized ResNet with GELU, LayerNorm |
| 2 | **CustomEfficientNetV4** | CNN | 1280 | Compound scaling, SE blocks |
| 3 | **CustomGhostNetV2** | CNN | 1280 | Ghost modules for efficiency |
| 4 | **CustomResNetMish** | CNN | 2048 | ResNet with Mish activation |
| 5 | **CustomCSPDarkNet** | CNN | 1024 | Cross-stage partial connections |
| 6 | **CustomInceptionV4** | CNN | 512 | Multi-scale feature extraction |
| 7 | **CustomViTHybrid** | Hybrid | 768 | CNN stem + ViT transformer |
| 8 | **CustomSwinTransformer** | Transformer | 1024 | Shifted window attention |
| 9 | **CustomCoAtNet** | Hybrid | 768 | Convolution + Attention stages |
| 10 | **CustomRegNet** | CNN | 1536 | Regularized design space |
| 11 | **CustomDenseNetHybrid** | CNN | 512 | Dense connections + SE |
| 12 | **CustomDeiTStyle** | Transformer | 768 | Data-efficient ViT |
| 13 | **CustomMaxViT** | Hybrid | 768 | Multi-axis attention |
| 14 | **CustomMobileOne** | CNN | 384 | Reparameterizable blocks |
| 15 | **CustomDynamicConvNet** | CNN | 1024 | Dynamic kernel selection |

### Training Pipeline

Each backbone follows a **two-stage training approach**:

```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: HEAD TRAINING                    │
│  • Freeze backbone (pretrained weights if available)         │
│  • Train only classifier head                                │
│  • Epochs: 40 | LR: 1e-3 | Patience: 5                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 2: FINE-TUNING                      │
│  • Unfreeze entire model                                     │
│  • Train with lower learning rate                            │
│  • Epochs: 25 | Backbone LR: 1e-6 | Head LR: 1e-3           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 3: EXPORT                           │
│  • PyTorch state_dict (.pth)                                 │
│  • TorchScript (.pt)                                         │
│  • ONNX (.onnx)                                              │
│  • TensorRT, CoreML, TFLite (optional)                      │
└─────────────────────────────────────────────────────────────┘
```

### Training Configuration

```python
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 40
EPOCHS_FINETUNE = 25
PATIENCE_HEAD = 5
PATIENCE_FT = 5
WEIGHT_DECAY = 1e-4
BACKBONE_LR = 1e-6
HEAD_LR = 1e-3
```

### Individual Backbone Results

| Backbone | Train Acc | Val Acc | Test Acc |
|----------|-----------|---------|----------|
| CustomCSPDarkNet | 98.20% | 95.85% | **96.04%** |
| CustomGhostNetV2 | 98.23% | 94.06% | 94.53% |
| CustomDynamicConvNet | 97.96% | 94.63% | 94.53% |
| CustomMobileOne | 96.83% | 94.16% | 94.25% |
| CustomInceptionV4 | 97.30% | 94.53% | 93.97% |
| CustomRegNet | 97.14% | 94.63% | 93.87% |
| CustomDenseNetHybrid | 95.77% | 92.84% | 93.69% |
| CustomResNetMish | 96.65% | 94.63% | 93.59% |
| CustomEfficientNetV4 | 96.87% | 92.55% | 93.31% |
| CustomConvNeXt | 98.77% | 93.03% | 92.93% |
| CustomDeiTStyle | 94.94% | 90.29% | 91.33% |
| CustomViTHybrid | 95.82% | 91.61% | 91.23% |
| CustomCoAtNet | 88.45% | 84.35% | 86.43% |
| CustomSwinTransformer | 87.92% | 83.69% | 85.30% |
| CustomMaxViT | 87.88% | 85.49% | 85.30% |

---

## Phase 2: 15-COIN Ensemble Pipeline

### Overview

The **15-COIN (15-model Cascaded Optimized Integration Network)** is a 7-stage ensemble system that progressively combines predictions from all 15 backbones to achieve superior classification accuracy.

### Pipeline Architecture

```
                         ┌──────────────────────────────────────┐
                         │         15 BACKBONE MODELS           │
                         │  (CustomConvNeXt, CustomEfficientNet, │
                         │   CustomGhostNet, ... 15 total)      │
                         └──────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ↓                     ↓                     ↓
              [Predictions]         [Predictions]         [Predictions]
                    │                     │                     │
═══════════════════════════════════════════════════════════════════════════
                              STAGE 1: INDIVIDUAL
                         Extract predictions from each backbone
═══════════════════════════════════════════════════════════════════════════
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ↓                                           ↓
═══════════════════════════════════════════════════════════════════════════
           STAGE 2: SCORE ENSEMBLES              STAGE 3: STACKING
           • Soft Voting                         • Logistic Regression
           • Hard Voting                         • XGBoost
           • Weighted Voting                     • MLP Stacker
           • Logit Averaging
═══════════════════════════════════════════════════════════════════════════
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ↓
═══════════════════════════════════════════════════════════════════════════
                              STAGE 4: FEATURE FUSION
                    • Concatenation MLP
                    • Attention-based Fusion
                    • Bilinear Pooling
═══════════════════════════════════════════════════════════════════════════
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ↓                                           ↓
═══════════════════════════════════════════════════════════════════════════
           STAGE 5: MIXTURE OF EXPERTS           STAGE 6: META-ENSEMBLE
           • 15 Expert networks                  • Combines 11 ensembles
           • Gating network (top-k=5)            • XGBoost controller
           • Sparse activation                   • MLP controller
═══════════════════════════════════════════════════════════════════════════
                                          │
                                          ↓
═══════════════════════════════════════════════════════════════════════════
                         STAGE 7: KNOWLEDGE DISTILLATION
                    • Teacher: Stage 6 Meta-Ensemble (96.61%)
                    • Student: CompactStudentModel (6.2M params)
                    • Temperature: 3.0, Alpha: 0.7
═══════════════════════════════════════════════════════════════════════════
                                          │
                                          ↓
                         ┌────────────────────────────┐
                         │     DEPLOYABLE MODEL       │
                         │   93.21% accuracy          │
                         │   24MB model size          │
                         │   6.2M parameters          │
                         └────────────────────────────┘
```

### Stage-by-Stage Description

#### Stage 1: Individual Backbone Predictions

**Purpose:** Extract class predictions and embeddings from all 15 pretrained backbones.

**Significance:** Creates the foundation for all ensemble methods by generating:
- Probability distributions (logits/softmax)
- Feature embeddings for fusion
- Per-class confidence scores

**Output:** 
- `{backbone}_train_predictions.npz`
- `{backbone}_val_predictions.npz`  
- `{backbone}_test_predictions.npz`

---

#### Stage 2: Score-Level Ensembles

**Purpose:** Combine backbone predictions using classical ensemble methods.

**Methods:**
| Method | Description | Test Accuracy |
|--------|-------------|---------------|
| Soft Voting | Average probability distributions | 95.95% |
| Hard Voting | Majority vote on predictions | 95.76% |
| Weighted Voting | Accuracy-weighted averaging | 95.95% |
| **Logit Averaging** | Average raw logits before softmax | **96.14%** |

**Significance:** Simple but effective—logit averaging captures uncertainty better than probabilities, leading to best Stage 2 performance.

---

#### Stage 3: Stacking Ensemble

**Purpose:** Train meta-learners on backbone predictions.

**Methods:**
| Method | Val Accuracy | Test Accuracy |
|--------|--------------|---------------|
| Logistic Regression | 96.32% | 96.32% |
| **XGBoost** | 95.57% | **96.51%** |
| MLP | 96.70% | 96.23% |

**Significance:** Learns optimal combinations of backbone outputs. XGBoost's gradient boosting captures non-linear relationships between model predictions.

---

#### Stage 4: Feature Fusion

**Purpose:** Deep fusion of backbone embeddings (not just predictions).

**Methods:**
| Method | Val Accuracy | Test Accuracy |
|--------|--------------|---------------|
| Concatenation MLP | 96.98% | 96.14% |
| **Attention Fusion** | 96.42% | **96.42%** |
| Bilinear Pooling | 96.14% | 95.48% |

**Significance:** Attention fusion learns which backbone features are most relevant for each input, providing adaptive weighting.

---

#### Stage 5: Mixture of Experts (MoE)

**Purpose:** Sparse expert selection—only activate most relevant backbones per sample.

**Configuration:**
- 15 Expert networks (one per backbone)
- Gating network selects top-k=5 experts
- Sparse activation for efficiency

**Results:** 95.48% test accuracy

**Significance:** Demonstrates that not all models are needed for every sample—selective activation improves efficiency without sacrificing much accuracy.

---

#### Stage 6: Meta-Ensemble

**Purpose:** Ultimate combination of ALL previous stages (Stages 2-5).

**Inputs Combined (11 total):**
- Stage 2: soft_voting, hard_voting, weighted_voting, logit_averaging (4)
- Stage 3: logistic_regression, xgboost, mlp (3)
- Stage 4: concat_mlp, attention_fusion, bilinear_pooling (3)
- Stage 5: moe (1)

**Controllers:**
| Method | Val Accuracy | Test Accuracy |
|--------|--------------|---------------|
| XGBoost | 95.76% | 96.42% |
| **MLP** | 96.23% | **96.61%** |

**Significance:** This is the **highest accuracy achieved** in the entire pipeline. The meta-ensemble learns to leverage strengths of different ensemble strategies for different types of inputs.

---

#### Stage 7: Knowledge Distillation

**Purpose:** Compress the massive ensemble into a single deployable model.

**Configuration:**
```python
Temperature = 3.0     # Softens probability distributions
Alpha = 0.7           # Weight of distillation loss vs hard label loss
Epochs = 100          # Extended training
Patience = 15         # Early stopping
```

**Teacher:** Stage 6 MLP Meta-Ensemble (96.61%)  
**Student:** CompactStudentModel architecture

**Results:**
| Metric | Value |
|--------|-------|
| Test Accuracy | 93.21% |
| Model Size | 23.85 MB |
| Parameters | 6,232,493 |
| Accuracy Drop | 3.39% |

**Significance:** Achieves **93.21% accuracy with only 6.2M parameters**—suitable for edge deployment while retaining most of the ensemble's knowledge.

---

## Results & Performance

### Final Pipeline Results Summary

| Stage | Best Method | Test Accuracy | Improvement |
|-------|-------------|---------------|-------------|
| Stage 1 | CustomCSPDarkNet | 96.04% | Baseline |
| Stage 2 | Logit Averaging | 96.14% | +0.10% |
| Stage 3 | XGBoost Stacker | 96.51% | +0.47% |
| Stage 4 | Attention Fusion | 96.42% | +0.38% |
| Stage 5 | MoE (top-5) | 95.48% | -0.56% |
| **Stage 6** | **MLP Meta** | **96.61%** | **+0.57%** |
| Stage 7 | Distilled Student | 93.21% | -2.83% |

### Execution Time

| Phase | Duration |
|-------|----------|
| Phase 1 (15 Backbones) | ~50 hours |
| Phase 2 (7-Stage Pipeline) | 4.72 hours |
| **Total** | **~55 hours** |

### Key Observations

1. **Ensemble gains are incremental but consistent**: Each stage adds small improvements, accumulating to 0.57% over the best individual model.

2. **Meta-ensemble outperforms all**: Combining diverse ensemble strategies (voting, stacking, fusion, MoE) yields the best results.

3. **Knowledge distillation is effective**: The student model retains 96.5% of teacher accuracy (93.21/96.61) with 10x fewer parameters.

4. **Transformer models underperform on this dataset**: CNNs (CSPDarkNet, GhostNet) outperform ViT-based models, likely due to dataset size.

---

## Technical Architecture

### Directory Structure

```
F:\DBT-Base-DIr\
├── Base_backbones.py          # Original monolithic training script
├── BASE-BACK/                 # Modularized codebase
│   ├── src/
│   │   ├── config/           # Settings, hyperparameters
│   │   ├── models/           # 15 backbone definitions
│   │   ├── training/         # Training loops, schedulers
│   │   ├── export/           # Multi-format export
│   │   └── utils/            # Logging, checkpoints
│   └── tests/                # Unit tests
├── ensemble_system/           # 15-COIN pipeline
│   ├── stage1_individual.py
│   ├── stage2_score_ensembles.py
│   ├── stage3_stacking.py
│   ├── stage4_feature_fusion.py
│   ├── stage5_mixture_experts.py
│   ├── stage6_meta_ensemble.py
│   ├── stage7_distillation.py
│   └── run_15coin_pipeline.py
├── checkpoints/              # Saved model weights
├── ensembles/                # Ensemble outputs
├── split_dataset/            # Train/val/test splits
├── Data/                     # Raw images by class
└── pretrained_weights/       # ImageNet pretrained weights
```

### Key Components

#### Custom Building Blocks

```python
# Core image processing block
class CoreImageBlock(nn.Module):
    """Conv2d + BatchNorm + Activation + optional LayerNorm"""
    
# Reparameterizable block for MobileOne
class MobileOneBlock(nn.Module):
    """Multi-branch training → single conv inference"""
    
# Window-based attention for Swin
class WindowAttention(nn.Module):
    """Multi-head self-attention with shifted windows"""
    
# Dynamic convolution
class DynamicConv(nn.Module):
    """Multiple kernel sizes fused at runtime"""
```

#### Data Pipeline

```python
# Windows-compatible image loading
class WindowsCompatibleImageFolder(ImageFolder):
    """Handles Windows path separators correctly"""

# Optimized transforms
def create_optimized_transforms(img_size, is_training=True):
    """
    Training: RandomResizedCrop, HorizontalFlip, ColorJitter, etc.
    Validation: Resize, CenterCrop, Normalize
    """
```

---

## Reproducibility Guide

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd DBT-Base-DIr

# Run complete pipeline from scratch
python reproduce_pipeline.py --mode full

# Or run individual phases
python reproduce_pipeline.py --mode backbones_only
python reproduce_pipeline.py --mode ensemble_only
```

### Environment Setup

**Requirements:**
- Python 3.8+
- PyTorch 2.0+ with CUDA
- NVIDIA GPU with 8GB+ VRAM
- 50GB+ disk space

**Install dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn xgboost matplotlib seaborn tqdm pillow
pip install onnx onnxruntime tensorrt  # Optional for export
```

### Running the Pipeline

See `reproduce_pipeline.py` for detailed options and `REPRODUCIBILITY.md` for comprehensive instructions.

---

## Bug Fixes & Lessons Learned

### Critical Bug: DataLoader Configuration for Stage 7

**Problem:** Knowledge distillation validation accuracy stuck at 17.72% (random chance for 13 classes).

**Root Cause:** 
```python
# WRONG - used for Stages 1-6 (inference only)
train_loader = create_dataloader(train_ds, shuffle=False)  
transform = create_transforms(is_training=False)

# CORRECT - needed for Stage 7 (training new model)
train_loader = create_dataloader(train_ds, shuffle=True)
transform = create_transforms(is_training=True)
```

**Lesson:** Inference stages need deterministic data order; training stages need shuffling and augmentation.

### Other Issues Resolved

1. **Undefined training history variables** in `stage7_distillation.py`
2. **Windows multiprocessing** requires `spawn` method
3. **Transformer positional encoding** must use `nn.Parameter`
4. **MobileOne initialization** scaled down to prevent gradient explosion

---

## Future Work

1. **Larger dataset**: Expand beyond 10,607 images
2. **Self-supervised pretraining**: Use unlabeled sugarcane images
3. **Real-time inference**: Optimize student model for mobile deployment
4. **Multi-task learning**: Add disease severity prediction
5. **Explainability**: Integrate Grad-CAM visualizations

---

## File Structure Reference

### Core Files

| File | Purpose |
|------|---------|
| `Base_backbones.py` | Original monolithic script (7,300+ lines) |
| `reproduce_pipeline.py` | **Main reproducibility script** |
| `PROJECT_SUMMARY.md` | This document |

### Phase 1 (BASE-BACK)

| Directory | Contents |
|-----------|----------|
| `BASE-BACK/src/models/` | 15 backbone class definitions |
| `BASE-BACK/src/training/` | Training loops, schedulers |
| `BASE-BACK/src/export/` | ONNX, TorchScript export |
| `BASE-BACK/src/config/` | Hyperparameters, paths |

### Phase 2 (Ensemble System)

| File | Stage |
|------|-------|
| `stage1_individual.py` | Extract backbone predictions |
| `stage2_score_ensembles.py` | Voting methods |
| `stage3_stacking.py` | Meta-learner training |
| `stage4_feature_fusion.py` | Deep feature fusion |
| `stage5_mixture_experts.py` | MoE implementation |
| `stage6_meta_ensemble.py` | Final meta-ensemble |
| `stage7_distillation.py` | Knowledge distillation |
| `run_15coin_pipeline.py` | Pipeline orchestrator |

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{dbt_sugarcane_2025,
  title = {Sugarcane Disease Classification Framework with 15-COIN Ensemble},
  author = {DBT Project Team},
  year = {2025},
  url = {https://github.com/your-repo}
}
```

---

## License

This project is for research and academic purposes. See LICENSE file for details.

---

**End of Documentation**

*Last generated: December 2025*
