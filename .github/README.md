# 🌿 Sugarcane Disease Classification System

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A comprehensive deep learning framework for sugarcane disease classification featuring **15 custom-built neural network architectures** and a **7-stage hierarchical ensemble system** achieving **96.61% accuracy**.

---

## 📊 Results Summary

| Model/Ensemble        | Test Accuracy | Parameters | Notes         |
| --------------------- | ------------- | ---------- | ------------- |
| **Meta-MLP Ensemble** | **96.61%**    | 262K       | Best overall  |
| XGBoost Stacker       | 96.51%        | -          | Non-neural    |
| Attention Fusion      | 96.42%        | 18.8M      | Feature-level |
| Logit Averaging       | 96.14%        | 0          | Zero overhead |
| Distilled Student     | 93.21%        | 6.2M       | Mobile-ready  |
| Best Single Backbone  | 95.39%        | ~25M       | CustomMaxViT  |

---

## 🚀 Quick Start

### Option 1: Full Reproducibility (Recommended)

```bash
# Clone and navigate
git clone https://github.com/SERVER-246/sugarcane-disease-classification.git
cd sugarcane-disease-classification

# Run interactive setup
python reproduce_pipeline.py --mode interactive
```

This will:

1. ✅ Validate your environment
2. ✅ Prompt for dataset location
3. ✅ Split dataset into train/val/test
4. ✅ Train 15 backbone models
5. ✅ Run 7-stage ensemble pipeline

### Option 2: Quick Test (30 minutes)

```bash
python reproduce_pipeline.py --mode quick_test --data-dir /path/to/images
```

### Option 3: Ensemble Only (Pre-trained backbones)

```bash
python reproduce_pipeline.py --mode ensemble_only
```

---

## 📁 Project Structure

```
sugarcane-disease-classification/
│
├── reproduce_pipeline.py         # 🔥 One-click full reproducibility
├── PROJECT_SUMMARY.md            # 📖 Complete documentation
├── Base_backbones.py             # 15 backbone architectures
│
├── ensemble_system/              # 7-stage ensemble pipeline
│   ├── run_15coin_pipeline.py    # Pipeline orchestrator
│   ├── stage1_predictions.py     # Individual predictions
│   ├── stage2_score_ensembles.py # Voting & averaging
│   ├── stage3_stacking.py        # Meta-learners
│   ├── stage4_feature_fusion.py  # Feature-level fusion
│   ├── stage5_moe.py             # Mixture of Experts
│   ├── stage6_meta_ensemble.py   # Meta-ensemble
│   └── stage7_distillation.py    # Knowledge distillation
│
├── Data/                         # Raw images by class
├── split_dataset/                # Train/val/test splits
│   ├── train/
│   ├── val/
│   └── test/
│
├── checkpoints/                  # Trained model weights
├── ensembles/                    # Ensemble results
├── metrics_output/               # Training metrics
└── plots_metrics/                # Visualizations
```

---

## 🏗️ Architecture Overview

### 15 Custom Backbone Models

| Category         | Architectures                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| **CNNs**         | ConvNeXt, EfficientNetV4, GhostNetV2, ResNetMish, CSPDarkNet, InceptionV4, DenseNetHybrid, RegNet |
| **Transformers** | ViTHybrid, SwinTransformer, DeiTStyle                                                             |
| **Hybrids**      | CoAtNet, MaxViT, MobileOne, DynamicConvNet                                                        |

### 7-Stage 15-COIN Ensemble Pipeline

```
Stage 1: Individual Predictions → 15 backbone outputs
    ↓
Stage 2: Score Ensembles → Voting, Averaging (96.14%)
    ↓
Stage 3: Stacking → LR, XGBoost (96.51%), MLP
    ↓
Stage 4: Feature Fusion → Concat, Attention (96.42%), Bilinear
    ↓
Stage 5: MoE → Mixture of Experts (95.48%)
    ↓
Stage 6: Meta-Ensemble → Combine all above (96.61%)
    ↓
Stage 7: Knowledge Distillation → Lightweight student (93.21%)
```

---

## 📋 Requirements

### Hardware

* **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
* **RAM**: 32GB+ recommended
* **Storage**: 50GB+ free space

### Software

```bash
# Core dependencies
pip install torch torchvision numpy pandas scikit-learn xgboost matplotlib seaborn tqdm pillow

# Optional (for model export)
pip install onnx onnxruntime tensorrt coremltools
```

---

## 📈 Dataset

**13 Sugarcane Disease Classes:**

| ID | Disease             | Training Samples |
| -- | ------------------- | ---------------- |
| 1  | Black Stripe        | 523              |
| 2  | Brown Spot          | 1,064            |
| 3  | Grassy Shoot        | 465              |
| 4  | Healthy             | 701              |
| 5  | Leaf Flecking       | 513              |
| 6  | Leaf Scorching      | 504              |
| 7  | Mosaic              | 500              |
| 8  | Pokkah Boeng        | 1,095            |
| 9  | Red Rot             | 809              |
| 10 | Ring Spot           | 528              |
| 11 | Smut                | 1,038            |
| 12 | Wilt                | 243              |
| 13 | Yellow Leaf Disease | 502              |

**Total: 10,607 images** (8,485 train / 1,061 val / 1,061 test)

---

## 🎯 Training Pipeline

### Two-Stage Training Protocol

```
Stage 1: Head Training (40 epochs)
├── Freeze backbone weights
├── Train only classification head
├── LR: 1e-3, Early stopping: 5 epochs
└── Purpose: Initialize head without disrupting features

Stage 2: Fine-tuning (25 epochs)
├── Unfreeze all layers
├── End-to-end training
├── LR: 1e-6, Early stopping: 5 epochs
└── Purpose: Adapt entire model to dataset
```

---

## 📊 Full Results

### Individual Backbone Performance

| Backbone              | Val Acc | Test Acc | Parameters |
| --------------------- | ------- | -------- | ---------- |
| CustomMaxViT          | 97.07%  | 95.39%   | 24.8M      |
| CustomConvNeXt        | 97.17%  | 94.72%   | 17.9M      |
| CustomSwinTransformer | 96.98%  | 94.81%   | 20.3M      |
| CustomEfficientNetV4  | 95.29%  | 93.78%   | 18.5M      |
| CustomCoAtNet         | 96.42%  | 93.78%   | 15.8M      |
| ...                   | ...     | ...      | ...        |

### Ensemble Performance (Stages 2-7)

| Stage | Method            | Test Accuracy |
| ----- | ----------------- | ------------- |
| 2     | Logit Averaging   | 96.14%        |
| 3     | XGBoost Stacker   | 96.51%        |
| 4     | Attention Fusion  | 96.42%        |
| 5     | MoE (8 experts)   | 95.48%        |
| 6     | **Meta-MLP**      | **96.61%**    |
| 7     | Distilled Student | 93.21%        |

---

## 🔧 Configuration

### Environment Variables

```bash
# Required
export DBT_BASE_DIR=/path/to/project
export DBT_SPLIT_DIR=/path/to/split_dataset

# Debug mode
export DBT_DEBUG_MODE=true
export DBT_DEBUG_BACKBONE=CustomCoAtNet
export DBT_DEBUG_HEAD_EPOCHS=5
export DBT_DEBUG_FT_EPOCHS=3
```

### Key Hyperparameters

```python
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 40
EPOCHS_FINETUNE = 25
PATIENCE_HEAD = 5
PATIENCE_FT = 5
HEAD_LR = 1e-3
BACKBONE_LR = 1e-6
WEIGHT_DECAY = 1e-4
```

---

## 📖 Documentation & Project History

* **Full technical documentation**: `PROJECT_SUMMARY.md`. For a comprehensive project overview, phase descriptions, training configuration, and detailed per-backbone results see `PROJECT_SUMMARY.md`.

* **Project evolution and engineering timeline**: `EVOLUTION.md` documents how the project moved from a single-file prototype to a modular, production-ready pipeline (phases: Base-1.py → Base_backbones.py → BASE-BACK/ → run_pipeline.py → ensemble_system → reproduce_pipeline.py). It includes the rationale behind modularization, checkpointing, K-fold CV, export system, and the introduction of the 7-stage ensemble pipeline.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* PyTorch team for the deep learning framework
* NVIDIA for CUDA and TensorRT
* The sugarcane research community

---

## 📞 Citation

If you use this code in your research, please cite:

```bibtex
@software{sugarcane_disease_classification,
  title = {Sugarcane Disease Classification Ensemble},
  author = {DBT Project Team},
  year = {2025},
  url = {https://github.com/SERVER-246/sugarcane-disease-classification}
}
```

---

**Last Updated:** December 2025 | **Status:** ✅ Complete | **Best Accuracy:** 96.61%
