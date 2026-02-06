"""
Ensemble Plotting Utilities
Comprehensive visualization functions for ensemble results
"""
from __future__ import annotations

import matplotlib
import numpy as np


matplotlib.use('Agg')
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


BASE_BACK_PATH = Path(__file__).parent.parent / 'BASE-BACK' / 'src'
if str(BASE_BACK_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_BACK_PATH))

from utils import logger


# High-quality plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 1200
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.format'] = 'tiff'
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str = "Confusion Matrix"
):
    """
    Generate and save confusion matrix plot with both raw counts and normalized values
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title(f'{title} (Raw Counts)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)

        # Normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title(f'{title} (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='y', rotation=0)

        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff', pil_kwargs={'compression': None})
        plt.close()

        logger.info(f"  Confusion matrix saved to: {save_path}")

    except Exception as e:
        logger.error(f"  x Failed to create confusion matrix: {e}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str = "ROC Curves",
    num_classes: int | None = None
):
    """
    Generate and save ROC curves for all classes
    """
    try:
        if num_classes is None:
            num_classes = len(class_names)

        # Binarize labels for multi-class ROC
        y_true_bin = np.asarray(label_binarize(y_true, classes=np.arange(num_classes)))

        _fig, ax = plt.subplots(figsize=(12, 10))

        # Plot ROC curve for each class
        colors = plt.colormaps['tab20'](np.linspace(0, 1, num_classes))

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{class_name} (AUC = {roc_auc:.3f})')

        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff', pil_kwargs={'compression': None})
        plt.close()

        logger.info(f"  ROC curves saved to: {save_path}")

    except Exception as e:
        logger.error(f"  x Failed to create ROC curves: {e}")


def plot_training_history(
    history: dict,
    save_path: Path,
    title: str = "Training History"
):
    """
    Plot training and validation metrics over epochs
    """
    try:
        _fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
            axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)

        # Plot accuracy
        if 'train_acc' in history and 'val_acc' in history:
            epochs = range(1, len(history['train_acc']) + 1)
            axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
            axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff', pil_kwargs={'compression': None})
        plt.close()

        logger.info(f"  Training history saved to: {save_path}")

    except Exception as e:
        logger.error(f"  x Failed to create training history: {e}")


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str = "Per-Class Metrics"
):
    """
    Plot per-class precision, recall, and F1-score
    """
    try:
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, _support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        x = np.arange(len(class_names))
        width = 0.25

        _fig, ax = plt.subplots(figsize=(14, 6))

        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])

        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=7)

        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff', pil_kwargs={'compression': None})
        plt.close()

        logger.info(f"  Per-class metrics saved to: {save_path}")

    except Exception as e:
        logger.error(f"  x Failed to create per-class metrics: {e}")


def plot_ensemble_comparison(
    results: dict | None = None,
    save_path: Path | None = None,
    title: str = "Ensemble Methods Comparison",
    metric: str = 'test_accuracy',
    # Alternative interface
    ensemble_names: list[str] | None = None,
    ensemble_accuracies: list[float] | None = None,
    output_dir: Path | None = None,
    prefix: str = ""
):
    """
    Compare multiple ensemble methods in a bar chart
    Supports two interfaces:
    1. results dict + save_path
    2. ensemble_names + ensemble_accuracies + output_dir + prefix
    """
    try:
        # Handle alternative interface
        if ensemble_names is not None and ensemble_accuracies is not None:
            methods = ensemble_names
            scores = ensemble_accuracies
            if save_path is None and output_dir is not None:
                save_path = Path(output_dir) / f'{prefix}_ensemble_comparison.tiff'
        elif results is not None:
            methods = list(results.keys())
            scores = [results[m][metric] for m in methods]
        else:
            logger.error("  x plot_ensemble_comparison: no data provided (results or ensemble_names/accuracies)")
            return

        _fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.colormaps['viridis'](np.linspace(0.3, 0.9, len(methods)))
        bars = ax.bar(methods, scores, color=colors, edgecolor='black', linewidth=1.5)

        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([min(scores) * 0.95, 1.0])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff', pil_kwargs={'compression': None})
        plt.close()

        logger.info(f"  Ensemble comparison saved to: {save_path}")

    except Exception as e:
        logger.error(f"  x Failed to create ensemble comparison: {e}")


def create_all_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    prefix: str = "",
    history: dict | None = None
):
    """
    Create all standard plots for an ensemble method
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Generating plots for {prefix}...")

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        output_dir / f'{prefix}_confusion_matrix.tiff',
        title=f'{prefix} Confusion Matrix'
    )

    # ROC curves
    plot_roc_curves(
        y_true, y_probs, class_names,
        output_dir / f'{prefix}_roc_curves.tiff',
        title=f'{prefix} ROC Curves',
        num_classes=len(class_names)
    )

    # Per-class metrics
    plot_per_class_metrics(
        y_true, y_pred, class_names,
        output_dir / f'{prefix}_per_class_metrics.tiff',
        title=f'{prefix} Per-Class Metrics'
    )

    # Training history (if available)
    if history is not None:
        plot_training_history(
            history,
            output_dir / f'{prefix}_training_history.tiff',
            title=f'{prefix} Training History'
        )
