"""
Visualization functions for model training metrics and evaluation
Generates high-quality TIFF plots at 1200 DPI
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize

# Handle both package imports and direct sys.path imports
try:
    from ..utils import logger
    from ..config.settings import PLOTS_DIR
except ImportError:
    from utils import logger
    from config.settings import PLOTS_DIR

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 1200
plt.rcParams['savefig.dpi'] = 1200
plt.rcParams['font.family'] = 'sans-serif'


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """Generate and save confusion matrix plot with both raw counts and normalized values"""
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
        
        logger.info(f"[OK] Confusion matrix saved to: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"x Failed to create confusion matrix: {e}")
        return False


def plot_roc_curves(y_true, y_probs, class_names, save_path, title="ROC Curves"):
    """Generate and save ROC curves for all classes"""
    try:
        n_classes = len(class_names)
        
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
        
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
        
        logger.info(f"[OK] ROC curves saved to: {save_path}")
        logger.info(f"  Micro-average AUC: {roc_auc['micro']:.4f}")
        
        return roc_auc
        
    except Exception as e:
        logger.error(f"x Failed to create ROC curves: {e}")
        return {}


def plot_training_history(history, save_path, title="Training History"):
    """Plot comprehensive training history including loss, accuracy, and F1 score"""
    try:
        if not history or ('head' not in history and 'finetune' not in history):
            logger.warning("No training history available for plotting")
            return False
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Extract data
        head_history = history.get('head', [])
        finetune_history = history.get('finetune', [])
        
        head_epochs = len(head_history)
        finetune_epochs = len(finetune_history)
        total_epochs = head_epochs + finetune_epochs
        
        if head_history:
            head_data = np.array(head_history)
            head_train_loss = head_data[:, 0]
            head_val_loss = head_data[:, 1]
            head_val_acc = head_data[:, 2]
            head_train_acc = head_data[:, 3]
            head_val_f1 = head_data[:, 4]
        else:
            head_train_loss = head_val_loss = head_val_acc = head_train_acc = head_val_f1 = np.array([])
        
        if finetune_history:
            finetune_data = np.array(finetune_history)
            finetune_train_loss = finetune_data[:, 0]
            finetune_val_loss = finetune_data[:, 1]
            finetune_val_acc = finetune_data[:, 2]
            finetune_train_acc = finetune_data[:, 3]
            finetune_val_f1 = finetune_data[:, 4]
        else:
            finetune_train_loss = finetune_val_loss = finetune_val_acc = finetune_train_acc = finetune_val_f1 = np.array([])
        
        # Concatenate data
        epochs_head = np.arange(1, head_epochs + 1)
        epochs_finetune = np.arange(head_epochs + 1, total_epochs + 1)
        
        # Plot 1: Training Loss
        ax = axes[0]
        if len(head_train_loss) > 0:
            ax.plot(epochs_head, head_train_loss, label='Head Train', color='#1f77b4', linewidth=2, marker='o', markersize=4)
        if len(finetune_train_loss) > 0:
            ax.plot(epochs_finetune, finetune_train_loss, label='Finetune Train', color='#ff7f0e', linewidth=2, marker='s', markersize=4)
        ax.axvline(x=head_epochs, color='red', linestyle='--', linewidth=2, label='Stage Boundary')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Validation Loss
        ax = axes[1]
        if len(head_val_loss) > 0:
            ax.plot(epochs_head, head_val_loss, label='Head Val', color='#2ca02c', linewidth=2, marker='o', markersize=4)
        if len(finetune_val_loss) > 0:
            ax.plot(epochs_finetune, finetune_val_loss, label='Finetune Val', color='#d62728', linewidth=2, marker='s', markersize=4)
        ax.axvline(x=head_epochs, color='red', linestyle='--', linewidth=2, label='Stage Boundary')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title('Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy
        ax = axes[2]
        if len(head_train_acc) > 0:
            ax.plot(epochs_head, head_train_acc, label='Head Train', color='#9467bd', linewidth=2, marker='o', markersize=4, alpha=0.7)
        if len(head_val_acc) > 0:
            ax.plot(epochs_head, head_val_acc, label='Head Val', color='#8c564b', linewidth=2, marker='o', markersize=4)
        if len(finetune_train_acc) > 0:
            ax.plot(epochs_finetune, finetune_train_acc, label='Finetune Train', color='#e377c2', linewidth=2, marker='s', markersize=4, alpha=0.7)
        if len(finetune_val_acc) > 0:
            ax.plot(epochs_finetune, finetune_val_acc, label='Finetune Val', color='#7f7f7f', linewidth=2, marker='s', markersize=4)
        ax.axvline(x=head_epochs, color='red', linestyle='--', linewidth=2, label='Stage Boundary')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy Progress', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Plot 4: F1 Score
        ax = axes[3]
        if len(head_val_f1) > 0:
            ax.plot(epochs_head, head_val_f1, label='Head Val F1', color='#bcbd22', linewidth=2, marker='o', markersize=4)
        if len(finetune_val_f1) > 0:
            ax.plot(epochs_finetune, finetune_val_f1, label='Finetune Val F1', color='#17becf', linewidth=2, marker='s', markersize=4)
        ax.axvline(x=head_epochs, color='red', linestyle='--', linewidth=2, label='Stage Boundary')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('F1 Score Progress', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff')
        plt.close()
        
        logger.info(f"[OK] Training history saved to: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"x Failed to create training history plot: {e}")
        logger.exception("Full traceback:")
        return False


def plot_per_class_metrics(y_true, y_pred, class_names, save_path, title="Per-Class Metrics"):
    """Generate per-class precision, recall, and F1-score bar plots"""
    try:
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        
        metrics = {
            'precision': [report[cls]['precision'] for cls in class_names],
            'recall': [report[cls]['recall'] for cls in class_names],
            'f1-score': [report[cls]['f1-score'] for cls in class_names]
        }
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(class_names))
        width = 0.25
        
        ax.bar(x - width, metrics['precision'], width, label='Precision', color='#1f77b4', alpha=0.8)
        ax.bar(x, metrics['recall'], width, label='Recall', color='#ff7f0e', alpha=0.8)
        ax.bar(x + width, metrics['f1-score'], width, label='F1-Score', color='#2ca02c', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, cls in enumerate(class_names):
            ax.text(i - width, metrics['precision'][i] + 0.02, f"{metrics['precision'][i]:.2f}", 
                   ha='center', va='bottom', fontsize=8)
            ax.text(i, metrics['recall'][i] + 0.02, f"{metrics['recall'][i]:.2f}", 
                   ha='center', va='bottom', fontsize=8)
            ax.text(i + width, metrics['f1-score'][i] + 0.02, f"{metrics['f1-score'][i]:.2f}", 
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='tiff')
        plt.close()
        
        logger.info(f"[OK] Per-class metrics saved to: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"x Failed to create per-class metrics plot: {e}")
        return False


def generate_all_visualizations(model, backbone_name, history, val_loader, 
                                class_names, criterion, device):
    """Generate all visualizations for a trained model"""
    logger.info(f"Generating comprehensive visualizations for {backbone_name}...")
    
    plot_paths = {}
    
    try:
        # Get validation predictions
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
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
        
        # Ensure PLOTS_DIR exists
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        # 1. Training History Plot
        history_path = PLOTS_DIR / f"{backbone_name}_training_history.tiff"
        if plot_training_history(history, history_path, f"{backbone_name} Training History"):
            plot_paths['training_history'] = str(history_path)
        
        # 2. Confusion Matrix
        cm_path = PLOTS_DIR / f"{backbone_name}_confusion_matrix.tiff"
        if plot_confusion_matrix(all_labels, all_preds, class_names, cm_path, 
                                f"{backbone_name} Confusion Matrix"):
            plot_paths['confusion_matrix'] = str(cm_path)
        
        # 3. ROC Curves
        roc_path = PLOTS_DIR / f"{backbone_name}_roc_curves.tiff"
        roc_aucs = plot_roc_curves(all_labels, all_probs, class_names, roc_path,
                                   f"{backbone_name} ROC Curves")
        if roc_aucs:
            plot_paths['roc_curves'] = str(roc_path)
        
        # 4. Per-Class Metrics
        metrics_path = PLOTS_DIR / f"{backbone_name}_per_class_metrics.tiff"
        if plot_per_class_metrics(all_labels, all_preds, class_names, metrics_path,
                                  f"{backbone_name} Per-Class Metrics"):
            plot_paths['per_class_metrics'] = str(metrics_path)
        
        logger.info(f"[OK] Generated {len(plot_paths)} visualizations for {backbone_name}")
        
    except Exception as e:
        logger.error(f"x Failed to generate visualizations for {backbone_name}: {e}")
        logger.exception("Full traceback:")
    
    return plot_paths


def save_visualization_summary(plot_paths, backbone_name, save_dir):
    """Save a summary of all generated visualizations"""
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
        
        logger.info(f"[OK] Visualization summary saved to: {summary_path}")
        return str(summary_path)
        
    except Exception as e:
        logger.error(f"x Failed to save visualization summary: {e}")
        return None

