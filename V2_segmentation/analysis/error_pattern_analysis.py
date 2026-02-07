"""
V2_segmentation/analysis/error_pattern_analysis.py
===================================================
Systematic error-pattern analysis for V1 backbones.

Runs inference on the validation/test set and produces:
  - Per-backbone confusion matrices
  - Per-class accuracy / F1 / recall breakdowns
  - Cross-backbone error correlation (which classes are hard for ALL models?)
  - Misclassified-sample inventories (image paths + predicted vs true)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from V2_segmentation.config import (
    ANALYSIS_DIR, CLASS_NAMES, DEVICE, NUM_CLASSES,
)

logger = logging.getLogger(__name__)


class ErrorPatternAnalyzer:
    """Analyze error patterns across V1 backbones.

    Parameters
    ----------
    model : nn.Module
        V1 backbone with trained weights.
    backbone_name : str
        Identifier string.
    device : torch.device
        Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        backbone_name: str,
        device: torch.device = DEVICE,
    ) -> None:
        self.model = model.to(device).eval()
        self.backbone_name = backbone_name
        self.device = device

    @torch.no_grad()
    def run_inference(
        self, dataloader: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on a dataloader.

        Returns
        -------
        preds : (N,) predicted class indices.
        labels : (N,) true class indices.
        probs : (N, num_classes) softmax probabilities.
        """
        all_preds, all_labels, all_probs = [], [], []

        for images, targets in dataloader:
            images = images.to(self.device)
            outputs = self.model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(targets.numpy())
            all_probs.append(probs.cpu().numpy())

        return (
            np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs),
        )

    def compute_confusion_matrix(
        self, preds: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Compute confusion matrix (true × pred).

        Returns
        -------
        cm : (num_classes, num_classes) array.
        """
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for t, p in zip(labels, preds):
            cm[t, p] += 1
        return cm

    def per_class_metrics(
        self, preds: np.ndarray, labels: np.ndarray
    ) -> dict[str, dict[str, float]]:
        """Per-class accuracy, precision, recall, F1.

        Returns
        -------
        dict mapping class_name → {accuracy, precision, recall, f1, support}.
        """
        cm = self.compute_confusion_matrix(preds, labels)
        metrics = {}
        for idx, name in enumerate(CLASS_NAMES):
            tp = cm[idx, idx]
            fn = cm[idx, :].sum() - tp
            fp = cm[:, idx].sum() - tp
            support = cm[idx, :].sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = tp / support if support > 0 else 0.0

            metrics[name] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
        return metrics

    def find_systematic_errors(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.05,
    ) -> list[dict[str, Any]]:
        """Find class pairs with high confusion rates.

        Parameters
        ----------
        threshold : float
            Minimum confusion rate (true→pred) to report.

        Returns
        -------
        list of dicts: {true_class, pred_class, rate, count}.
        """
        cm = self.compute_confusion_matrix(preds, labels)
        errors = []
        for i in range(NUM_CLASSES):
            row_sum = cm[i].sum()
            if row_sum == 0:
                continue
            for j in range(NUM_CLASSES):
                if i == j:
                    continue
                rate = cm[i, j] / row_sum
                if rate >= threshold:
                    errors.append({
                        "true_class": CLASS_NAMES[i],
                        "pred_class": CLASS_NAMES[j],
                        "rate": float(rate),
                        "count": int(cm[i, j]),
                    })
        errors.sort(key=lambda x: x["rate"], reverse=True)
        return errors

    def save_analysis(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray,
        output_dir: Path | None = None,
    ) -> Path:
        """Save full error analysis to disk.

        Saves confusion matrix, per-class metrics, systematic errors, and plots.
        """
        output_dir = Path(output_dir or ANALYSIS_DIR / self.backbone_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        cm = self.compute_confusion_matrix(preds, labels)
        metrics = self.per_class_metrics(preds, labels)
        errors = self.find_systematic_errors(preds, labels)

        np.save(output_dir / "confusion_matrix.npy", cm)
        with open(output_dir / "per_class_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(output_dir / "systematic_errors.json", "w") as f:
            json.dump(errors, f, indent=2)

        # Overall accuracy
        overall_acc = (preds == labels).mean()
        logger.info(f"  {self.backbone_name} accuracy: {overall_acc:.4f}")
        logger.info(f"  Top systematic errors: {errors[:5]}")

        # Plot confusion matrix
        self._plot_confusion(cm, output_dir)

        logger.info(f"  Error analysis saved to {output_dir}")
        return output_dir

    def _plot_confusion(self, cm: np.ndarray, output_dir: Path) -> None:
        """Save confusion matrix plot."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 10))
            # Normalize by row
            cm_norm = cm.astype(float)
            row_sums = cm_norm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = cm_norm / row_sums

            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(NUM_CLASSES))
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(NUM_CLASSES))
            ax.set_yticklabels(CLASS_NAMES, fontsize=7)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{self.backbone_name} — Confusion Matrix (normalized)")
            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(str(output_dir / "confusion_matrix.png"), dpi=150)
            plt.close(fig)
        except ImportError:
            logger.warning("  matplotlib not available — skipping confusion plot")
