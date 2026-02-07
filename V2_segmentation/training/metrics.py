"""
V2_segmentation/training/metrics.py
====================================
Evaluation metrics for V2 dual-head models:
  - Classification: accuracy, per-class F1, confusion matrix
  - Segmentation: per-channel IoU, Dice coefficient, mean IoU
  - Combined tracking for Phase A/B/C evaluation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from V2_segmentation.config import CLASS_NAMES, NUM_CLASSES, NUM_SEG_CHANNELS, SEG_CHANNELS

logger = logging.getLogger(__name__)


# ============================================================================
#  Classification metrics
# ============================================================================

def classification_accuracy(
    logits: torch.Tensor, labels: torch.Tensor
) -> float:
    """Compute top-1 accuracy from raw logits and integer labels."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def per_class_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-class precision, recall, F1 from predicted and true labels.

    Parameters
    ----------
    all_preds : (N,) int array of predicted class indices
    all_labels : (N,) int array of true class indices
    class_names : list of class names (optional)

    Returns
    -------
    dict mapping class_name → {precision, recall, f1, support}
    """
    names = class_names or CLASS_NAMES
    metrics: dict[str, dict[str, float]] = {}

    for idx, name in enumerate(names):
        tp = int(((all_preds == idx) & (all_labels == idx)).sum())
        fp = int(((all_preds == idx) & (all_labels != idx)).sum())
        fn = int(((all_preds != idx) & (all_labels == idx)).sum())
        support = int((all_labels == idx).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return metrics


def confusion_matrix(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Compute confusion matrix (true rows × predicted cols)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(all_labels, all_preds):
        cm[true, pred] += 1
    return cm


# ============================================================================
#  Segmentation metrics
# ============================================================================

def iou_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute IoU per channel for multi-label segmentation.

    Parameters
    ----------
    pred : (B, C, H, W) — raw logits or probabilities
    target : (B, C, H, W) — binary ground truth
    threshold : float — threshold for converting logits → binary
    smooth : float — numerical stability

    Returns
    -------
    (C,) tensor of per-channel IoU values
    """
    # Binarize predictions
    if pred.requires_grad:
        pred = pred.detach()
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    target_bin = (target >= threshold).float()

    # Flatten spatial dims
    pred_flat = pred_bin.reshape(pred_bin.shape[0], pred_bin.shape[1], -1)  # (B, C, H*W)
    target_flat = target_bin.reshape(target_bin.shape[0], target_bin.shape[1], -1)

    intersection = (pred_flat * target_flat).sum(dim=2).sum(dim=0)  # (C,)
    union = (pred_flat + target_flat).clamp(max=1.0).sum(dim=2).sum(dim=0)

    iou = (intersection + smooth) / (union + smooth)
    return iou


def dice_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute Dice coefficient per channel.

    Parameters
    ----------
    pred : (B, C, H, W) logits or probabilities
    target : (B, C, H, W) binary masks

    Returns
    -------
    (C,) tensor of per-channel Dice values
    """
    if pred.requires_grad:
        pred = pred.detach()
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    target_bin = (target >= threshold).float()

    pred_flat = pred_bin.reshape(pred_bin.shape[0], pred_bin.shape[1], -1)
    target_flat = target_bin.reshape(target_bin.shape[0], target_bin.shape[1], -1)

    intersection = (pred_flat * target_flat).sum(dim=2).sum(dim=0)
    cardinality = pred_flat.sum(dim=2).sum(dim=0) + target_flat.sum(dim=2).sum(dim=0)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice


def mean_iou(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> float:
    """Compute mean IoU across all segmentation channels."""
    ious = iou_per_channel(pred, target, threshold)
    return ious.mean().item()


def mean_dice(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> float:
    """Compute mean Dice across all segmentation channels."""
    dices = dice_per_channel(pred, target, threshold)
    return dices.mean().item()


# ============================================================================
#  Metric tracker (accumulates over batches)
# ============================================================================

class MetricTracker:
    """Accumulate and summarize metrics over an epoch.

    Usage
    -----
    >>> tracker = MetricTracker()
    >>> for batch in loader:
    ...     tracker.update(cls_logits, cls_labels, seg_logits, seg_masks, loss_dict)
    >>> summary = tracker.compute()
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators."""
        self._cls_correct = 0
        self._cls_total = 0
        self._all_preds: list[np.ndarray] = []
        self._all_labels: list[np.ndarray] = []
        self._seg_iou_sum = np.zeros(NUM_SEG_CHANNELS, dtype=np.float64)
        self._seg_dice_sum = np.zeros(NUM_SEG_CHANNELS, dtype=np.float64)
        self._seg_batches = 0
        self._loss_sums: dict[str, float] = {}
        self._loss_counts: dict[str, int] = {}

    @torch.no_grad()
    def update(
        self,
        cls_logits: torch.Tensor | None = None,
        cls_labels: torch.Tensor | None = None,
        seg_logits: torch.Tensor | None = None,
        seg_targets: torch.Tensor | None = None,
        loss_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Accumulate metrics from one batch."""
        # Classification metrics
        if cls_logits is not None and cls_labels is not None:
            preds = cls_logits.argmax(dim=1).cpu().numpy()
            labels = cls_labels.cpu().numpy()
            self._cls_correct += int((preds == labels).sum())
            self._cls_total += len(labels)
            self._all_preds.append(preds)
            self._all_labels.append(labels)

        # Segmentation metrics
        if seg_logits is not None and seg_targets is not None:
            ious = iou_per_channel(seg_logits, seg_targets).cpu().numpy()
            dices = dice_per_channel(seg_logits, seg_targets).cpu().numpy()
            self._seg_iou_sum += ious
            self._seg_dice_sum += dices
            self._seg_batches += 1

        # Loss tracking
        if loss_dict is not None:
            for key, val in loss_dict.items():
                v = val.item() if isinstance(val, torch.Tensor) else float(val)
                self._loss_sums[key] = self._loss_sums.get(key, 0.0) + v
                self._loss_counts[key] = self._loss_counts.get(key, 0) + 1

    def compute(self) -> dict[str, Any]:
        """Compute summary metrics for the epoch.

        Returns
        -------
        dict with keys like:
            cls_accuracy, per_class_f1, mean_iou, per_channel_iou,
            mean_dice, per_channel_dice, avg_loss, avg_loss_seg, etc.
        """
        result: dict[str, Any] = {}

        # Classification
        if self._cls_total > 0:
            result["cls_accuracy"] = self._cls_correct / self._cls_total
            all_preds = np.concatenate(self._all_preds)
            all_labels = np.concatenate(self._all_labels)
            pcm = per_class_metrics(all_preds, all_labels)
            result["per_class_f1"] = {k: v["f1"] for k, v in pcm.items()}
            result["macro_f1"] = float(np.mean([v["f1"] for v in pcm.values()]))
            result["confusion_matrix"] = confusion_matrix(all_preds, all_labels)

        # Segmentation
        if self._seg_batches > 0:
            per_ch_iou = self._seg_iou_sum / self._seg_batches
            per_ch_dice = self._seg_dice_sum / self._seg_batches
            result["mean_iou"] = float(per_ch_iou.mean())
            result["per_channel_iou"] = {
                SEG_CHANNELS[i]: float(per_ch_iou[i])
                for i in range(NUM_SEG_CHANNELS)
            }
            result["mean_dice"] = float(per_ch_dice.mean())
            result["per_channel_dice"] = {
                SEG_CHANNELS[i]: float(per_ch_dice[i])
                for i in range(NUM_SEG_CHANNELS)
            }

        # Losses
        for key, total in self._loss_sums.items():
            count = self._loss_counts[key]
            result[f"avg_{key}"] = total / count if count > 0 else 0.0

        return result

    def log_summary(self, prefix: str = "  ") -> None:
        """Log a human-readable summary."""
        summary = self.compute()

        parts = []
        if "cls_accuracy" in summary:
            parts.append(f"acc={summary['cls_accuracy']:.4f}")
        if "macro_f1" in summary:
            parts.append(f"F1={summary['macro_f1']:.4f}")
        if "mean_iou" in summary:
            parts.append(f"mIoU={summary['mean_iou']:.4f}")
        if "mean_dice" in summary:
            parts.append(f"Dice={summary['mean_dice']:.4f}")
        if "avg_loss" in summary:
            parts.append(f"loss={summary['avg_loss']:.4f}")

        logger.info(f"{prefix}{' | '.join(parts)}")
