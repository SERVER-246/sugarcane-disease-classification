"""
V2_segmentation/ensemble_v2/stage8_seg_informed.py
===================================================
Stage 8: Segmentation-Informed Ensemble Weighting.

Weight each backbone's classification vote by its segmentation quality
(IoU on validation set). Backbones that produce better segmentation masks
get higher voting weight in the classification ensemble.

ISOLATION: IoU computed on val set only, weights are FIXED, applied to test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from V2_segmentation.config import (
    BACKBONES, ENSEMBLE_V2_DIR, NUM_SEG_CHANNELS,
)

logger = logging.getLogger(__name__)


def compute_mean_iou(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    num_channels: int = NUM_SEG_CHANNELS,
) -> float:
    """Compute mean IoU across channels for a set of masks.

    Parameters
    ----------
    pred_masks : (N, C, H, W) float32
    gt_masks : (N, C, H, W) float32

    Returns mean IoU across channels (excluding background).
    """
    ious = []
    for ch in range(1, num_channels):  # skip background
        pred = (pred_masks[:, ch] > 0.5).astype(np.float32)
        gt = (gt_masks[:, ch] > 0.5).astype(np.float32)
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() - intersection
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


class Stage8SegInformed:
    """Segmentation-quality-weighted ensemble.

    For each backbone:
      1. Compute mean IoU on validation set (requires val seg masks)
      2. Normalize IoU scores to weights
      3. Apply weights to classification probabilities
      4. Final prediction = weighted sum of backbone probs
    """

    def __init__(
        self,
        stage1_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.stage1_dir = stage1_dir or (ENSEMBLE_V2_DIR / "stage1_individual")
        self.output_dir = output_dir or (ENSEMBLE_V2_DIR / "stage8_seg_informed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backbone_weights: dict[str, float] = {}

    def compute_weights(
        self,
        val_seg_gt_dir: Path | None = None,
    ) -> dict[str, float]:
        """Compute per-backbone weights from val set IoU.

        If ground truth seg masks aren't available, falls back to
        confidence-based scoring (mean per-pixel confidence from seg output).
        """
        weights: dict[str, float] = {}

        for backbone in BACKBONES:
            seg_path = self.stage1_dir / f"{backbone}_val_seg_masks.npy"
            if not seg_path.exists():
                logger.warning(f"No val seg masks for {backbone}")
                weights[backbone] = 1.0  # default equal weight
                continue

            seg_masks = np.load(str(seg_path))  # (N, C, H, W)

            if val_seg_gt_dir is not None:
                # Compute IoU against ground truth
                gt_path = val_seg_gt_dir / f"val_seg_gt.npy"
                if gt_path.exists():
                    gt = np.load(str(gt_path))
                    iou = compute_mean_iou(seg_masks, gt)
                    weights[backbone] = max(iou, 0.1)  # floor at 0.1
                    continue

            # Fallback: use prediction confidence as proxy for quality
            confidence = seg_masks.max(axis=1).mean()
            # Penalize low-confidence predictions
            weights[backbone] = float(np.clip(confidence, 0.1, 1.0))

        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        self.backbone_weights = weights
        logger.info(f"Stage 8 backbone weights: {weights}")
        return weights

    def apply_weights(self, split: str = "test") -> dict[str, Any]:
        """Apply seg-informed weights to classification predictions.

        Returns weighted ensemble predictions and accuracy.
        """
        if not self.backbone_weights:
            self.compute_weights()

        all_probs = []
        all_weights = []
        labels = None

        for backbone in BACKBONES:
            w = self.backbone_weights.get(backbone, 1.0 / len(BACKBONES))
            probs_path = self.stage1_dir / f"{backbone}_{split}_cls_probs.npy"
            if not probs_path.exists():
                continue

            probs = np.load(str(probs_path))
            all_probs.append(probs * w)
            all_weights.append(w)

            if labels is None:
                label_path = self.stage1_dir / f"{backbone}_{split}_labels.npy"
                if label_path.exists():
                    labels = np.load(str(label_path))

        if not all_probs:
            return {"status": "FAIL", "reason": "No predictions found"}

        # Weighted sum
        ensemble_probs = np.sum(all_probs, axis=0)
        preds = ensemble_probs.argmax(axis=1)
        accuracy = float((preds == labels).mean()) if labels is not None else 0.0

        # Save
        np.save(str(self.output_dir / f"seg_weighted_{split}.npy"), ensemble_probs)

        result = {
            "status": "PASS",
            "split": split,
            "accuracy": round(accuracy, 4),
            "n_backbones_used": len(all_probs),
            "weights": {k: round(v, 4) for k, v in self.backbone_weights.items()},
        }
        logger.info(f"Stage 8 [{split}]: seg-weighted acc={accuracy:.4f}")
        return result

    def run(self) -> dict[str, Any]:
        """Run full Stage 8 pipeline."""
        self.compute_weights()
        val_result = self.apply_weights("val")
        test_result = self.apply_weights("test")
        return {
            "val": val_result,
            "test": test_result,
            "weights": self.backbone_weights,
        }
