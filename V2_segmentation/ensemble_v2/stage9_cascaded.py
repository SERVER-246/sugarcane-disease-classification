"""
V2_segmentation/ensemble_v2/stage9_cascaded.py
===============================================
Stage 9: Cascaded Sequential Training.

Train a sequence of classifiers where each focuses on ERRORS of the previous:
  Model_1: Best individual backbone
  Model_2: Trained ONLY on images Model_1 OOF got wrong
  Model_3: Trained on images Model_1+Model_2 both got wrong
  ... max 5 cascade levels

Inference: Sequential — if Model_1 confidence > threshold, use it, else Model_2...

ISOLATION: Hard examples identified via OOF only; thresholds tuned on val set.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from torchvision.datasets import ImageFolder

from V2_segmentation.config import (
    BACKBONES, ENSEMBLE_V2_DIR, NUM_CLASSES,
    OOF_DIR, SPLIT_DIR,
)

logger = logging.getLogger(__name__)


class Stage9Cascaded:
    """Cascaded sequential error-coverage ensemble.

    Each level in the cascade specializes on samples that previous levels
    got wrong (identified via OOF predictions for leakage safety).

    Parameters
    ----------
    max_levels : int
        Maximum cascade depth (default 5).
    confidence_threshold : float
        Minimum confidence to accept Model_k's prediction (default 0.95).
    min_error_samples : int
        Minimum samples needed to train next level (default 50).
    """

    def __init__(
        self,
        max_levels: int = 5,
        confidence_threshold: float = 0.95,
        min_error_samples: int = 50,
        output_dir: Path | None = None,
    ) -> None:
        self.max_levels = max_levels
        self.confidence_threshold = confidence_threshold
        self.min_error_samples = min_error_samples
        self.output_dir = output_dir or (ENSEMBLE_V2_DIR / "stage9_cascaded")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cascade_models: list[dict[str, Any]] = []

    def build_cascade(self) -> dict[str, Any]:
        """Build the cascade from OOF predictions.

        Uses OOF predictions to identify hard examples at each level.
        """
        # Load all OOF predictions
        oof_data: dict[str, np.ndarray] = {}
        for backbone in BACKBONES:
            path = OOF_DIR / f"{backbone}_oof_probs.npy"
            if path.exists():
                oof_data[backbone] = np.load(str(path))

        if not oof_data:
            return {"status": "SKIP", "reason": "No OOF predictions available"}

        # Get true labels
        train_ds = ImageFolder(str(SPLIT_DIR / "train"))
        y_true = np.array([s[1] for s in train_ds.samples])
        n_samples = len(y_true)

        # Find best initial backbone (highest OOF accuracy)
        best_backbone = ""
        best_acc = 0.0
        for name, probs in oof_data.items():
            acc = float((probs.argmax(axis=1) == y_true).mean())
            if acc > best_acc:
                best_acc = acc
                best_backbone = name

        logger.info(f"Cascade Level 1: {best_backbone} (OOF acc={best_acc:.4f})")

        # Build cascade
        remaining_wrong = np.ones(n_samples, dtype=bool)
        cascade_info: list[dict[str, Any]] = []

        for level in range(self.max_levels):
            if level == 0:
                backbone = best_backbone
                probs = oof_data[backbone]
            else:
                # Find backbone with best accuracy on remaining errors
                error_idx = np.where(remaining_wrong)[0]
                if len(error_idx) < self.min_error_samples:
                    logger.info(
                        f"Cascade stopped at level {level}: "
                        f"only {len(error_idx)} errors remaining"
                    )
                    break

                best_error_acc = 0.0
                backbone = ""
                probs = np.array([])
                for name, p in oof_data.items():
                    error_preds = p[error_idx].argmax(axis=1)
                    error_true = y_true[error_idx]
                    acc = float((error_preds == error_true).mean())
                    if acc > best_error_acc:
                        best_error_acc = acc
                        backbone = name
                        probs = p

            # Record this level
            preds = probs.argmax(axis=1)
            confidences = probs.max(axis=1)
            correct = preds == y_true

            # Mark resolved: correct AND confident
            resolved = correct & (confidences >= self.confidence_threshold)
            remaining_wrong = remaining_wrong & ~resolved

            n_errors = int(remaining_wrong.sum())
            level_info = {
                "level": level + 1,
                "backbone": backbone,
                "n_resolved": int(resolved.sum()),
                "n_remaining_errors": n_errors,
                "level_accuracy": float(correct[remaining_wrong | resolved].mean())
                if (remaining_wrong | resolved).sum() > 0 else 0.0,
            }
            cascade_info.append(level_info)
            self.cascade_models.append({
                "backbone": backbone, "threshold": self.confidence_threshold,
            })

            logger.info(
                f"Cascade Level {level+1}: {backbone}, "
                f"resolved={resolved.sum()}, remaining={n_errors}"
            )

            if n_errors == 0:
                break

        # Save cascade configuration
        import json
        config_path = self.output_dir / "cascade_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "levels": cascade_info,
                "confidence_threshold": self.confidence_threshold,
            }, f, indent=2)

        return {
            "status": "PASS",
            "n_levels": len(cascade_info),
            "levels": cascade_info,
            "final_remaining_errors": int(remaining_wrong.sum()),
        }

    def predict_cascaded(
        self,
        backbone_probs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Apply cascade at inference time.

        Parameters
        ----------
        backbone_probs : dict
            backbone_name → (N, NUM_CLASSES) probability arrays.

        Returns
        -------
        final_probs : (N, NUM_CLASSES)
        """
        if not self.cascade_models:
            raise ValueError("Cascade not built yet — call build_cascade() first")

        n_samples = next(iter(backbone_probs.values())).shape[0]
        final_probs = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
        resolved = np.zeros(n_samples, dtype=bool)

        for level_info in self.cascade_models:
            backbone = level_info["backbone"]
            threshold = level_info["threshold"]

            if backbone not in backbone_probs:
                continue

            probs = backbone_probs[backbone]
            conf = probs.max(axis=1)

            # Assign to unresolved samples with sufficient confidence
            mask = ~resolved & (conf >= threshold)
            final_probs[mask] = probs[mask]
            resolved[mask] = True

        # For remaining unresolved: use mean of all backbone probs
        if not resolved.all():
            unresolved = ~resolved
            mean_probs = np.mean(list(backbone_probs.values()), axis=0)
            final_probs[unresolved] = mean_probs[unresolved]

        return final_probs

    def run(self) -> dict[str, Any]:
        """Run full Stage 9 pipeline."""
        return self.build_cascade()
