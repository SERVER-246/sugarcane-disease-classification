"""
V2_segmentation/ensemble_v2/stage10_adversarial.py
===================================================
Stage 10: Adversarial Boosting Ensemble (AdaBoost-style).

Each round:
  1. Compute sample weights from previous round's OOF errors
  2. Hard examples get exponentially higher weight
  3. Train with weighted sampling
  4. 5–10 boosting rounds

ISOLATION: All reweighting via OOF; final weights tuned on val set.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from V2_segmentation.config import (
    BACKBONES, ENSEMBLE_V2_DIR, NUM_CLASSES, OOF_DIR, SPLIT_DIR,
)

logger = logging.getLogger(__name__)


class Stage10Adversarial:
    """AdaBoost-style adversarial boosting across backbones.

    Rather than training new models, this reweights EXISTING backbone
    predictions based on their error patterns, similar to AdaBoost.M1.

    Parameters
    ----------
    n_rounds : int
        Number of boosting rounds (default 7).
    learning_rate : float
        AdaBoost learning rate / shrinkage (default 0.5).
    """

    def __init__(
        self,
        n_rounds: int = 7,
        learning_rate: float = 0.5,
        output_dir: Path | None = None,
    ) -> None:
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.output_dir = output_dir or (ENSEMBLE_V2_DIR / "stage10_adversarial")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.round_weights: list[float] = []
        self.round_backbones: list[str] = []

    def run(self) -> dict[str, Any]:
        """Run adversarial boosting.

        Uses OOF predictions to compute per-sample weights.
        Each round selects the backbone that best covers remaining errors.
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
        from torchvision.datasets import ImageFolder
        train_ds = ImageFolder(str(SPLIT_DIR / "train"))
        y_true = np.array([s[1] for s in train_ds.samples])
        n_samples = len(y_true)

        # Initialize sample weights uniformly
        sample_weights = np.ones(n_samples) / n_samples
        round_log: list[dict[str, Any]] = []

        used_backbones: set[str] = set()

        for round_idx in range(self.n_rounds):
            # Find backbone with lowest weighted error
            best_backbone = ""
            best_error = 1.0
            best_correct = np.array([])

            for name, probs in oof_data.items():
                if name in used_backbones and len(used_backbones) < len(oof_data):
                    continue  # prefer unused backbones first

                preds = probs.argmax(axis=1)
                correct = preds == y_true
                weighted_error = float(sample_weights[~correct].sum())

                if weighted_error < best_error:
                    best_error = weighted_error
                    best_backbone = name
                    best_correct = correct

            if best_backbone == "" or best_error >= 0.5:
                logger.info(f"Boosting stopped at round {round_idx+1}: error={best_error:.4f}")
                break

            used_backbones.add(best_backbone)

            # Compute model weight (AdaBoost.M1 formula)
            alpha = self.learning_rate * 0.5 * np.log((1 - best_error) / max(best_error, 1e-10))
            self.round_weights.append(float(alpha))
            self.round_backbones.append(best_backbone)

            # Update sample weights
            sample_weights[~best_correct] *= np.exp(alpha)
            sample_weights[best_correct] *= np.exp(-alpha)
            sample_weights /= sample_weights.sum()  # normalize

            round_log.append({
                "round": round_idx + 1,
                "backbone": best_backbone,
                "weighted_error": round(best_error, 4),
                "alpha": round(alpha, 4),
                "max_sample_weight": float(sample_weights.max()),
            })

            logger.info(
                f"Boost round {round_idx+1}: {best_backbone}, "
                f"error={best_error:.4f}, alpha={alpha:.4f}"
            )

        # Save boosting config
        config = {
            "n_rounds": len(round_log),
            "backbones": self.round_backbones,
            "weights": self.round_weights,
            "rounds": round_log,
        }
        with open(self.output_dir / "boost_config.json", "w") as f:
            json.dump(config, f, indent=2)

        return {
            "status": "PASS",
            "n_rounds": len(round_log),
            "rounds": round_log,
        }

    def predict_boosted(
        self,
        backbone_probs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Apply boosting weights to backbone predictions.

        Returns weighted ensemble probabilities.
        """
        if not self.round_weights:
            raise ValueError("Boosting not trained — call run() first")

        n_samples = next(iter(backbone_probs.values())).shape[0]
        weighted_sum = np.zeros((n_samples, NUM_CLASSES), dtype=np.float64)

        for backbone, alpha in zip(self.round_backbones, self.round_weights):
            if backbone in backbone_probs:
                weighted_sum += alpha * backbone_probs[backbone]

        # Normalize
        total_alpha = sum(self.round_weights)
        if total_alpha > 0:
            weighted_sum /= total_alpha

        return weighted_sum.astype(np.float32)
