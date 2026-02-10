"""
V2_segmentation/evaluation/overfit_detector.py
===============================================
Automated overfitting detection during training and ensemble stages.

Per Section 8.3:
  - train_acc - val_acc gap > 5% → WARNING
  - train_acc - val_acc gap > 10% → HALT

Also tracks per-class overfit and per-fold variance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from V2_segmentation.config import EVAL_DIR

logger = logging.getLogger(__name__)


class OverfitDetector:
    """Monitor train-val gaps for overfitting signals.

    Tracks metrics across training phases and ensemble stages.
    Records all observations for audit trail.
    """

    WARNING_THRESHOLD = 0.05   # 5% gap
    HALT_THRESHOLD = 0.10      # 10% gap

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or EVAL_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.observations: list[dict[str, Any]] = []

    def check(
        self,
        train_metric: float,
        val_metric: float,
        context: str = "",
        metric_name: str = "accuracy",
    ) -> dict[str, Any]:
        """Check a single train-val pair for overfitting.

        Parameters
        ----------
        train_metric : float
            Training accuracy (or other metric).
        val_metric : float
            Validation accuracy (or other metric).
        context : str
            Description (e.g. "CustomCoAtNet Phase B epoch 15").
        metric_name : str
            Name of the metric being compared.

        Returns
        -------
        dict with status (PASS/WARNING/HALT), gap, context.
        """
        gap = train_metric - val_metric

        if gap > self.HALT_THRESHOLD:
            status = "HALT"
            logger.error(
                f"OVERFIT HALT: {context} — "
                f"train_{metric_name}={train_metric:.4f}, "
                f"val_{metric_name}={val_metric:.4f}, gap={gap:.4f}"
            )
        elif gap > self.WARNING_THRESHOLD:
            status = "WARNING"
            logger.warning(
                f"Overfit warning: {context} — "
                f"train_{metric_name}={train_metric:.4f}, "
                f"val_{metric_name}={val_metric:.4f}, gap={gap:.4f}"
            )
        else:
            status = "PASS"

        result = {
            "status": status,
            "train": round(train_metric, 5),
            "val": round(val_metric, 5),
            "gap": round(gap, 5),
            "metric": metric_name,
            "context": context,
        }
        self.observations.append(result)
        return result

    def check_per_class(
        self,
        train_per_class: dict[str, float],
        val_per_class: dict[str, float],
        context: str = "",
    ) -> dict[str, Any]:
        """Check per-class train-val gaps.

        Returns dict with per-class status and worst offender.
        """
        per_class: dict[str, dict[str, Any]] = {}
        worst_gap = 0.0
        worst_class = ""

        for cls_name in train_per_class:
            train_v = train_per_class.get(cls_name, 0.0)
            val_v = val_per_class.get(cls_name, 0.0)
            gap = train_v - val_v

            if gap > worst_gap:
                worst_gap = gap
                worst_class = cls_name

            per_class[cls_name] = {
                "train": round(train_v, 4),
                "val": round(val_v, 4),
                "gap": round(gap, 4),
                "status": (
                    "HALT" if gap > self.HALT_THRESHOLD
                    else "WARNING" if gap > self.WARNING_THRESHOLD
                    else "PASS"
                ),
            }

        result = {
            "context": context,
            "worst_class": worst_class,
            "worst_gap": round(worst_gap, 4),
            "per_class": per_class,
            "overall_status": (
                "HALT" if worst_gap > self.HALT_THRESHOLD
                else "WARNING" if worst_gap > self.WARNING_THRESHOLD
                else "PASS"
            ),
        }
        self.observations.append(result)
        return result

    def check_kfold_variance(
        self,
        fold_metrics: list[float],
        context: str = "",
        max_std: float = 0.03,
    ) -> dict[str, Any]:
        """Check K-fold variance for instability.

        High variance across folds suggests overfitting to specific data subsets.

        Parameters
        ----------
        fold_metrics : list[float]
            Per-fold metric values.
        max_std : float
            Maximum acceptable standard deviation (default 3%).
        """
        mean = float(np.mean(fold_metrics))
        std = float(np.std(fold_metrics))
        spread = float(np.max(fold_metrics) - np.min(fold_metrics))

        status = "PASS" if std <= max_std else "WARNING"
        if std > 2 * max_std:
            status = "HALT"

        result = {
            "status": status,
            "context": context,
            "n_folds": len(fold_metrics),
            "fold_metrics": [round(m, 4) for m in fold_metrics],
            "mean": round(mean, 4),
            "std": round(std, 4),
            "spread": round(spread, 4),
            "max_allowed_std": max_std,
        }
        self.observations.append(result)
        return result

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all overfit observations."""
        n_halt = sum(1 for o in self.observations if o.get("status") == "HALT")
        n_warn = sum(1 for o in self.observations if o.get("status") == "WARNING")
        n_pass = sum(1 for o in self.observations if o.get("status") == "PASS")

        return {
            "total_checks": len(self.observations),
            "pass": n_pass,
            "warnings": n_warn,
            "halts": n_halt,
            "overall_status": (
                "HALT" if n_halt > 0 else "WARNING" if n_warn > 0 else "PASS"
            ),
        }

    def save(self) -> Path:
        """Save all observations to JSON."""
        path = self.output_dir / "overfit_report.json"
        with open(path, "w") as f:
            json.dump({
                "summary": self.get_summary(),
                "observations": self.observations,
            }, f, indent=2)
        logger.info(f"Overfit report saved to {path}")
        return path
