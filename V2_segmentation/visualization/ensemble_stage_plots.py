"""
V2_segmentation/visualization/ensemble_stage_plots.py
======================================================
Per-ensemble-stage evaluation plots:
  - Confusion matrix (labeled) — TIFF 1200 DPI
  - ROC curves (one-vs-rest per class) — TIFF 1200 DPI
  - Per-class metrics bar chart — TIFF 1200 DPI

These match V1's ensemble/{stage}/..._confusion_matrix.tiff, etc.
Reuses BackbonePlots internally (same format, different naming).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EnsembleStagePlots:
    """Generate per-stage evaluation plots for ensemble pipeline."""

    def __init__(self, output_dir: Path | None = None) -> None:
        from V2_segmentation.config import PLOTS_V2_DIR
        self.output_dir = output_dir or PLOTS_V2_DIR / "ensemble_eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_stage(
        self,
        stage_name: str,
        all_labels: np.ndarray | None = None,
        all_probs: np.ndarray | None = None,
        all_preds: np.ndarray | None = None,
        class_names: list[str] | None = None,
    ) -> dict[str, Path]:
        """Generate confusion matrix, ROC, and per-class plots for one stage.

        Parameters
        ----------
        stage_name : e.g. "stage2_soft_voting", "stage8_seg_informed"
        all_labels : (N,) int true labels
        all_probs : (N, C) float probabilities (for ROC)
        all_preds : (N,) int predicted labels (for confusion matrix);
                    if None, argmax(all_probs) is used
        class_names : disease class names

        Returns
        -------
        dict mapping plot type → path
        """
        from V2_segmentation.visualization.backbone_plots import BackbonePlots
        from V2_segmentation.training.metrics import per_class_metrics, confusion_matrix

        if class_names is None:
            from V2_segmentation.config import CLASS_NAMES
            class_names = list(CLASS_NAMES)

        # Derive preds from probs if not provided
        if all_preds is None and all_probs is not None:
            all_preds = np.argmax(all_probs, axis=1)

        stage_dir = self.output_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        plotter = BackbonePlots(output_dir=stage_dir)

        paths: dict[str, Path] = {}

        if all_preds is not None and all_labels is not None:
            # Confusion matrix
            cm = confusion_matrix(all_preds, all_labels, num_classes=len(class_names))
            paths["confusion_matrix"] = plotter.plot_confusion_matrix(
                cm, stage_name, class_names,
            )

            # Per-class metrics
            pcm = per_class_metrics(all_preds, all_labels, class_names)
            paths["per_class_metrics"] = plotter.plot_per_class_metrics(
                pcm, stage_name,
            )

        if all_labels is not None and all_probs is not None:
            # ROC curves
            paths["roc_curves"] = plotter.plot_roc_curves(
                all_labels, all_probs, stage_name, class_names,
            )

        logger.info(f"Generated {len(paths)} plots for {stage_name}")
        return paths

    def plot_from_saved_eval(
        self,
        npz_path: str | Path,
        stage_name: str | None = None,
    ) -> dict[str, Path]:
        """Re-generate stage plots from a saved _eval.npz file.

        Parameters
        ----------
        npz_path : path to a .npz containing 'all_labels', 'all_probs',
                   and optionally 'confusion_matrix'.
        stage_name : override name; defaults to stem of npz_path.
        """
        npz_path = Path(npz_path)
        if stage_name is None:
            stage_name = npz_path.stem.replace("_eval", "")

        data = np.load(str(npz_path), allow_pickle=True)
        all_labels = data.get("all_labels")
        all_probs = data.get("all_probs")

        return self.plot_stage(
            stage_name=stage_name,
            all_labels=all_labels,
            all_probs=all_probs,
        )
