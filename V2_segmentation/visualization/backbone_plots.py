"""
V2_segmentation/visualization/backbone_plots.py
=================================================
Per-backbone publication-quality plots matching V1 output:
  - Confusion matrix (disease labels on axes) — TIFF 1200 DPI
  - ROC curves (one-vs-rest per class with AUC) — TIFF 1200 DPI
  - Per-class metrics bar chart (precision, recall, F1) — TIFF 1200 DPI

These are the exact equivalents of V1's:
  plots_metrics/{backbone}_confusion_matrix.tiff
  plots_metrics/{backbone}_roc_curves.tiff
  plots_metrics/{backbone}_per_class_metrics.tiff
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DPI = 1200


class BackbonePlots:
    """Generate per-backbone evaluation plots (V1-matching format)."""

    def __init__(self, output_dir: Path | None = None) -> None:
        from V2_segmentation.config import PLOTS_V2_DIR
        self.output_dir = output_dir or PLOTS_V2_DIR / "backbone_eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    #  Confusion Matrix
    # ================================================================

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        backbone_name: str,
        class_names: list[str] | None = None,
    ) -> Path:
        """Plot labeled confusion matrix as TIFF.

        Parameters
        ----------
        cm : (N, N) int array — confusion matrix (true rows × pred cols)
        backbone_name : used in title and filename
        class_names : list of disease names for axis labels
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if class_names is None:
                from V2_segmentation.config import CLASS_NAMES
                class_names = list(CLASS_NAMES)

            n = len(class_names)
            fig, ax = plt.subplots(1, 1, figsize=(max(10, n * 0.9), max(8, n * 0.8)))

            # Normalize for color (row-wise %)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(
                cm.astype(np.float64), row_sums,
                out=np.zeros_like(cm, dtype=np.float64),
                where=row_sums != 0,
            )

            im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Annotate cells with counts
            thresh = cm_norm.max() / 2.0
            for i in range(n):
                for j in range(n):
                    color = "white" if cm_norm[i, j] > thresh else "black"
                    ax.text(
                        j, i, f"{cm[i, j]}",
                        ha="center", va="center", fontsize=7, color=color,
                    )

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(
                [c.replace("_", "\n") for c in class_names],
                rotation=45, ha="right", fontsize=7,
            )
            ax.set_yticklabels(class_names, fontsize=7)
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("True", fontsize=10)
            ax.set_title(f"Confusion Matrix: {backbone_name}", fontsize=12)

            plt.tight_layout()
            path = self.output_dir / f"{backbone_name}_confusion_matrix.tiff"
            fig.savefig(str(path), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved confusion matrix: {path.name}")
            return path

        except ImportError:
            logger.warning("matplotlib not available — skipping confusion matrix")
            return self.output_dir / f"{backbone_name}_confusion_matrix_FAILED.txt"

    # ================================================================
    #  ROC Curves (One-vs-Rest)
    # ================================================================

    def plot_roc_curves(
        self,
        all_labels: np.ndarray,
        all_probs: np.ndarray,
        backbone_name: str,
        class_names: list[str] | None = None,
    ) -> Path:
        """Plot per-class ROC curves with AUC.

        Parameters
        ----------
        all_labels : (N,) int array — true class indices
        all_probs : (N, C) float array — predicted probabilities per class
        backbone_name : used in title and filename
        class_names : list of disease names
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc

            if class_names is None:
                from V2_segmentation.config import CLASS_NAMES
                class_names = list(CLASS_NAMES)

            n_classes = len(class_names)
            labels_bin = np.asarray(label_binarize(all_labels, classes=list(range(n_classes))))

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            cmap = plt.get_cmap("tab20")
            colors = [cmap(i / max(n_classes, 1)) for i in range(n_classes)]

            all_auc = []
            for i, (name, color) in enumerate(zip(class_names, colors)):
                if labels_bin[:, i].sum() == 0:
                    continue  # Skip classes with no positive samples
                fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
                roc_auc = auc(fpr, tpr)
                all_auc.append(roc_auc)
                short_name = name.replace("_", " ")
                ax.plot(
                    fpr, tpr, color=color, linewidth=1.2,
                    label=f"{short_name} (AUC={roc_auc:.3f})",
                )

            # Diagonal reference line
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)

            macro_auc = float(np.mean(all_auc)) if all_auc else 0.0
            ax.set_xlabel("False Positive Rate", fontsize=10)
            ax.set_ylabel("True Positive Rate", fontsize=10)
            ax.set_title(
                f"ROC Curves: {backbone_name}\n(Macro AUC = {macro_auc:.4f})",
                fontsize=12,
            )
            ax.legend(fontsize=6, loc="lower right", ncol=2)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            path = self.output_dir / f"{backbone_name}_roc_curves.tiff"
            fig.savefig(str(path), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved ROC curves: {path.name}")
            return path

        except ImportError as e:
            logger.warning(f"Missing dependency for ROC — {e}")
            return self.output_dir / f"{backbone_name}_roc_curves_FAILED.txt"

    # ================================================================
    #  Per-Class Metrics Bar Chart (Precision / Recall / F1)
    # ================================================================

    def plot_per_class_metrics(
        self,
        per_class: dict[str, dict[str, float]],
        backbone_name: str,
    ) -> Path:
        """Plot grouped bar chart of precision, recall, F1 per disease class.

        Parameters
        ----------
        per_class : dict[class_name] → {precision, recall, f1, support}
            Output from metrics.per_class_metrics().
        backbone_name : used in title and filename
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            names = list(per_class.keys())
            n = len(names)
            precisions = [per_class[c]["precision"] for c in names]
            recalls = [per_class[c]["recall"] for c in names]
            f1s = [per_class[c]["f1"] for c in names]
            supports = [per_class[c].get("support", 0) for c in names]

            x = np.arange(n)
            width = 0.25

            fig, ax = plt.subplots(1, 1, figsize=(max(12, n * 1.0), 6))
            ax.bar(x - width, precisions, width, label="Precision", color="#3498db")
            ax.bar(x, recalls, width, label="Recall", color="#2ecc71")
            b3 = ax.bar(x + width, f1s, width, label="F1-Score", color="#e74c3c")

            # Annotate F1 values on top
            for bar, val in zip(b3, f1s):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6,
                )

            # Support count below x-axis
            for i, s in enumerate(supports):
                ax.text(x[i], -0.05, f"n={s}", ha="center", fontsize=5, color="gray")

            ax.set_xticks(x)
            ax.set_xticklabels(
                [c.replace("_", "\n") for c in names],
                fontsize=7, rotation=45, ha="right",
            )
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_title(f"Per-Class Metrics: {backbone_name}", fontsize=12)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            path = self.output_dir / f"{backbone_name}_per_class_metrics.tiff"
            fig.savefig(str(path), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved per-class metrics: {path.name}")
            return path

        except ImportError:
            logger.warning("matplotlib not available — skipping per-class metrics")
            return self.output_dir / f"{backbone_name}_per_class_metrics_FAILED.txt"

    # ================================================================
    #  Convenience: generate all 3 plots at once
    # ================================================================

    def plot_all(
        self,
        backbone_name: str,
        confusion_matrix: np.ndarray | None = None,
        all_labels: np.ndarray | None = None,
        all_probs: np.ndarray | None = None,
        per_class: dict[str, dict[str, float]] | None = None,
        class_names: list[str] | None = None,
    ) -> dict[str, Path]:
        """Generate all available plots for a backbone.

        Parameters
        ----------
        confusion_matrix : (N, N) int array
        all_labels : (N,) int labels (needed for ROC)
        all_probs : (N, C) probabilities (needed for ROC)
        per_class : per_class_metrics() output (needed for bar chart)

        Returns
        -------
        dict mapping plot name → saved path
        """
        paths: dict[str, Path] = {}

        if confusion_matrix is not None:
            paths["confusion_matrix"] = self.plot_confusion_matrix(
                confusion_matrix, backbone_name, class_names,
            )

        if all_labels is not None and all_probs is not None:
            paths["roc_curves"] = self.plot_roc_curves(
                all_labels, all_probs, backbone_name, class_names,
            )

        if per_class is not None:
            paths["per_class_metrics"] = self.plot_per_class_metrics(
                per_class, backbone_name,
            )

        return paths

    # ================================================================
    #  Load from checkpoint + re-plot
    # ================================================================

    def plot_from_saved_eval(
        self,
        backbone_name: str,
        eval_dir: Path | None = None,
    ) -> dict[str, Path]:
        """Re-generate plots from saved evaluation artifacts.

        Looks for:
          {eval_dir}/{backbone}_eval.npz  — all_labels, all_probs, confusion_matrix
          {eval_dir}/{backbone}_per_class.json — per_class metrics
        """
        from V2_segmentation.config import CKPT_V2_DIR
        d = eval_dir or CKPT_V2_DIR
        paths: dict[str, Path] = {}

        # Load npz if available
        npz_path = d / f"{backbone_name}_eval.npz"
        if npz_path.exists():
            data = np.load(str(npz_path), allow_pickle=True)
            cm = data.get("confusion_matrix")
            labels = data.get("all_labels")
            probs = data.get("all_probs")

            if cm is not None:
                paths["confusion_matrix"] = self.plot_confusion_matrix(
                    cm, backbone_name,
                )
            if labels is not None and probs is not None:
                paths["roc_curves"] = self.plot_roc_curves(
                    labels, probs, backbone_name,
                )

        # Load per-class JSON if available
        import json
        json_path = d / f"{backbone_name}_per_class.json"
        if json_path.exists():
            with open(json_path) as f:
                per_class = json.load(f)
            paths["per_class_metrics"] = self.plot_per_class_metrics(
                per_class, backbone_name,
            )

        return paths
