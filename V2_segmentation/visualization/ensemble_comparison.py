"""
V2_segmentation/visualization/ensemble_comparison.py
=====================================================
Bar chart comparisons across ensemble stages.

Shows accuracy progression from Stage 1 through Stage 12,
highlighting where each stage adds (or loses) performance.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

DPI = 1200


class EnsembleComparison:
    """Visualize ensemble stage performance progression."""

    def __init__(self, output_dir: Path | None = None) -> None:
        from V2_segmentation.config import PLOTS_V2_DIR
        self.output_dir = output_dir or PLOTS_V2_DIR / "ensemble"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_stage_progression(
        self,
        stage_results: dict[str, dict[str, float]],
        metric: str = "accuracy",
        name: str = "ensemble_progression",
    ) -> Path:
        """Bar chart of metric across ensemble stages.

        Parameters
        ----------
        stage_results : dict[stage_name] → dict with metric values
            e.g. {"Stage 1": {"accuracy": 0.92}, "Stage 2": {"accuracy": 0.94}, ...}
        metric : which metric to plot
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            stage_names = list(stage_results.keys())
            values = [stage_results[s].get(metric, 0) for s in stage_names]

            fig, ax = plt.subplots(1, 1, figsize=(14, 6))

            # Color: green if improved, red if degraded, gray for baseline
            colors = []
            for i, v in enumerate(values):
                if i == 0:
                    colors.append("#4a90d9")
                elif v > values[i - 1]:
                    colors.append("#27ae60")
                elif v < values[i - 1]:
                    colors.append("#e74c3c")
                else:
                    colors.append("#95a5a6")

            bars = ax.bar(range(len(values)), values, color=colors, edgecolor="black", linewidth=0.5)

            # Annotate values
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                )

            # Annotate deltas
            for i in range(1, len(values)):
                delta = values[i] - values[i - 1]
                sign = "+" if delta >= 0 else ""
                ax.text(
                    i, values[i] * 0.98,
                    f"{sign}{delta:.3f}", ha="center", va="top", fontsize=6,
                    color="white", fontweight="bold",
                )

            ax.set_xticks(range(len(stage_names)))
            ax.set_xticklabels(
                [s.replace("Stage ", "S") for s in stage_names],
                rotation=45, ha="right", fontsize=8,
            )
            ax.set_ylabel(metric.capitalize(), fontsize=10)
            ax.set_title(f"Ensemble {metric.capitalize()} Progression (12 Stages)", fontsize=11)
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            path = self.output_dir / f"{name}.tiff"
            fig.savefig(str(path), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved ensemble comparison: {path}")
            return path

        except ImportError:
            logger.warning("matplotlib not available")
            return self.output_dir / f"{name}_FAILED.txt"

    def plot_per_class_comparison(
        self,
        stage_results: dict[str, dict[str, float]],
        class_names: list[str] | None = None,
        name: str = "ensemble_per_class",
    ) -> Path:
        """Grouped bar chart: per-class accuracy across selected stages."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if class_names is None:
                from V2_segmentation.config import CLASS_NAMES
                class_names = list(CLASS_NAMES)

            stages = list(stage_results.keys())
            n_stages = len(stages)
            n_classes = len(class_names)

            x = np.arange(n_classes)
            width = 0.8 / max(n_stages, 1)
            cmap = plt.get_cmap("tab20")
            colors = [cmap(i / max(n_stages, 1)) for i in range(n_stages)]

            fig, ax = plt.subplots(1, 1, figsize=(16, 6))

            for si, stage_name in enumerate(stages):
                vals = [stage_results[stage_name].get(cls, 0) for cls in class_names]
                ax.bar(x + si * width, vals, width, label=stage_name, color=colors[si])

            ax.set_xticks(x + width * n_stages / 2)
            ax.set_xticklabels(
                [c.replace("_", "\n") for c in class_names],
                fontsize=7, rotation=45, ha="right",
            )
            ax.set_ylabel("Accuracy", fontsize=9)
            ax.set_title("Per-Class Accuracy Across Ensemble Stages", fontsize=11)
            ax.legend(fontsize=6, loc="best", ncol=2)
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            path = self.output_dir / f"{name}.tiff"
            fig.savefig(str(path), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            return path

        except ImportError:
            return self.output_dir / f"{name}_FAILED.txt"
