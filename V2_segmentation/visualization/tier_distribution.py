"""
V2_segmentation/visualization/tier_distribution.py
====================================================
Pseudo-label tier distribution charts.

Shows Tier A / B / C counts per class and globally.
Provides visual feedback on pseudo-label quality pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

DPI = 1200


class TierDistribution:
    """Visualize pseudo-label tier assignments."""

    def __init__(self, output_dir: Path | None = None) -> None:
        from V2_segmentation.config import PLOTS_V2_DIR
        self.output_dir = output_dir or PLOTS_V2_DIR / "tiers"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_global_pie(
        self,
        tier_counts: dict[str, int],
        name: str = "tier_global_pie",
    ) -> Path:
        """Global pie chart of tier distribution.

        Parameters
        ----------
        tier_counts : {"A": 500, "B": 300, "C": 200}
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = []
            sizes = []
            colors_map = {"A": "#27ae60", "B": "#f39c12", "C": "#e74c3c"}
            plot_colors = []

            for tier in ["A", "B", "C"]:
                count = tier_counts.get(tier, 0)
                if count > 0:
                    labels.append(f"Tier {tier} ({count})")
                    sizes.append(count)
                    plot_colors.append(colors_map.get(tier, "#95a5a6"))

            if not sizes:
                logger.warning("No tier data to plot")
                return self.output_dir / f"{name}_EMPTY.txt"

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            _wedges, _texts, _autotexts = ax.pie(
                sizes, labels=labels, autopct="%1.1f%%",
                colors=plot_colors, startangle=90,
                textprops={"fontsize": 9},
            )
            ax.set_title("Pseudo-Label Quality Tier Distribution", fontsize=12)

            plt.tight_layout()
            path = self.output_dir / f"{name}.tiff"
            fig.savefig(str(path), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved tier pie chart: {path}")
            return path

        except ImportError:
            return self.output_dir / f"{name}_FAILED.txt"

    def plot_per_class_bars(
        self,
        per_class_tiers: dict[str, dict[str, int]],
        name: str = "tier_per_class",
    ) -> Path:
        """Stacked bar chart of tiers per disease class.

        Parameters
        ----------
        per_class_tiers : dict[class_name] → {"A": n, "B": n, "C": n}
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            classes = sorted(per_class_tiers.keys())
            tier_a = [per_class_tiers[c].get("A", 0) for c in classes]
            tier_b = [per_class_tiers[c].get("B", 0) for c in classes]
            tier_c = [per_class_tiers[c].get("C", 0) for c in classes]

            x = np.arange(len(classes))
            width = 0.6

            fig, ax = plt.subplots(1, 1, figsize=(14, 6))

            ax.bar(x, tier_a, width, label="Tier A", color="#27ae60")
            ax.bar(x, tier_b, width, bottom=tier_a, label="Tier B", color="#f39c12")
            ax.bar(
                x, tier_c, width,
                bottom=[a + b for a, b in zip(tier_a, tier_b)],
                label="Tier C", color="#e74c3c",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [c.replace("_", "\n") for c in classes],
                fontsize=7, rotation=45, ha="right",
            )
            ax.set_ylabel("Number of Images", fontsize=9)
            ax.set_title("Pseudo-Label Tier Distribution by Disease Class", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            path = self.output_dir / f"{name}.tiff"
            fig.savefig(str(path), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            return path

        except ImportError:
            return self.output_dir / f"{name}_FAILED.txt"

    def plot_quality_scores_histogram(
        self,
        scores: list[float],
        tier_boundaries: tuple[float, float] = (0.80, 0.50),
        name: str = "quality_score_hist",
    ) -> Path:
        """Histogram of quality scores with tier boundary lines."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

            bins = np.linspace(0, 1, 50)
            ax.hist(scores, bins, color="#4a90d9", alpha=0.7, edgecolor="black", linewidth=0.5)

            # Tier boundaries
            ax.axvline(tier_boundaries[0], color="#27ae60", linestyle="--", linewidth=2, label=f"Tier A ≥ {tier_boundaries[0]}")
            ax.axvline(tier_boundaries[1], color="#f39c12", linestyle="--", linewidth=2, label=f"Tier B ≥ {tier_boundaries[1]}")

            ax.set_xlabel("Quality Score", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title("Pseudo-Label Quality Score Distribution", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            path = self.output_dir / f"{name}.tiff"
            fig.savefig(str(path), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            return path

        except ImportError:
            return self.output_dir / f"{name}_FAILED.txt"
