"""
V2_segmentation/visualization/validation_demo.py
==================================================
Before/after visualization of the validation gate.

Shows which images pass/fail the seg-based validation,
with overlay annotations showing the reason.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DPI = 1200


class ValidationDemo:
    """Visualize validation gate decisions."""

    def __init__(self, output_dir: Path | None = None) -> None:
        from V2_segmentation.config import PLOTS_V2_DIR
        self.output_dir = output_dir or PLOTS_V2_DIR / "validation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_accepted_vs_rejected(
        self,
        results: list[dict[str, Any]],
        image_paths: list[str | Path],
        max_per_group: int = 8,
        name: str = "validation_demo",
    ) -> Path:
        """Show accepted and rejected images side by side.

        Parameters
        ----------
        results : list of dicts from SegValidator.validate()
        image_paths : corresponding image paths
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            accepted = [(r, p) for r, p in zip(results, image_paths) if r["accepted"]]
            rejected = [(r, p) for r, p in zip(results, image_paths) if not r["accepted"]]

            accepted = accepted[:max_per_group]
            rejected = rejected[:max_per_group]

            cols = max(len(accepted), len(rejected), 1)
            fig, axes = plt.subplots(2, cols, figsize=(cols * 3, 6), squeeze=False)

            # Row 0: Accepted
            for i, (res, path) in enumerate(accepted):
                img = cv2.imread(str(path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    axes[0][i].imshow(img)
                axes[0][i].set_title(
                    f"✓ PR={res['plant_ratio']:.2f}",
                    fontsize=7, color="green",
                )
                axes[0][i].axis("off")

            # Row 1: Rejected
            for i, (res, path) in enumerate(rejected):
                img = cv2.imread(str(path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    axes[1][i].imshow(img)
                reason = res.get("reason", "")
                axes[1][i].set_title(
                    f"✗ {reason[:30]}",
                    fontsize=6, color="red",
                )
                axes[1][i].axis("off")

            # Clear unused
            for row in range(2):
                start = len(accepted) if row == 0 else len(rejected)
                for j in range(start, cols):
                    axes[row][j].axis("off")

            axes[0][0].set_ylabel("Accepted", fontsize=10, color="green")
            axes[1][0].set_ylabel("Rejected", fontsize=10, color="red")

            fig.suptitle("Segmentation Validation Gate Demo", fontsize=12)
            plt.tight_layout()

            path_out = self.output_dir / f"{name}.tiff"
            fig.savefig(str(path_out), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved validation demo: {path_out}")
            return path_out

        except ImportError:
            logger.warning("matplotlib not available")
            return self.output_dir / f"{name}_FAILED.txt"

    def plot_plant_ratio_distribution(
        self,
        results: list[dict[str, Any]],
        class_names: list[str] | None = None,
        name: str = "plant_ratio_dist",
    ) -> Path:
        """Histogram of plant_ratio values, colored by accept/reject."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            accepted_ratios = [r["plant_ratio"] for r in results if r["accepted"]]
            rejected_ratios = [r["plant_ratio"] for r in results if not r["accepted"]]

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

            bins = np.linspace(0, 1, 50)
            if accepted_ratios:
                ax.hist(accepted_ratios, bins, alpha=0.6, label="Accepted", color="green")
            if rejected_ratios:
                ax.hist(rejected_ratios, bins, alpha=0.6, label="Rejected", color="red")

            ax.set_xlabel("Plant Ratio", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title("Plant Ratio Distribution (Accept vs Reject)", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            path_out = self.output_dir / f"{name}.tiff"
            fig.savefig(str(path_out), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            return path_out

        except ImportError:
            return self.output_dir / f"{name}_FAILED.txt"
