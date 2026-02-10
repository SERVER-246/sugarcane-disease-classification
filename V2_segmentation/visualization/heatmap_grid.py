"""
V2_segmentation/visualization/heatmap_grid.py
===============================================
Grid visualization of GradCAM heatmaps across backbones and disease classes.

Produces a backbone × class grid showing where each model attends.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

DPI = 1200


class HeatmapGrid:
    """Backbone × class GradCAM heatmap grid."""

    def __init__(self, output_dir: Path | None = None) -> None:
        from V2_segmentation.config import PLOTS_V2_DIR
        self.output_dir = output_dir or PLOTS_V2_DIR / "heatmaps"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_grid(
        self,
        heatmaps: dict[str, dict[str, np.ndarray]],
        backbone_names: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        name: str = "gradcam_grid",
    ) -> Path:
        """Plot heatmap grid.

        Parameters
        ----------
        heatmaps : dict[backbone_name][class_name] → (H, W) float heatmap
        backbone_names : list of backbones for rows
        class_names : list of classes for columns
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if backbone_names is None:
                backbone_names = sorted(heatmaps.keys())
            if class_names is None:
                all_classes: set[str] = set()
                for bk in heatmaps.values():
                    all_classes.update(bk.keys())
                class_names = sorted(all_classes)

            n_rows = len(backbone_names)
            n_cols = len(class_names)

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * 2.5, n_rows * 2.5),
                squeeze=False,
            )

            for r, bk in enumerate(backbone_names):
                for c, cls in enumerate(class_names):
                    ax = axes[r][c]
                    hm = heatmaps.get(bk, {}).get(cls)
                    if hm is not None:
                        ax.imshow(hm, cmap="jet", vmin=0, vmax=1)
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8)
                    ax.axis("off")
                    if r == 0:
                        ax.set_title(cls.replace("_", "\n"), fontsize=6)
                    if c == 0:
                        ax.set_ylabel(bk.replace("Custom", ""), fontsize=6, rotation=0, labelpad=50)

            fig.suptitle("GradCAM Heatmap Grid (Backbone × Disease Class)", fontsize=10)
            plt.tight_layout()

            path = self.output_dir / f"{name}.tiff"
            fig.savefig(str(path), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved heatmap grid: {path}")
            return path

        except ImportError:
            logger.warning("matplotlib not available")
            return self.output_dir / f"{name}_FAILED.txt"

    def plot_single_backbone(
        self,
        backbone_name: str,
        heatmaps: dict[str, np.ndarray],
        images: dict[str, np.ndarray] | None = None,
        name: str | None = None,
    ) -> Path:
        """Plot heatmaps for a single backbone across classes."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import cv2

            class_names = sorted(heatmaps.keys())
            n = len(class_names)
            rows = 2 if images else 1
            fig, axes = plt.subplots(rows, n, figsize=(n * 2.5, rows * 2.5), squeeze=False)

            for c, cls in enumerate(class_names):
                if images and cls in images:
                    img_rgb = cv2.cvtColor(images[cls], cv2.COLOR_BGR2RGB)
                    axes[0][c].imshow(img_rgb)
                    axes[0][c].axis("off")
                    axes[0][c].set_title(cls.replace("_", "\n"), fontsize=6)
                    axes[1][c].imshow(heatmaps[cls], cmap="jet", vmin=0, vmax=1)
                    axes[1][c].axis("off")
                else:
                    axes[0][c].imshow(heatmaps[cls], cmap="jet", vmin=0, vmax=1)
                    axes[0][c].axis("off")
                    axes[0][c].set_title(cls.replace("_", "\n"), fontsize=6)

            fig.suptitle(f"GradCAM: {backbone_name}", fontsize=10)
            plt.tight_layout()

            fname = name or f"heatmap_{backbone_name}"
            path = self.output_dir / f"{fname}.tiff"
            fig.savefig(str(path), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            return path

        except ImportError:
            return self.output_dir / f"heatmap_{backbone_name}_FAILED.txt"
