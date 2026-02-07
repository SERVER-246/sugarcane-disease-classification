"""
V2_segmentation/analysis/class_attention_maps.py
=================================================
Per-class average attention maps.

For each disease class, averages GradCAM heatmaps over representative samples.
This reveals class-specific spatial priors (e.g. Smut targets apical whip,
Brown_spot targets leaf surface spots) that inform pseudo-label quality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from V2_segmentation.config import (
    ANALYSIS_DIR, CLASS_NAMES, CLASS_TO_IDX, DEVICE,
)
from V2_segmentation.analysis.gradcam_generator import GradCAMGenerator

logger = logging.getLogger(__name__)


class ClassAttentionMapper:
    """Generate per-class average attention maps.

    Parameters
    ----------
    model : nn.Module
        V1 backbone with trained weights.
    backbone_name : str
        Backbone identifier.
    device : torch.device
        Computation device.
    max_samples_per_class : int
        Max images per class for averaging.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        backbone_name: str,
        device: torch.device = DEVICE,
        max_samples_per_class: int = 50,
    ) -> None:
        self.model = model
        self.backbone_name = backbone_name
        self.device = device
        self.max_samples = max_samples_per_class
        self.gradcam = GradCAMGenerator(model, backbone_name, device)

    def generate_all_class_maps(
        self,
        dataloader: Any,
    ) -> dict[str, np.ndarray]:
        """Generate average attention map for every class.

        Parameters
        ----------
        dataloader : DataLoader yielding (images, labels).

        Returns
        -------
        dict mapping class_name → (H, W) heatmap in [0, 1].
        """
        class_maps: dict[str, np.ndarray] = {}

        for class_name in CLASS_NAMES:
            class_idx = CLASS_TO_IDX[class_name]
            logger.info(f"  Generating attention map for {class_name} (idx={class_idx})...")
            avg_map = self.gradcam.generate_class_average(
                dataloader, class_idx, max_samples=self.max_samples
            )
            class_maps[class_name] = avg_map

        return class_maps

    def save_attention_maps(
        self,
        class_maps: dict[str, np.ndarray],
        output_dir: Path | None = None,
    ) -> Path:
        """Save per-class attention maps as .npy and plots.

        Returns output_dir.
        """
        output_dir = Path(output_dir or ANALYSIS_DIR / self.backbone_name / "attention_maps")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual maps
        for class_name, heatmap in class_maps.items():
            np.save(output_dir / f"{class_name}.npy", heatmap)

        # Save combined visualization
        self._plot_grid(class_maps, output_dir)

        logger.info(f"  Attention maps saved to {output_dir}")
        return output_dir

    def _plot_grid(
        self,
        class_maps: dict[str, np.ndarray],
        output_dir: Path,
    ) -> None:
        """Plot all class attention maps in a grid."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n = len(class_maps)
            cols = 4
            rows = (n + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

            i = -1
            for i, (class_name, heatmap) in enumerate(sorted(class_maps.items())):
                ax = axes_flat[i]
                ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
                ax.set_title(class_name, fontsize=9)
                ax.axis("off")

            # Hide unused axes
            for j in range(i + 1, len(axes_flat)):
                axes_flat[j].axis("off")

            fig.suptitle(
                f"{self.backbone_name} — Per-Class Average GradCAM",
                fontsize=14,
            )
            plt.tight_layout()
            plt.savefig(
                str(output_dir / "class_attention_grid.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
        except ImportError:
            logger.warning("  matplotlib not available — skipping grid plot")

    def cleanup(self) -> None:
        """Remove hooks."""
        self.gradcam.cleanup()
