"""
V2_segmentation/visualization/seg_overlay.py
==============================================
Overlay segmentation masks on original images for visual inspection.

Produces:
  - Side-by-side: original | seg mask | overlay
  - Channel-decomposed views (5 channels color-coded)
  - Grid of samples across disease classes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Channel colors: BG, Healthy, Structural, Surface, Degradation
CHANNEL_COLORS = [
    (64, 64, 64),      # Ch0: BG - dark gray
    (0, 200, 0),       # Ch1: Healthy - green
    (255, 165, 0),     # Ch2: Structural Anomaly - orange
    (255, 255, 0),     # Ch3: Surface Disease Sign - yellow
    (200, 0, 0),       # Ch4: Tissue Degradation - red
]

CHANNEL_NAMES = [
    "Background", "Healthy_Plant_Tissue", "Structural_Anomaly",
    "Surface_Disease_Sign", "Tissue_Degradation",
]

DPI = 1200


class SegOverlay:
    """Overlay segmentation masks on images."""

    def __init__(self, output_dir: Path | None = None, alpha: float = 0.45) -> None:
        from V2_segmentation.config import PLOTS_V2_DIR
        self.output_dir = output_dir or PLOTS_V2_DIR / "overlays"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha

    def overlay_single(
        self,
        image: np.ndarray,
        mask_5ch: np.ndarray,
        title: str = "",
    ) -> np.ndarray:
        """Create side-by-side overlay for one image.

        Parameters
        ----------
        image : (H, W, 3) uint8 BGR
        mask_5ch : (H, W, 5) float32

        Returns
        -------
        (H, W*3, 3) uint8 BGR: [original | colored_mask | overlay]
        """
        h, w = image.shape[:2]
        if mask_5ch.shape[:2] != (h, w):
            mask_5ch = cv2.resize(mask_5ch, (w, h), interpolation=cv2.INTER_LINEAR)

        # Argmax for dominant channel
        dominant = np.argmax(mask_5ch, axis=2)
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for ch, color in enumerate(CHANNEL_COLORS):
            colored[dominant == ch] = color

        # Blend
        overlay = cv2.addWeighted(image, 1 - self.alpha, colored, self.alpha, 0)

        # Stack side by side
        panel = np.concatenate([image, colored, overlay], axis=1)

        if title:
            cv2.putText(
                panel, title, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

        return panel

    def overlay_channels(
        self,
        image: np.ndarray,
        mask_5ch: np.ndarray,
    ) -> np.ndarray:
        """Create per-channel decomposition (5 small panels + original).

        Returns (2*H, 3*W, 3) grid.
        """
        h, w = image.shape[:2]
        if mask_5ch.shape[:2] != (h, w):
            mask_5ch = cv2.resize(mask_5ch, (w, h), interpolation=cv2.INTER_LINEAR)

        panels = [image.copy()]
        for ch in range(5):
            ch_mask = mask_5ch[:, :, ch]
            ch_colored = np.zeros((h, w, 3), dtype=np.uint8)
            ch_colored[:] = (np.array(CHANNEL_COLORS[ch]) * ch_mask[:, :, None]).astype(np.uint8)
            blended = cv2.addWeighted(image, 0.6, ch_colored, 0.4, 0)
            cv2.putText(
                blended, CHANNEL_NAMES[ch], (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            panels.append(blended)

        # 2 rows × 3 cols
        row1 = np.concatenate(panels[:3], axis=1)
        row2 = np.concatenate(panels[3:], axis=1)
        return np.concatenate([row1, row2], axis=0)

    def save_overlay_grid(
        self,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        labels: Sequence[str],
        name: str = "overlay_grid",
        cols: int = 4,
    ) -> Path:
        """Save a grid of overlays.

        Parameters
        ----------
        images : list of (H, W, 3) uint8
        masks : list of (H, W, 5) float32
        labels : list of class names or titles
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n = len(images)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

            if rows == 1:
                axes = [axes] if cols == 1 else list(axes)
            else:
                axes = [ax for row in axes for ax in row]

            for i, (img, mask, label) in enumerate(zip(images, masks, labels)):
                panel = self.overlay_single(img, mask)
                # Convert BGR to RGB for matplotlib
                panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
                axes[i].imshow(panel_rgb)
                axes[i].set_title(label, fontsize=8)
                axes[i].axis("off")

            # Clear unused axes
            for j in range(n, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            path = self.output_dir / f"{name}.tiff"
            fig.savefig(str(path), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved overlay grid: {path}")
            return path

        except ImportError:
            logger.warning("matplotlib not available for grid generation")
            return self.output_dir / f"{name}_FAILED.txt"
