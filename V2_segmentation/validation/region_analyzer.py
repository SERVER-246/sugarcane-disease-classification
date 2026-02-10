"""
V2_segmentation/validation/region_analyzer.py
==============================================
Connected component analysis for segmentation masks.

Provides spatial structure analysis:
  - Number of connected components
  - Largest component area + bounding box
  - Component size distribution
  - Spatial spread metrics
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RegionAnalyzer:
    """Analyze connected components in binary masks.

    Used by SegValidator to check spatial coherence of plant regions.
    """

    def __init__(self, connectivity: int = 8) -> None:
        """
        Parameters
        ----------
        connectivity : int
            4 or 8 for connected component analysis.
        """
        self.connectivity = connectivity

    def analyze(self, binary_mask: np.ndarray) -> dict[str, Any]:
        """Analyze connected components in a binary mask.

        Parameters
        ----------
        binary_mask : (H, W) float32 or uint8
            Foreground mask (values > 0.5 = foreground).

        Returns
        -------
        dict with component analysis results.
        """
        mask_uint8 = (binary_mask > 0.5).astype(np.uint8)
        n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=self.connectivity
        )

        # Label 0 is background
        n_components = n_labels - 1

        if n_components == 0:
            return {
                "n_components": 0,
                "largest_area": 0,
                "total_fg_area": 0,
                "largest_bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
                "component_areas": [],
                "spatial_spread": 0.0,
                "compactness": 0.0,
            }

        # Component areas (exclude background = label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        total_fg = int(areas.sum())
        largest_idx = areas.argmax() + 1  # +1 because we skipped bg
        largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])

        # Bounding box of largest component
        bbox = {
            "x": int(stats[largest_idx, cv2.CC_STAT_LEFT]),
            "y": int(stats[largest_idx, cv2.CC_STAT_TOP]),
            "w": int(stats[largest_idx, cv2.CC_STAT_WIDTH]),
            "h": int(stats[largest_idx, cv2.CC_STAT_HEIGHT]),
        }

        # Spatial spread: standard deviation of component centroids
        if n_components > 1:
            fg_centroids = centroids[1:]  # skip background
            spatial_spread = float(np.asarray(fg_centroids, dtype=np.float64).std(axis=0).mean())
        else:
            spatial_spread = 0.0

        # Compactness: largest_area / total_fg_area
        compactness = largest_area / max(total_fg, 1)

        return {
            "n_components": n_components,
            "largest_area": largest_area,
            "total_fg_area": total_fg,
            "largest_bbox": bbox,
            "component_areas": sorted(areas.tolist(), reverse=True)[:10],
            "spatial_spread": round(spatial_spread, 2),
            "compactness": round(compactness, 4),
        }

    def analyze_5ch(
        self,
        mask_5ch: np.ndarray,
    ) -> dict[str, dict[str, Any]]:
        """Analyze each channel of a 5-channel mask separately.

        Parameters
        ----------
        mask_5ch : (H, W, 5) float32

        Returns
        -------
        dict mapping channel name → analysis.
        """
        from V2_segmentation.config import SEG_CHANNELS

        results: dict[str, dict[str, Any]] = {}
        for ch_idx, ch_name in SEG_CHANNELS.items():
            ch_mask = mask_5ch[:, :, ch_idx]
            results[ch_name] = self.analyze(ch_mask)
        return results
