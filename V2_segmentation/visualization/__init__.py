"""
V2 Segmentation Visualization Tools
=====================================
Publication-quality plots and overlays for segmentation pipeline.

All plots saved as labeled .tiff at 1200 DPI per V1 convention.

Components:
  - BackbonePlots: Confusion matrix, ROC curves, per-class P/R/F1 bars (V1-matching)
  - EnsembleStagePlots: Same plots per ensemble stage
  - SegOverlay: Overlay seg masks on original images
  - HeatmapGrid: Grid of heatmaps across backbones
  - TrainingCurves: Phase A/B/C loss + metric curves
  - EnsembleComparison: Bar charts comparing ensemble stages
  - ValidationDemo: Before/after validation gate visualization
  - TierDistribution: Pseudo-label tier pie/bar charts
"""

from .backbone_plots import BackbonePlots
from .ensemble_stage_plots import EnsembleStagePlots
from .seg_overlay import SegOverlay
from .heatmap_grid import HeatmapGrid
from .training_curves import TrainingCurves
from .ensemble_comparison import EnsembleComparison
from .validation_demo import ValidationDemo
from .tier_distribution import TierDistribution

__all__ = [
    "BackbonePlots", "EnsembleStagePlots",
    "SegOverlay", "HeatmapGrid", "TrainingCurves",
    "EnsembleComparison", "ValidationDemo", "TierDistribution",
]
