"""
V2 Segmentation-Based Validation Gate
======================================
Replaces heuristic image validator with learned segmentation-based validation.

Components:
  - SegValidator: Accept/reject using 5-channel seg mask analysis
  - RegionAnalyzer: Connected component analysis of seg masks
  - CalibrateGate: Per-class threshold calibration from gold labels
"""

from .seg_validator import SegValidator
from .region_analyzer import RegionAnalyzer
from .calibrate_gate import CalibrateGate
