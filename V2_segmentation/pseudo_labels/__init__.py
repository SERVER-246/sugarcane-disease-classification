"""
V2 Pseudo-Label Generation Pipeline
====================================
Generates 5-channel segmentation masks from V1 models without manual annotation.

Three independent mask sources → weighted fusion → quality scoring → tier assignment.

Pipeline:
  1. GrabCut (green-channel seeded foreground extraction)
  2. GradCAM (ensemble of V1 backbone activation maps)
  3. SAM (optional zero-shot segmentation)
  4. Fusion (weighted pixel-wise combination)
  5. Quality scoring + tier assignment (A/B/C)
  6. Class-specific sanity checks
  7. Human spot-check (blocking gate)
  8. Iterative refinement (post-Phase-A self-training)
"""

from .grabcut_generator import GrabCutGenerator
from .gradcam_mask_generator import GradCAMMaskGenerator
from .mask_combiner import MaskCombiner
from .mask_quality_scorer import MaskQualityScorer
from .class_sanity_checker import ClassSanityChecker
from .iterative_refiner import IterativeRefiner
