"""
V2 Enhanced Ensemble Pipeline (Stages 1–12)
============================================
12-stage ensemble combining V2 dual-head backbone predictions.

Stages 1–7: Preserved from V1, re-run on V2 predictions
Stages 8–12: New seg-informed stages

Stage  8: Segmentation-informed weighting (IoU-based)
Stage  9: Cascaded sequential training (error-coverage)
Stage 10: Adversarial boosting (AdaBoost-style)
Stage 11: Cross-architecture referee (ambiguity resolver)
Stage 12: Upgraded multi-teacher distillation
"""

from .stage1_individual_v2 import Stage1IndividualV2
from .stage2_to_7_rerun import Stage2To7Rerun
from .stage8_seg_informed import Stage8SegInformed
from .stage9_cascaded import Stage9Cascaded
from .stage10_adversarial import Stage10Adversarial
from .stage11_referee import Stage11Referee
from .stage12_distillation_v2 import Stage12DistillationV2
from .ensemble_orchestrator import EnsembleOrchestratorV2
