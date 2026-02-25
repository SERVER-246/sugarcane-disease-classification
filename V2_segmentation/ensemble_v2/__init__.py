"""
V2 Enhanced Ensemble Pipeline (Stages 1–12)
============================================
12-stage ensemble combining V2 dual-head backbone predictions.

Stages 1–3: Preserved from V1, re-run on V2 predictions
Stages 4–6: V2-native implementations (feature fusion, MoE, meta-learner)
Stage  7: Superseded by Stage 12
Stages 8–12: New seg-informed stages

Stage  4: Feature-level fusion (ConcatMLP, Attention, Bilinear)
Stage  5: Mixture of Experts (gating network with Top-K routing)
Stage  6: Meta-ensemble controller (XGBoost + MLP)
Stage  8: Segmentation-informed weighting (IoU-based)
Stage  9: Cascaded sequential training (error-coverage)
Stage 10: Adversarial boosting (AdaBoost-style)
Stage 11: Cross-architecture referee (ambiguity resolver)
Stage 12: Upgraded multi-teacher distillation
"""

from .stage1_individual_v2 import Stage1IndividualV2
from .stage2_to_7_rerun import Stage2To7Rerun
from .stage4_feature_fusion_v2 import Stage4FeatureFusionV2
from .stage5_mixture_experts_v2 import Stage5MixtureExpertsV2
from .stage6_meta_learner_v2 import Stage6MetaLearnerV2
from .stage8_seg_informed import Stage8SegInformed
from .stage9_cascaded import Stage9Cascaded
from .stage10_adversarial import Stage10Adversarial
from .stage11_referee import Stage11Referee
from .stage12_distillation_v2 import Stage12DistillationV2
from .ensemble_orchestrator import EnsembleOrchestratorV2
