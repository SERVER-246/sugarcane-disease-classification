"""V2 Models: Backbone adapter, DeepLabV3+ decoder, dual-head architecture."""
from .backbone_adapter import BackboneFeatureExtractor, BACKBONE_HOOK_SPEC
from .decoder import DeepLabV3PlusDecoder
from .dual_head import DualHeadModel
from .model_factory import build_v2_model, load_v1_into_v2
