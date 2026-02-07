"""
V2_segmentation/models/backbone_adapter.py
==========================================
Hook-based feature extraction for all 15 V1 backbones.

Each backbone has different stage naming, channel dims, spatial resolutions,
and format (spatial BCHW vs sequence BNC).  This module provides a unified
``BackboneFeatureExtractor`` that registers forward hooks on the right modules
and returns a dict of multi-scale features ready for the DeepLabV3+ decoder.

Two feature maps are extracted per backbone:
  - ``low_level``  — high-resolution (~56×56), used for the decoder skip connection
  - ``high_level`` — low-resolution (7–28×), used as ASPP input
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ============================================================================
#  Hook specification dataclass
# ============================================================================

@dataclass
class HookSpec:
    """Specification for a single hook point on a backbone."""
    module_path: str          # Dot-separated path to the nn.Module to hook
    channels: int             # Number of output channels / embed dim
    spatial: int              # Expected H=W spatial resolution
    is_sequence: bool = False # True if output is (B, N, C) instead of (B, C, H, W)
    needs_reshape: bool = False  # True if we must reshape BNC → BCHW


@dataclass
class BackboneSpec:
    """Full hook specification for one backbone."""
    low_level: HookSpec
    high_level: HookSpec
    gradcam_layer: str        # Module path for GradCAM target
    classifier_attr: str      # "head" | "classifier" | "fc"  (for head isolation)
    notes: str = ""

# ============================================================================
#  Per-backbone hook specifications  (verified against Base_backbones.py)
# ============================================================================

BACKBONE_HOOK_SPEC: dict[str, BackboneSpec] = {

    # ── LIGHT tier ──────────────────────────────────────────────────────────

    "CustomEfficientNetV4": BackboneSpec(
        # stem→112(32) | stages[0]→112(16) | stages[1]→56(24) | ... | head_conv→7(1280)
        low_level=HookSpec("stages.1", 24, 56),
        high_level=HookSpec("head_conv", 1280, 7),
        gradcam_layer="head_conv",
        classifier_attr="classifier",
    ),
    "CustomDenseNetHybrid": BackboneSpec(
        # conv1→pool1→56(64) | dense1→56(256) | trans1→pool2→28(128) | dense2→28(512)
        # | trans2→pool3→14(256) | dense3→14(512)
        low_level=HookSpec("dense1", 256, 56),
        high_level=HookSpec("dense3", 512, 14),
        gradcam_layer="dense3",
        classifier_attr="fc",
    ),
    "CustomInceptionV4": BackboneSpec(
        # stem→56(256) | inception_blocks→56(256) | reduction→28(512)
        low_level=HookSpec("inception_blocks", 256, 56),
        high_level=HookSpec("reduction", 512, 28),
        gradcam_layer="reduction",
        classifier_attr="fc",
    ),
    "CustomMobileOne": BackboneSpec(
        # stem→112(32) | stages[0]→56(64) | stages[1]→28(128) | stages[2]→14(256)
        # | stages[3]→14(384)
        low_level=HookSpec("stages.0", 64, 56),
        high_level=HookSpec("stages.3", 384, 14),
        gradcam_layer="stages.3",
        classifier_attr="fc",
    ),
    "CustomGhostNetV2": BackboneSpec(
        # stem→112(32) | stages[0]→56(48) | stages[1]→28(96) | stages[2]→14(192)
        # | stages[3]→7(384) | stages[4]→7(512) | conv_head→7(1280)
        low_level=HookSpec("stages.0", 48, 56),
        high_level=HookSpec("conv_head", 1280, 7),
        gradcam_layer="conv_head",
        classifier_attr="classifier",
    ),

    # ── MEDIUM tier ─────────────────────────────────────────────────────────

    "CustomConvNeXt": BackboneSpec(
        # stem→56(96) | stages[0]→56(96) | stages[1]=DS→28 | stages[2]→28(192)
        # | stages[3]=DS→14 | stages[4]→14(384) | stages[5]=DS→7 | stages[6]→7(768)
        low_level=HookSpec("stages.0", 96, 56),
        high_level=HookSpec("stages.6", 768, 7),
        gradcam_layer="stages.6",
        classifier_attr="head",
        notes="stages[1,3,5] are LayerNorm+Conv2d downsamplers",
    ),
    "CustomResNetMish": BackboneSpec(
        # conv1→112(64) | maxpool→56(64) | layer1→56(256) | layer2→28(512)
        # | layer3→14(1024) | layer4→7(2048)
        low_level=HookSpec("layer1", 256, 56),
        high_level=HookSpec("layer4", 2048, 7),
        gradcam_layer="layer4",
        classifier_attr="fc",
    ),
    "CustomRegNet": BackboneSpec(
        # stem→112(32) | stages[0]→112(128) stride=1 | stages[1]→56(256) | stages[2]→28(640)
        # | stages[3]→14(1536)
        low_level=HookSpec("stages.1", 256, 56),
        high_level=HookSpec("stages.3", 1536, 14),
        gradcam_layer="stages.3",
        classifier_attr="fc",
        notes="MishBottleneck with 4× expansion: widths×4 = actual channels",
    ),

    # ── HIGH tier ───────────────────────────────────────────────────────────

    "CustomCSPDarkNet": BackboneSpec(
        # stem→224(64) stride=1! | stages[0]→224(128) no pool
        # | pool→112 stages[1]→112(256) | pool→56 stages[2]→56(512)
        # | pool→28 stages[3]→28(1024) | AdaptiveAvgPool→1
        # NOTE: pool is inline F.max_pool2d in forward(), not a hookable module
        low_level=HookSpec("stages.2", 512, 56),
        high_level=HookSpec("stages.3", 1024, 28),
        gradcam_layer="stages.3",
        classifier_attr="fc",
        notes="Stem stride=1 (no downsample). Pool before stages 1-3 is inline code.",
    ),
    "CustomDynamicConvNet": BackboneSpec(
        # stem→112(64) | stages[0]→56(128) | stages[1]→28(256)
        # | stages[2]→14(512) | stages[3]→7(1024)
        low_level=HookSpec("stages.0", 128, 56),
        high_level=HookSpec("stages.3", 1024, 7),
        gradcam_layer="stages.3",
        classifier_attr="fc",
    ),

    # ── HEAVY tier (transformers / hybrids) ─────────────────────────────────

    "CustomDeiTStyle": BackboneSpec(
        # patch_embed is nn.Sequential of 4 conv stages:
        # [0-2]: 224→112 (192ch) | [3-5]: 112→56 (384ch) | [6-8]: 56→28 (768ch) | [9]: 28→14 (768ch)
        # Then flatten→sequence, transformer blocks
        low_level=HookSpec("patch_embed.5", 384, 56),   # After GELU of 2nd conv stage
        high_level=HookSpec("patch_embed", 768, 14),     # Full patch_embed output (still spatial)
        gradcam_layer="patch_embed",
        classifier_attr="head",
        notes="Pure transformer. patch_embed output is spatial; everything after is sequence.",
    ),
    "CustomSwinTransformer": BackboneSpec(
        # patch_embed→56(128) SPATIAL | then flatten→sequence
        # layers[0]=SwinBlocks 56×56 | layers[1]=PatchMerging→28(256)
        # layers[2]=SwinBlocks 28×28 | layers[3]=PatchMerging→14(512)
        # layers[4]=SwinBlocks 14×14 | layers[5]=PatchMerging→7(1024)
        # layers[6]=SwinBlocks 7×7
        low_level=HookSpec("patch_embed", 128, 56),
        high_level=HookSpec("layers.6", 1024, 7, is_sequence=True, needs_reshape=True),
        gradcam_layer="layers.6",
        classifier_attr="head",
        notes="All layers after patch_embed operate in sequence format (B,N,C).",
    ),
    "CustomMaxViT": BackboneSpec(
        # stem→56(128) | stage1→28(256) CNN | stage2_cnn→14(512) CNN
        # | stage2_attn→14(512) SEQ (external residual!) | reshape back
        # | stage3_proj→14(768) SPATIAL | stage3_transformer→14(768) SEQ
        low_level=HookSpec("stem", 128, 56),
        high_level=HookSpec("stage3_proj", 768, 14),
        gradcam_layer="stage3_proj",
        classifier_attr="head",
        notes="stage3_proj is last spatial layer. stage2/3 attn use external residual.",
    ),
    "CustomCoAtNet": BackboneSpec(
        # stem→56(128) | stage1→28(256) CNN | stage2_cnn→14(512) CNN
        # | stage2_attn→14(512) SEQ (external residual) | reshape
        # | stage3_proj→14(768) SPATIAL | stage3_transformer→14(768) SEQ
        low_level=HookSpec("stem", 128, 56),
        high_level=HookSpec("stage3_proj", 768, 14),
        gradcam_layer="stage3_proj",
        classifier_attr="head",
        notes="stage3_proj is last spatial layer. Attention blocks use external residual.",
    ),
    "CustomViTHybrid": BackboneSpec(
        # Deep CNN stem (12 layers in nn.Sequential):
        # [0-1]: 224→112 (64ch) | [2-4]: 112→56 (128ch) | [5-7]: 56→28 (256ch)
        # [8-10]: 28→14 (512ch) | [11]: 14→14 (768ch, projection)
        # Then flatten→sequence + CLS token + pos embed + 18 transformer blocks
        low_level=HookSpec("stem.4", 128, 56),   # End of 128ch block before 56→28
        high_level=HookSpec("stem.11", 768, 14),  # Final CNN projection (spatial)
        gradcam_layer="stem.11",
        classifier_attr="head",
        notes="stem is flat nn.Sequential(12 CoreImageBlocks). Transformers are sequence-only.",
    ),
}

# ============================================================================
#  Module path resolver
# ============================================================================

def resolve_module(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dot-separated module path to the actual nn.Module.

    Examples
    --------
    >>> _resolve_module(model, "stages.0")   # model.stages[0]
    >>> _resolve_module(model, "stem.4")     # model.stem[4]
    >>> _resolve_module(model, "conv_head")  # model.conv_head
    """
    parts = path.split(".")
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
        else:
            current = getattr(current, part)
    return current


# ============================================================================
#  BackboneFeatureExtractor
# ============================================================================

class BackboneFeatureExtractor:
    """Non-invasive feature extraction from V1 backbones via forward hooks.

    Registers hooks on the modules specified by ``BACKBONE_HOOK_SPEC`` and
    captures their outputs during ``forward()``.  After calling :meth:`extract`,
    the captured features are returned as a dict suitable for the DeepLabV3+
    decoder.

    Parameters
    ----------
    model : nn.Module
        A V1 backbone instance (e.g. ``CustomEfficientNetV4``).
    backbone_name : str
        Key into ``BACKBONE_HOOK_SPEC``.

    Usage
    -----
    >>> extractor = BackboneFeatureExtractor(model, "CustomEfficientNetV4")
    >>> feats = extractor.extract(images)
    >>> feats["low_level"]   # (B, 24, 56, 56)
    >>> feats["high_level"]  # (B, 1280, 7, 7)
    >>> feats["cls_logits"]  # (B, num_classes)
    """

    def __init__(self, model: nn.Module, backbone_name: str) -> None:
        if backbone_name not in BACKBONE_HOOK_SPEC:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Available: {sorted(BACKBONE_HOOK_SPEC)}"
            )
        self.model = model
        self.backbone_name = backbone_name
        self.spec = BACKBONE_HOOK_SPEC[backbone_name]
        self._features: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._register_hooks()

    # ── hook management ─────────────────────────────────────────────────

    def _make_hook(self, name: str) -> Any:
        """Create a forward-hook closure that stores the output."""
        def hook_fn(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self._features[name] = output
        return hook_fn

    def _register_hooks(self) -> None:
        """Register hooks on the low-level and high-level modules."""
        for label, hook_spec in [
            ("low_level", self.spec.low_level),
            ("high_level", self.spec.high_level),
        ]:
            target_module = resolve_module(self.model, hook_spec.module_path)
            handle = target_module.register_forward_hook(self._make_hook(label))
            self._handles.append(handle)
            logger.debug(
                f"  Hook registered: {self.backbone_name}.{hook_spec.module_path} → '{label}'"
            )

    def clear_features(self) -> None:
        """Clear cached features from previous forward pass."""
        self._features.clear()

    def get_feature(self, name: str) -> torch.Tensor | None:
        """Retrieve a captured feature tensor by name."""
        return self._features.get(name)

    def remove_hooks(self) -> None:
        """Remove all registered hooks (call when done)."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._features.clear()

    # ── feature extraction ──────────────────────────────────────────────

    def _maybe_reshape(
        self, tensor: torch.Tensor, hook_spec: HookSpec
    ) -> torch.Tensor:
        """Reshape sequence (B, N, C) → spatial (B, C, H, W) if needed."""
        if not hook_spec.needs_reshape:
            return tensor
        B, N, C = tensor.shape
        h = w = hook_spec.spatial
        if N != h * w:
            # Attempt to infer spatial dims from token count
            h = w = int(math.sqrt(N))
            if h * w != N:
                raise RuntimeError(
                    f"Cannot reshape sequence length {N} to spatial "
                    f"(expected {hook_spec.spatial}²={hook_spec.spatial**2})"
                )
        return tensor.transpose(1, 2).reshape(B, C, h, w).contiguous()

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run forward pass and return captured features.

        Returns
        -------
        dict with keys:
            ``low_level``  — (B, C_low,  H_low,  W_low)   spatial tensor
            ``high_level`` — (B, C_high, H_high, W_high)   spatial tensor
            ``cls_logits`` — (B, num_classes)               raw logits
        """
        self._features.clear()
        cls_logits = self.model(x)

        # Post-process captured features
        result: dict[str, torch.Tensor] = {"cls_logits": cls_logits}
        for label, hook_spec in [
            ("low_level", self.spec.low_level),
            ("high_level", self.spec.high_level),
        ]:
            feat = self._features.get(label)
            if feat is None:
                raise RuntimeError(
                    f"Hook '{label}' did not capture output for "
                    f"{self.backbone_name}.{hook_spec.module_path}"
                )
            result[label] = self._maybe_reshape(feat, hook_spec)
        return result

    def extract_with_grad(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Like ``extract`` but keeps gradients (for GradCAM etc.)."""
        self._features.clear()
        cls_logits = self.model(x)
        result: dict[str, torch.Tensor] = {"cls_logits": cls_logits}
        for label, hook_spec in [
            ("low_level", self.spec.low_level),
            ("high_level", self.spec.high_level),
        ]:
            feat = self._features.get(label)
            if feat is None:
                raise RuntimeError(
                    f"Hook '{label}' did not capture output for "
                    f"{self.backbone_name}.{hook_spec.module_path}"
                )
            result[label] = self._maybe_reshape(feat, hook_spec)
        return result

    # ── convenience ─────────────────────────────────────────────────────

    def get_gradcam_target_module(self) -> nn.Module:
        """Return the module used as GradCAM target layer."""
        return resolve_module(self.model, self.spec.gradcam_layer)

    def get_classifier_head(self) -> nn.Module:
        """Return the backbone's classification head module."""
        return getattr(self.model, self.spec.classifier_attr)

    def __repr__(self) -> str:
        return (
            f"BackboneFeatureExtractor({self.backbone_name}, "
            f"low={self.spec.low_level.module_path}@{self.spec.low_level.spatial}², "
            f"high={self.spec.high_level.module_path}@{self.spec.high_level.spatial}²)"
        )
