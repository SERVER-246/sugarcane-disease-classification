"""
Red Rot Symptom Segmentation Engine
===================================
Stage 2 of the Expert View two-stage pipeline.

When the broad 13-class classifier (Stage 1) predicts Red Rot, this engine runs
the dedicated Red Rot symptom-segmentation specialist (DeepLabV3+ with a
``CustomResNetMish`` encoder, trained on the full annotated Red Rot dataset —
Snapshot 3 retrain, the best individual backbone at ~22.8% mIoU) and returns a
per-symptom mask overlay, the percentage of affected tissue, and the list of
detected symptoms.

The ``DeepLabV3Plus`` / ``ASPP`` graph below is copied verbatim from
``Paper_redrot/train_seed_segmentation.py`` so the trained ``state_dict`` loads
with an exact key match. The drift-prone feature-hook specification is imported
live from ``V2_segmentation.models.backbone_adapter`` (the same source the
training script uses), so encoder hook points can never silently diverge.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from inference_server import config


logger = logging.getLogger("inference_server.redrot_segmentation")


# ---------------------------------------------------------------------------
# Segmentation model graph (verbatim from train_seed_segmentation.py)
# ---------------------------------------------------------------------------
class _ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ASPPPool(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        return F.interpolate(self.pool(x), size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 256, rates: tuple[int, ...] = (6, 12, 18)) -> None:
        super().__init__()
        branches: list[nn.Module] = [
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        ]
        for r in rates:
            branches.append(_ASPPConv(in_ch, out_ch, r))
        branches.append(_ASPPPool(in_ch, out_ch))
        self.convs = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(len(branches) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([c(x) for c in self.convs], dim=1))


class _Hook:
    """Explicit feature-hook specification (server-owned, matches the deployed weights)."""

    def __init__(self, module_path: str, channels: int, needs_reshape: bool = False, spatial: int = 0) -> None:
        self.module_path = module_path
        self.channels = channels
        self.needs_reshape = needs_reshape
        self.spatial = spatial


# Authoritative hook points for each deployable Red Rot specialist backbone.
# Sourced from the adapter that actually trained the Snapshot 3 retrain weights
# (migration bundle), NOT the repo's V2 adapter, which has since drifted.
# Channels here are defaults only — the loader overrides them from the checkpoint.
SEG_HOOK_SPEC: dict[str, dict[str, _Hook]] = {
    "CustomResNetMish": {
        "low_level": _Hook("layer2.0.conv1", 128),
        "high_level": _Hook("layer4", 2048),
    },
    "CustomDenseNetHybrid": {
        "low_level": _Hook("trans1", 128),
        "high_level": _Hook("norm", 512),
    },
    "CustomRegNet": {
        "low_level": _Hook("stages.2.0.conv1", 160),
        "high_level": _Hook("stages.3", 1536),
    },
}


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ for symptom-level segmentation using a V1 backbone encoder."""

    def __init__(
        self,
        backbone: nn.Module,
        backbone_name: str,
        num_classes: int,
        low_level: _Hook,
        high_level: _Hook,
        decoder_channels: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        self._low_spec = low_level
        self._high_spec = high_level

        self._features: dict[str, torch.Tensor] = {}
        self._hooks: list[Any] = []
        self._register_hooks()

        self.aspp = ASPP(high_level.channels, decoder_channels)

        skip_ch = 48
        self.skip_reduce = nn.Sequential(
            nn.Conv2d(low_level.channels, skip_ch, 1, bias=False),
            nn.BatchNorm2d(skip_ch),
            nn.ReLU(True),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(decoder_channels + skip_ch, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )

        self.seg_head = nn.Conv2d(decoder_channels, num_classes, 1)

    def _register_hooks(self) -> None:
        for name, spec_item in [("low_level", self._low_spec), ("high_level", self._high_spec)]:
            module = self.backbone
            for attr in spec_item.module_path.split("."):
                if attr.isdigit():
                    module = module[int(attr)]  # type: ignore[index]
                else:
                    module = getattr(module, attr)

            def make_hook(n: str):
                def hook_fn(mod: nn.Module, inp: Any, out: torch.Tensor) -> None:
                    self._features[n] = out
                return hook_fn

            h = module.register_forward_hook(make_hook(name))
            self._hooks.append(h)

    def _get_feature(self, name: str) -> torch.Tensor:
        feat = self._features.get(name)
        if feat is None:
            raise RuntimeError(f"Hook '{name}' did not fire for {self.backbone_name}")
        spec_item = self._low_spec if name == "low_level" else self._high_spec
        if spec_item.needs_reshape and feat.dim() == 3:
            B, N, C = feat.shape
            h = w = spec_item.spatial
            if h * w != N:
                h = w = int(math.sqrt(N))
            feat = feat.transpose(1, 2).reshape(B, C, h, w).contiguous()
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        self._features.clear()

        _ = self.backbone(x)

        low_level = self._get_feature("low_level")
        high_level = self._get_feature("high_level")

        x_aspp = self.aspp(high_level)

        skip = self.skip_reduce(low_level)
        x_aspp = F.interpolate(x_aspp, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x_cat = torch.cat([x_aspp, skip], dim=1)
        x_ref = self.refine(x_cat)

        logits = self.seg_head(x_ref)
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return logits


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SymptomStat:
    name: str
    pixel_fraction: float   # fraction of the whole image covered by this symptom
    mean_confidence: float  # mean softmax probability over pixels predicted as this symptom
    color: str = "#000000"  # overlay color (hex) for the legend


@dataclass
class RedRotSegResult:
    affected_area_pct: float
    dominant_symptom: str | None
    symptoms: list[SymptomStat]
    overlay_png_b64: str
    mask_height: int
    mask_width: int
    backbone: str
    num_symptom_classes_detected: int = field(default=0)


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_models: dict[str, DeepLabV3Plus] = {}
_device: torch.device = torch.device("cpu")
_idx_to_name: dict[int, str] = {}
_num_classes: int = 0
_palette: np.ndarray | None = None


def _ensure_project_root_on_path() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_category_mapping() -> dict[int, str]:
    """Load the authoritative index -> symptom name mapping."""
    cat_path = Path(config.REDROT_SEG_CATEGORY_MAP)
    if not cat_path.exists():
        raise FileNotFoundError(
            f"Red Rot category mapping not found at {cat_path}. "
            f"Set DBT_REDROT_CATEGORY_MAP to the category_mapping.json produced by Script A."
        )
    with open(cat_path, encoding="utf-8") as f:
        name_to_idx: dict[str, int] = json.load(f)
    return {int(v): k for k, v in name_to_idx.items()}


def _build_palette(num_classes: int) -> np.ndarray:
    """Deterministic distinct RGB colors; index 0 (background) is black."""
    base = np.array([
        [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [67, 99, 216],
        [245, 130, 49], [145, 30, 180], [66, 212, 244], [240, 50, 230], [191, 239, 69],
        [250, 190, 212], [70, 153, 144], [220, 190, 255], [154, 99, 36], [255, 250, 200],
        [128, 0, 0], [170, 255, 195], [128, 128, 0], [255, 216, 177], [0, 0, 117],
        [169, 169, 169],
    ], dtype=np.uint8)
    if num_classes <= len(base):
        return base[:num_classes]
    reps = int(math.ceil(num_classes / len(base)))
    return np.tile(base, (reps, 1))[:num_classes]


def _load_one(backbone_name: str) -> DeepLabV3Plus:
    """Build and load one Red Rot segmentation backbone from its retrain checkpoint."""
    from Base_backbones import create_custom_backbone

    ckpt_path = config.REDROT_SEG_RETRAIN_DIR / f"{backbone_name}_retrain_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Red Rot checkpoint not found at {ckpt_path}. "
            f"Set DBT_REDROT_SEG_DIR to the seg_checkpoints_retrain directory."
        )
    if backbone_name not in SEG_HOOK_SPEC:
        raise RuntimeError(
            f"No server hook spec for '{backbone_name}'. Add it to SEG_HOOK_SPEC "
            f"(module paths) before deploying this Red Rot specialist."
        )

    ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
    if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt):
        raise RuntimeError(f"Unexpected checkpoint format at {ckpt_path}: missing 'model_state_dict'.")
    state_dict = ckpt["model_state_dict"]

    ckpt_num_classes = int(ckpt["num_classes"]) if "num_classes" in ckpt else _num_classes
    if ckpt_num_classes != _num_classes:
        raise RuntimeError(
            f"[{backbone_name}] checkpoint has {ckpt_num_classes} classes but the category "
            f"mapping has {_num_classes}. They must match."
        )

    fused_mode = any("fused_conv" in k for k in state_dict)
    backbone = create_custom_backbone(backbone_name, ckpt_num_classes)
    if backbone_name in ("CustomMobileOne", "CustomDynamicConvNet") and fused_mode and hasattr(backbone, "fuse_model"):
        backbone.fuse_model()

    # Size the decoder straight from the checkpoint tensors (drift-proof).
    spec = SEG_HOOK_SPEC[backbone_name]
    low_ch = int(state_dict["skip_reduce.0.weight"].shape[1])
    high_ch = int(state_dict["aspp.convs.0.0.weight"].shape[1])
    low_level = _Hook(spec["low_level"].module_path, low_ch, spec["low_level"].needs_reshape, spec["low_level"].spatial)
    high_level = _Hook(spec["high_level"].module_path, high_ch, spec["high_level"].needs_reshape, spec["high_level"].spatial)

    model = DeepLabV3Plus(backbone, backbone_name, num_classes=ckpt_num_classes, low_level=low_level, high_level=high_level)

    # Drop stale V1 backbone classification-head weights (never used by DeepLabV3+),
    # and fail on any genuine mismatch.
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    stale: set[str] = set()
    for k, v in state_dict.items():
        if not k.startswith("backbone."):
            continue
        if k not in model_keys or (hasattr(v, "shape") and v.shape != model_sd[k].shape):
            stale.add(k)

    unexpected = (set(state_dict) - model_keys) - stale
    if unexpected:
        raise RuntimeError(f"[{backbone_name}] unexpected keys in seg checkpoint: {sorted(unexpected)}")

    clean_state = {k: v for k, v in state_dict.items() if k not in stale}
    shape_errors = [
        f"{k}: ckpt {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}"
        for k, v in clean_state.items()
        if k in model_sd and hasattr(v, "shape") and v.shape != model_sd[k].shape
    ]
    if shape_errors:
        raise RuntimeError(f"[{backbone_name}] shape mismatch: " + "; ".join(shape_errors))

    model.load_state_dict(clean_state, strict=False)
    return model.to(_device).eval()


def load_redrot_segmenter(device: str | None = None) -> dict[str, DeepLabV3Plus]:
    """Load and cache the Red Rot segmentation ensemble. Fails loudly on error."""
    global _models, _device, _idx_to_name, _num_classes, _palette  # noqa: PLW0603

    if _models:
        return _models

    _ensure_project_root_on_path()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _device = torch.device(device)

    _idx_to_name = _load_category_mapping()
    _num_classes = len(_idx_to_name)
    _palette = _build_palette(_num_classes)

    loaded: dict[str, DeepLabV3Plus] = {}
    for name in config.REDROT_SEG_BACKBONES:
        loaded[name] = _load_one(name)
    if not loaded:
        raise RuntimeError("No Red Rot segmentation backbones configured (DBT_REDROT_SEG_BACKBONES).")

    _models = loaded
    logger.info(
        "Red Rot segmenter loaded: %d-backbone ensemble %s (%d classes) on %s",
        len(_models), list(_models), _num_classes, _device,
    )
    return _models


def is_ready() -> bool:
    return bool(_models)


def _preprocess(image: Image.Image) -> torch.Tensor:
    size = config.REDROT_SEG_IMG_SIZE
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.array(config.IMAGENET_MEAN, dtype=np.float32)) / np.array(config.IMAGENET_STD, dtype=np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def segment_redrot(image_bytes: bytes) -> RedRotSegResult:
    """Run Red Rot symptom segmentation and return per-symptom stats + overlay."""
    models = load_redrot_segmenter()

    try:
        pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    tensor = _preprocess(pil).to(_device)

    with torch.no_grad():
        prob_sum: torch.Tensor | None = None
        for m in models.values():
            p = torch.softmax(m(tensor), dim=1)
            prob_sum = p if prob_sum is None else prob_sum + p
        assert prob_sum is not None
        probs = (prob_sum / len(models)).squeeze(0)           # (C, H, W) soft-vote
        conf, pred = probs.max(dim=0)                          # (H, W), (H, W)

    pred_np = pred.cpu().numpy().astype(np.int64)
    conf_np = conf.cpu().numpy().astype(np.float32)
    h, w = pred_np.shape
    total = int(h * w)

    symptoms: list[SymptomStat] = []
    foreground_pixels = 0
    for idx in range(1, _num_classes):
        mask = pred_np == idx
        count = int(mask.sum())
        if count == 0:
            continue
        foreground_pixels += count
        r, g, b = (int(v) for v in _palette[idx])
        symptoms.append(
            SymptomStat(
                name=_idx_to_name.get(idx, f"class_{idx}"),
                pixel_fraction=round(count / total, 6),
                mean_confidence=round(float(conf_np[mask].mean()), 6),
                color=f"#{r:02x}{g:02x}{b:02x}",
            )
        )

    symptoms.sort(key=lambda s: s.pixel_fraction, reverse=True)
    affected_area_pct = round(100.0 * foreground_pixels / total, 4)
    dominant = symptoms[0].name if symptoms else None

    # Color overlay (RGBA PNG; background fully transparent)
    assert _palette is not None
    rgb = _palette[pred_np]                                   # (H, W, 3)
    alpha = np.where(pred_np > 0, 180, 0).astype(np.uint8)    # translucent symptoms
    rgba = np.dstack([rgb, alpha])
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return RedRotSegResult(
        affected_area_pct=affected_area_pct,
        dominant_symptom=dominant,
        symptoms=symptoms,
        overlay_png_b64=overlay_b64,
        mask_height=h,
        mask_width=w,
        backbone="+".join(models),
        num_symptom_classes_detected=len(symptoms),
    )
