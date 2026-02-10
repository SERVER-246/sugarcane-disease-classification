"""
V2_segmentation/validation/seg_validator.py
============================================
Segmentation-based image validation gate.

Replaces the heuristic validator from V1 with learned segmentation analysis.
Uses a V2 backbone's segmentation head to determine if an image contains
valid sugarcane disease content.

Decision logic (from Section 10.1):
  1. plant_area = pixels in ANY of Ch1 + Ch2 + Ch3 + Ch4
  2. plant_ratio = plant_area / total_area
  3. IF plant_ratio < per_class_threshold → REJECT
  4. IF largest_connected_component < 500px² → REJECT
  5. IF Ch2 + Ch3 + Ch4 > 0 → PASS (any disease signal)
  6. IF Ch1 > 0.10 → PASS (healthy tissue alone)
  7. ELSE → REJECT
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from V2_segmentation.config import (
    DEVICE, GATE_DEFAULT_MIN_PLANT_RATIO, GATE_MIN_COMPONENT_SIZE_PX,
    GATE_THRESHOLDS, IMG_SIZE, NUM_SEG_CHANNELS, SEG_GATE_BACKBONE,
)
from V2_segmentation.validation.region_analyzer import RegionAnalyzer

logger = logging.getLogger(__name__)


class SegValidator:
    """Segmentation-based image validation.

    Uses a lightweight V2 backbone (default: CustomMobileOne) to produce
    a seg mask, then applies rules to accept/reject.

    Parameters
    ----------
    backbone_name : str
        Which V2 backbone to use for seg inference.
    thresholds : dict or None
        Per-class min_plant_ratio thresholds. Uses calibrated defaults if None.
    min_component_size : int
        Minimum connected component size in pixels.
    """

    def __init__(
        self,
        backbone_name: str = SEG_GATE_BACKBONE,
        thresholds: dict[str, float] | None = None,
        min_component_size: int = GATE_MIN_COMPONENT_SIZE_PX,
    ) -> None:
        self.backbone_name = backbone_name
        self.thresholds = thresholds or dict(GATE_THRESHOLDS)
        self.min_component_size = min_component_size
        self.region_analyzer = RegionAnalyzer()
        self._model: torch.nn.Module | None = None

    def load_model(self) -> bool:
        """Load the validation gate model."""
        try:
            from V2_segmentation.config import CKPT_V2_DIR, BACKBONE_PROFILES
            from V2_segmentation.models.model_factory import build_v2_model

            profile = BACKBONE_PROFILES.get(self.backbone_name, {})
            model = build_v2_model(
                self.backbone_name,
                decoder_channels=profile.get("decoder_channels", 256),
                skip_channels=profile.get("skip_channels", 48),
            )

            # Try to load checkpoint
            for suffix in ("_v2_best.pth", "_v2_final.pth"):
                ckpt = CKPT_V2_DIR / f"{self.backbone_name}{suffix}"
                if ckpt.exists():
                    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
                    if isinstance(state, dict) and "model_state_dict" in state:
                        model.load_state_dict(state["model_state_dict"], strict=False)
                    else:
                        model.load_state_dict(state, strict=False)
                    break

            model.to(DEVICE).eval()
            self._model = model
            logger.info(f"Validation gate loaded: {self.backbone_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load validation gate model: {e}")
            return False

    def validate(
        self,
        image_path: str | Path,
        class_name: str | None = None,
    ) -> dict[str, Any]:
        """Validate a single image.

        Parameters
        ----------
        image_path : str or Path
            Path to input image.
        class_name : str or None
            Expected disease class (for per-class thresholds).

        Returns
        -------
        dict with: accepted (bool), reason (str), plant_ratio, channel_ratios,
                   component_analysis, seg_mask (optional).
        """
        # Get segmentation mask
        seg_mask = self._get_seg_mask(image_path)
        if seg_mask is None:
            return {
                "accepted": False,
                "reason": "Failed to generate segmentation mask",
                "plant_ratio": 0.0,
            }

        # Compute plant ratio (Ch1 + Ch2 + Ch3 + Ch4)
        plant = np.asarray((seg_mask[:, :, 1:] > 0.5).any(axis=2))
        total = seg_mask.shape[0] * seg_mask.shape[1]
        plant_ratio = float(np.sum(plant) / total)

        # Channel ratios
        channel_ratios = {}
        for ch in range(NUM_SEG_CHANNELS):
            ch_ratio = float(np.mean(seg_mask[:, :, ch] > 0.5))
            channel_ratios[ch] = ch_ratio

        # Component analysis
        plant_f32: np.ndarray = np.array(plant, dtype=np.float32)
        component_info = self.region_analyzer.analyze(plant_f32)

        # ── Decision logic ────────────────────────────────────────────
        threshold = self.thresholds.get(
            class_name, GATE_DEFAULT_MIN_PLANT_RATIO
        ) if class_name else GATE_DEFAULT_MIN_PLANT_RATIO

        # Rule 1: Plant ratio check
        if plant_ratio < threshold:
            return self._result(
                False,
                f"plant_ratio={plant_ratio:.3f} < threshold={threshold:.3f}",
                plant_ratio, channel_ratios, component_info,
            )

        # Rule 2: Minimum component size
        if component_info["largest_area"] < self.min_component_size:
            return self._result(
                False,
                f"largest_component={component_info['largest_area']}px "
                f"< {self.min_component_size}px",
                plant_ratio, channel_ratios, component_info,
            )

        # Rule 3: Any disease signal (Ch2 + Ch3 + Ch4)
        disease_ratio = channel_ratios.get(2, 0) + channel_ratios.get(3, 0) + channel_ratios.get(4, 0)
        if disease_ratio > 0.01:
            return self._result(
                True,
                f"disease signal detected (ratio={disease_ratio:.3f})",
                plant_ratio, channel_ratios, component_info,
            )

        # Rule 4: Healthy tissue alone
        if channel_ratios.get(1, 0) > 0.10:
            return self._result(
                True,
                f"healthy tissue detected (Ch1={channel_ratios[1]:.3f})",
                plant_ratio, channel_ratios, component_info,
            )

        # Default: reject
        return self._result(
            False,
            "no significant plant/disease signal",
            plant_ratio, channel_ratios, component_info,
        )

    def validate_batch(
        self,
        image_paths: list[str | Path],
        class_names: list[str | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Validate a batch of images."""
        names: list[str | None] = class_names if class_names is not None else [None] * len(image_paths)
        return [
            self.validate(p, c)
            for p, c in zip(image_paths, names)
        ]

    def _get_seg_mask(self, image_path: str | Path) -> np.ndarray | None:
        """Run seg model on image, return (H, W, 5) mask."""
        if self._model is None:
            if not self.load_model():
                return None

        try:
            img = Image.open(str(image_path)).convert("RGB")
            img_resized = TF.resize(img, [IMG_SIZE, IMG_SIZE])  # type: ignore[arg-type]
            img_tensor = TF.to_tensor(img_resized)  # type: ignore[arg-type]
            img_norm = TF.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_batch = img_norm.unsqueeze(0).to(DEVICE)

            model = self._model
            assert model is not None
            with torch.no_grad():
                outputs = model(img_batch)
                seg_logits = outputs.get("seg_logits")
                if seg_logits is None:
                    return None
                seg_probs = F.softmax(seg_logits, dim=1)
                seg_probs = F.interpolate(
                    seg_probs, size=(IMG_SIZE, IMG_SIZE),
                    mode="bilinear", align_corners=False,
                )
                return seg_probs[0].permute(1, 2, 0).cpu().numpy()
        except Exception as e:
            logger.warning(f"Seg inference failed for {image_path}: {e}")
            return None

    def _result(
        self,
        accepted: bool,
        reason: str,
        plant_ratio: float,
        channel_ratios: dict[int, float],
        component_info: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "accepted": accepted,
            "reason": reason,
            "plant_ratio": round(plant_ratio, 4),
            "channel_ratios": {k: round(v, 4) for k, v in channel_ratios.items()},
            "component_analysis": component_info,
        }
