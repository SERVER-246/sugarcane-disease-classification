"""
V2_segmentation/pseudo_labels/mask_combiner.py
===============================================
Weighted pixel-wise fusion of multi-source pseudo-labels into 5-channel
segmentation masks.

Combines:
  - GrabCut foreground mask    (weight 0.3)
  - GradCAM ensemble heatmap   (weight 0.5)
  - SAM mask (optional)        (weight 0.2, redistributed if unavailable)

Produces per-image:
  - 5-channel mask (H, W, 5): BG, Healthy, Structural, Surface, Degradation
  - Confidence map (H, W): per-pixel fusion confidence [0, 1]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from V2_segmentation.config import (
    DISEASE_CHANNEL_MAP, FUSION_WEIGHTS, IMG_SIZE,
    NUM_SEG_CHANNELS, UNCERTAINTY_THRESHOLD,
)

logger = logging.getLogger(__name__)


class MaskCombiner:
    """Combine multi-source masks into 5-channel segmentation pseudo-labels.

    Parameters
    ----------
    fusion_weights : dict
        Keys: ``"grabcut"``, ``"gradcam"``, ``"sam"``. Values: float weights.
    uncertainty_threshold : float
        Pixels with confidence below this are flagged as uncertain.
    """

    def __init__(
        self,
        fusion_weights: dict[str, float] | None = None,
        uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
    ) -> None:
        self.fusion_weights = fusion_weights or dict(FUSION_WEIGHTS)
        self.uncertainty_threshold = uncertainty_threshold

    def combine(
        self,
        grabcut_mask: np.ndarray,
        gradcam_mask: np.ndarray,
        sam_mask: np.ndarray | None,
        class_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine masks from all sources into a 5-channel semantic mask.

        Parameters
        ----------
        grabcut_mask : (H, W) float32
            GrabCut foreground probability [0, 1].
        gradcam_mask : (H, W) float32
            Ensemble GradCAM heatmap [0, 1].
        sam_mask : (H, W) float32 or None
            SAM foreground probability [0, 1].
        class_name : str
            Disease class (e.g. ``"Brown_spot"``).

        Returns
        -------
        mask_5ch : (H, W, 5) float32
            5-channel segmentation mask.
        confidence : (H, W) float32
            Per-pixel confidence [0, 1].
        """
        H, W = IMG_SIZE, IMG_SIZE

        # Ensure correct shapes
        grabcut_mask = self._ensure_shape(grabcut_mask, H, W)
        gradcam_mask = self._ensure_shape(gradcam_mask, H, W)

        # ── Step 1: Weighted foreground fusion ───────────────────────
        w = dict(self.fusion_weights)
        if sam_mask is None:
            # Redistribute SAM weight proportionally
            sam_w = w.pop("sam", 0.2)
            total_remaining = w.get("grabcut", 0.3) + w.get("gradcam", 0.5)
            if total_remaining > 0:
                w["grabcut"] = w.get("grabcut", 0.3) * (1 + sam_w / total_remaining)
                w["gradcam"] = w.get("gradcam", 0.5) * (1 + sam_w / total_remaining)
        else:
            sam_mask = self._ensure_shape(sam_mask, H, W)

        # Weighted pixel-wise combination
        fg_score = (
            w.get("grabcut", 0.375) * grabcut_mask
            + w.get("gradcam", 0.625) * gradcam_mask
        )
        if sam_mask is not None:
            fg_score += w.get("sam", 0.2) * sam_mask

        # Normalize to [0, 1]
        max_val = fg_score.max()
        if max_val > 0:
            fg_score = fg_score / max_val

        # ── Step 2: Confidence computation ───────────────────────────
        # Agreement among sources increases confidence
        sources = [grabcut_mask > 0.5, gradcam_mask > 0.5]
        if sam_mask is not None:
            sources.append(sam_mask > 0.5)

        agreement = np.mean(sources, axis=0).astype(np.float32)
        confidence = 0.6 * fg_score + 0.4 * agreement

        # ── Step 3: Binary foreground decision ───────────────────────
        fg_binary = (fg_score >= 0.5).astype(np.float32)

        # ── Step 4: Assign disease channels ──────────────────────────
        mask_5ch = self._assign_channels(fg_binary, gradcam_mask, class_name)

        # ── Step 5: Mark uncertain pixels ────────────────────────────
        uncertain = confidence < self.uncertainty_threshold
        # Reduce confidence for uncertain pixels
        confidence[uncertain] *= 0.5

        return mask_5ch, confidence

    def _assign_channels(
        self,
        fg_binary: np.ndarray,
        gradcam_heatmap: np.ndarray,
        class_name: str,
    ) -> np.ndarray:
        """Assign foreground pixels to 5-channel semantic categories.

        Uses DISEASE_CHANNEL_MAP to determine primary/secondary channels,
        and GradCAM intensity to distinguish healthy from diseased regions.
        """
        H, W = fg_binary.shape[:2]
        mask_5ch = np.zeros((H, W, NUM_SEG_CHANNELS), dtype=np.float32)

        # Background: everywhere that's NOT foreground
        mask_5ch[:, :, 0] = 1.0 - fg_binary

        if class_name not in DISEASE_CHANNEL_MAP:
            # Unknown class: all foreground → Ch1 (Healthy)
            mask_5ch[:, :, 1] = fg_binary
            return mask_5ch

        ch_info = DISEASE_CHANNEL_MAP[class_name]
        primary_ch = ch_info["primary"]
        secondary_ch = ch_info.get("secondary")

        if class_name == "Healthy":
            # All foreground → Healthy Plant Tissue
            mask_5ch[:, :, 1] = fg_binary
        else:
            # Use GradCAM intensity to split: high activation = disease, low = healthy
            grad_norm = gradcam_heatmap.copy()
            max_g = grad_norm.max()
            if max_g > 0:
                grad_norm = grad_norm / max_g

            # Disease regions: high GradCAM activation within foreground
            disease_region = fg_binary * (grad_norm > 0.4).astype(np.float32)
            healthy_region = fg_binary * (1.0 - disease_region)

            # Primary disease channel
            mask_5ch[:, :, primary_ch] = disease_region

            # Secondary channel (if exists)
            if secondary_ch is not None:
                # Split disease region: higher activation → primary, moderate → secondary
                high_act = fg_binary * (grad_norm > 0.6).astype(np.float32)
                mod_act = disease_region * (1.0 - high_act)
                mask_5ch[:, :, primary_ch] = high_act
                mask_5ch[:, :, secondary_ch] = mod_act

            # Healthy tissue: foreground minus disease
            mask_5ch[:, :, 1] = healthy_region

        # Ensure mask sums to 1.0 per pixel (softmax-like normalization)
        row_sums = mask_5ch.sum(axis=2, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-8)
        mask_5ch = mask_5ch / row_sums

        return mask_5ch

    def _ensure_shape(self, arr: np.ndarray, H: int, W: int) -> np.ndarray:
        """Resize mask to target (H, W) if needed."""
        if arr.shape[:2] != (H, W):
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
        return arr.astype(np.float32)

    def combine_for_split(
        self,
        split_name: str,
        grabcut_dir: Path,
        gradcam_dir: Path,
        sam_dir: Path | None,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Combine masks for an entire dataset split.

        Reads individual method masks from their directories, produces
        5-channel masks + confidence maps, saves to output_dir.

        Returns stats dict.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        stats: dict[str, Any] = {
            "total": 0, "success": 0, "failed": 0,
            "sam_available": sam_dir is not None,
        }

        for class_dir in sorted(grabcut_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            mask_out = output_dir / class_name
            mask_out.mkdir(parents=True, exist_ok=True)

            for gc_path in sorted(class_dir.glob("*_grabcut.npy")):
                stem = gc_path.stem.replace("_grabcut", "")
                stats["total"] += 1

                try:
                    # Load GrabCut mask
                    gc_mask = np.load(str(gc_path))

                    # Load GradCAM mask
                    grad_path = gradcam_dir / class_name / f"{stem}_gradcam_mask.npy"
                    if not grad_path.exists():
                        grad_path = gradcam_dir / class_name / f"{stem}_gradcam.npy"
                    grad_mask = np.load(str(grad_path)) if grad_path.exists() else \
                        np.zeros_like(gc_mask)

                    # Load SAM mask (optional)
                    sam_mask = None
                    if sam_dir is not None:
                        sam_path = sam_dir / class_name / f"{stem}_sam.npy"
                        if sam_path.exists():
                            sam_mask = np.load(str(sam_path))

                    # Combine
                    mask_5ch, confidence = self.combine(
                        gc_mask, grad_mask, sam_mask, class_name
                    )

                    # Save
                    np.save(str(mask_out / f"{stem}_mask.npy"), mask_5ch)
                    np.save(str(mask_out / f"{stem}_conf.npy"), confidence)
                    stats["success"] += 1

                except Exception as e:
                    logger.error(f"Combine failed for {stem} ({class_name}): {e}")
                    stats["failed"] += 1

        logger.info(
            f"Mask combination ({split_name}): "
            f"{stats['success']}/{stats['total']} succeeded"
        )
        return stats
