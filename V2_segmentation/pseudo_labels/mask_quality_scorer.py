"""
V2_segmentation/pseudo_labels/mask_quality_scorer.py
=====================================================
Per-image quality scoring and tier assignment for pseudo-label masks.

Quality score formula (from Section 5.2, Layer 2):
  QUALITY = 0.4 × largest_component_ratio
          + 0.3 × (1 - uncertain_ratio)
          + 0.2 × min(1.0, plant_ratio / 0.2)
          + 0.1 × (1 - boundary_smoothness_penalty)

Tier assignment:
  quality ≥ 0.80 → TIER_A (full loss weight)
  quality ≥ 0.50 → TIER_B (0.5× loss weight)
  quality < 0.50 → TIER_C (excluded from training)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from V2_segmentation.config import (
    NUM_SEG_CHANNELS, TIER_A_THRESHOLD, TIER_B_THRESHOLD,
    UNCERTAINTY_THRESHOLD,
)

logger = logging.getLogger(__name__)


def compute_plant_ratio(mask_5ch: np.ndarray) -> float:
    """Fraction of pixels belonging to ANY plant channel (Ch1-Ch4)."""
    if mask_5ch.ndim == 3 and mask_5ch.shape[2] == NUM_SEG_CHANNELS:
        plant = mask_5ch[:, :, 1:].sum(axis=2)
    else:
        plant = mask_5ch
    return float((plant > 0.5).mean())


def compute_largest_component_ratio(fg_mask: np.ndarray) -> float:
    """Ratio of largest connected component to total foreground area."""
    binary = (fg_mask > 0.5).astype(np.uint8)
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    if n_labels <= 1:
        return 0.0  # no foreground

    # Skip label 0 (background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    total_fg = areas.sum()
    if total_fg == 0:
        return 0.0
    return float(areas.max() / total_fg)


def compute_boundary_smoothness_penalty(fg_mask: np.ndarray) -> float:
    """Boundary complexity penalty via isoperimetric quotient.

    Circularity = 4π × area / perimeter². Ranges [0, 1], 1 = perfect circle.
    Penalty = 1 - circularity (high for irregular shapes).
    """
    binary = (fg_mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.5  # neutral penalty

    # Largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    if perimeter == 0 or area == 0:
        return 0.5

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    circularity = min(circularity, 1.0)

    return float(1.0 - circularity)


def compute_uncertain_ratio(
    confidence: np.ndarray,
    threshold: float = UNCERTAINTY_THRESHOLD,
) -> float:
    """Fraction of pixels with confidence below the threshold."""
    return float((confidence < threshold).mean())


class MaskQualityScorer:
    """Score mask quality and assign tiers.

    Parameters
    ----------
    tier_a_threshold : float
        Minimum quality for TIER_A (default 0.80).
    tier_b_threshold : float
        Minimum quality for TIER_B (default 0.50).
    uncertainty_threshold : float
        Confidence threshold for uncertain pixel ratio.
    """

    def __init__(
        self,
        tier_a_threshold: float = TIER_A_THRESHOLD,
        tier_b_threshold: float = TIER_B_THRESHOLD,
        uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
    ) -> None:
        self.tier_a_threshold = tier_a_threshold
        self.tier_b_threshold = tier_b_threshold
        self.uncertainty_threshold = uncertainty_threshold

    def score(
        self,
        mask_5ch: np.ndarray,
        confidence: np.ndarray,
    ) -> dict[str, Any]:
        """Compute quality score and tier for a single mask.

        Parameters
        ----------
        mask_5ch : (H, W, 5) float32
        confidence : (H, W) float32

        Returns
        -------
        dict with keys: quality_score, tier, plant_ratio,
             largest_component_ratio, uncertain_ratio,
             boundary_penalty, component_count
        """
        # Foreground = any non-background channel
        fg = mask_5ch[:, :, 1:].sum(axis=2)
        fg_binary = (fg > 0.5).astype(np.float32)

        plant_ratio = compute_plant_ratio(mask_5ch)
        lcr = compute_largest_component_ratio(fg_binary)
        uncertain_ratio = compute_uncertain_ratio(confidence, self.uncertainty_threshold)
        boundary_penalty = compute_boundary_smoothness_penalty(fg_binary)

        # Count connected components
        binary_uint8 = (fg_binary > 0.5).astype(np.uint8)
        n_components = cv2.connectedComponentsWithStats(binary_uint8)[0] - 1

        # Quality formula from Section 5.2
        quality = (
            0.4 * lcr
            + 0.3 * (1.0 - uncertain_ratio)
            + 0.2 * min(1.0, plant_ratio / 0.2)
            + 0.1 * (1.0 - boundary_penalty)
        )
        quality = float(np.clip(quality, 0.0, 1.0))

        # Tier assignment
        if quality >= self.tier_a_threshold:
            tier = "A"
        elif quality >= self.tier_b_threshold:
            tier = "B"
        else:
            tier = "C"

        return {
            "quality_score": quality,
            "tier": tier,
            "plant_ratio": plant_ratio,
            "largest_component_ratio": lcr,
            "uncertain_ratio": uncertain_ratio,
            "boundary_penalty": boundary_penalty,
            "component_count": n_components,
        }

    def score_split(
        self,
        masks_dir: Path,
        confidence_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Score all masks in a split directory.

        Parameters
        ----------
        masks_dir : Path
            Directory with per-class mask .npy files (*_mask.npy).
        confidence_dir : Path or None
            Directory with confidence .npy files (*_conf.npy). If None,
            uses default confidence of 0.8 everywhere.
        output_dir : Path or None
            If given, saves per-image tier text files and summary CSV.

        Returns
        -------
        dict with overall stats and per-class distributions.
        """
        all_scores: list[dict[str, Any]] = []
        tier_counts = {"A": 0, "B": 0, "C": 0}
        per_class: dict[str, dict[str, int]] = {}

        # Count total masks for progress bar
        all_masks = []
        for class_dir in sorted(masks_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for mask_path in sorted(class_dir.glob("*_mask.npy")):
                all_masks.append((class_dir.name, mask_path))

        # Process with progress bar
        pbar = tqdm(all_masks, desc="Quality scoring", unit="mask")
        for class_name, mask_path in pbar:
            stem = mask_path.stem.replace("_mask", "")
            pbar.set_postfix({"class": class_name[:12], "file": stem[:15]})

            if class_name not in per_class:
                per_class[class_name] = {"A": 0, "B": 0, "C": 0}

            mask_5ch = np.load(str(mask_path))

            # Load or generate default confidence
            conf_path = None
            if confidence_dir is not None:
                conf_path = confidence_dir / class_name / f"{stem}_conf.npy"
            if conf_path is not None and conf_path.exists():
                confidence = np.load(str(conf_path))
            else:
                # Default: use mask channel max as proxy
                confidence = mask_5ch.max(axis=2)

            result = self.score(mask_5ch, confidence)
            result["class"] = class_name
            result["stem"] = stem
            all_scores.append(result)

            tier = result["tier"]
            tier_counts[tier] += 1
            per_class[class_name][tier] += 1

            # Save tier file
            if output_dir is not None:
                tier_dir = output_dir / class_name
                tier_dir.mkdir(parents=True, exist_ok=True)
                (tier_dir / f"{stem}_tier.txt").write_text(tier)

        total = sum(tier_counts.values())

        # Save CSV summary
        if output_dir is not None and all_scores:
            csv_path = output_dir / "quality_scores.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_scores[0].keys())
                writer.writeheader()
                writer.writerows(all_scores)

        stats = {
            "total": total,
            "tier_counts": tier_counts,
            "tier_pct": {
                k: round(v / max(total, 1) * 100, 1) for k, v in tier_counts.items()
            },
            "per_class": per_class,
            "mean_quality": float(np.mean([s["quality_score"] for s in all_scores]))
            if all_scores else 0.0,
        }

        logger.info(
            f"Quality scoring: {total} masks — "
            f"A={tier_counts['A']}, B={tier_counts['B']}, C={tier_counts['C']} "
            f"(mean quality={stats['mean_quality']:.3f})"
        )
        return stats
