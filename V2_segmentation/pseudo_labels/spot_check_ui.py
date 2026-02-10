"""
V2_segmentation/pseudo_labels/spot_check_ui.py
===============================================
Human spot-check tool for pseudo-label quality assurance.

Displays original image + generated mask overlay + quality scores and allows
human annotation: Accept / Marginal / Reject.

Per Section 5.2 Layer 4:
  - Sample: 10 images × 13 classes × 3 tiers = 390 images
  - TIER_A acceptance ≥ 90% required
  - TIER_B acceptance ≥ 70% required
  - If TIER_C > 30% of dataset → HALT
  - This gate is BLOCKING — training cannot proceed until passed
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from V2_segmentation.config import (
    CLASS_NAMES, SEG_MASKS_DIR,
    SPLIT_DIR, RUN_SEED,
)

logger = logging.getLogger(__name__)

# Channel → overlay color (BGR)
CHANNEL_COLORS = {
    0: (50, 50, 50),       # Background: dark grey
    1: (0, 180, 0),        # Healthy: green
    2: (255, 100, 0),      # Structural: orange
    3: (0, 100, 255),      # Surface: red-blue
    4: (0, 0, 200),        # Degradation: red
}


def _create_overlay(
    image_bgr: np.ndarray,
    mask_5ch: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Create a color overlay of the 5-channel mask on the original image."""
    _h, _w = image_bgr.shape[:2]
    overlay = image_bgr.copy()

    # Get dominant channel per pixel
    dominant = mask_5ch.argmax(axis=2)

    for ch_idx, color in CHANNEL_COLORS.items():
        region = dominant == ch_idx
        if ch_idx == 0:
            continue  # skip background
        overlay[region] = (
            alpha * np.array(color) + (1 - alpha) * overlay[region]
        ).astype(np.uint8)

    return overlay


def sample_images_for_review(
    masks_dir: Path,
    tier_dir: Path,
    n_per_class_per_tier: int = 10,
    seed: int = RUN_SEED,
) -> list[dict[str, Any]]:
    """Sample images for human spot-check review.

    Parameters
    ----------
    masks_dir : Path
        Directory with per-class *_mask.npy.
    tier_dir : Path
        Directory with per-class *_tier.txt.
    n_per_class_per_tier : int
        Images to sample per class per tier.
    seed : int
        Random seed.

    Returns
    -------
    List of dicts: {class_name, stem, tier, mask_path, tier_path}
    """
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []

    for class_name in CLASS_NAMES:
        class_masks = masks_dir / class_name
        class_tiers = tier_dir / class_name
        if not class_masks.exists():
            continue

        # Group by tier
        by_tier: dict[str, list[str]] = {"A": [], "B": [], "C": []}
        for mask_path in sorted(class_masks.glob("*_mask.npy")):
            stem = mask_path.stem.replace("_mask", "")
            tier_path = class_tiers / f"{stem}_tier.txt"
            tier = tier_path.read_text().strip() if tier_path.exists() else "B"
            by_tier.setdefault(tier, []).append(stem)

        # Sample from each tier
        for tier, stems in by_tier.items():
            selected = rng.sample(stems, min(n_per_class_per_tier, len(stems)))
            for stem in selected:
                samples.append({
                    "class_name": class_name,
                    "stem": stem,
                    "tier": tier,
                    "mask_path": str(class_masks / f"{stem}_mask.npy"),
                    "tier_path": str(class_tiers / f"{stem}_tier.txt"),
                })

    logger.info(f"Sampled {len(samples)} images for spot-check review")
    return samples


class SpotCheckUI:
    """Interactive spot-check tool for pseudo-label validation.

    Supports two modes:
      - ``interactive``: matplotlib-based GUI (blocks until user input)
      - ``batch``: auto-save overlays to disk for async review
    """

    def __init__(
        self,
        split_dir: Path = SPLIT_DIR / "train",
        masks_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.split_dir = split_dir
        self.masks_dir = masks_dir or (SEG_MASKS_DIR / "combined" / "train")
        self.output_dir = output_dir or (SEG_MASKS_DIR / "audit")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict[str, Any]] = []

    def run_batch_review(
        self,
        samples: list[dict[str, Any]],
    ) -> Path:
        """Generate overlay images for batch (offline) review.

        Saves overlay PNGs and a review template JSON that can be
        filled in by a human reviewer.

        Returns path to review template JSON.
        """
        review_dir = self.output_dir / "spot_check_overlays"
        review_dir.mkdir(parents=True, exist_ok=True)

        template: list[dict[str, Any]] = []

        for sample in samples:
            class_name = sample["class_name"]
            stem = sample["stem"]
            tier = sample["tier"]

            # Find original image
            img_path = self._find_original_image(class_name, stem)
            if img_path is None:
                logger.warning(f"Original image not found for {class_name}/{stem}")
                continue

            # Load mask
            mask_path = Path(sample["mask_path"])
            if not mask_path.exists():
                continue
            mask_5ch = np.load(str(mask_path))

            # Create overlay
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_bgr = cv2.resize(img_bgr, (mask_5ch.shape[1], mask_5ch.shape[0]))
            overlay = _create_overlay(img_bgr, mask_5ch)

            # Save overlay
            fname = f"{class_name}__{stem}__tier{tier}.png"
            cv2.imwrite(str(review_dir / fname), overlay)

            template.append({
                "class_name": class_name,
                "stem": stem,
                "tier": tier,
                "overlay_file": fname,
                "verdict": "PENDING",  # Human fills: ACCEPT / MARGINAL / REJECT
            })

        # Save review template
        template_path = self.output_dir / "spot_check_template.json"
        with open(template_path, "w") as f:
            json.dump(template, f, indent=2)

        logger.info(
            f"Batch review prepared: {len(template)} overlays in {review_dir}, "
            f"template at {template_path}"
        )
        return template_path

    def load_review_results(self, results_path: Path | None = None) -> dict[str, Any]:
        """Load completed review results and check acceptance rates.

        Returns
        -------
        dict with acceptance rates per tier and pass/fail status.
        """
        results_path = results_path or (self.output_dir / "spot_check_results.json")
        if not results_path.exists():
            logger.error(f"Review results not found at {results_path}")
            return {"status": "MISSING", "message": "No results file found"}

        with open(results_path) as f:
            results = json.load(f)

        # Compute per-tier acceptance
        tier_stats: dict[str, dict[str, int]] = {}
        for r in results:
            tier = r.get("tier", "?")
            verdict = r.get("verdict", "PENDING")
            if tier not in tier_stats:
                tier_stats[tier] = {"total": 0, "accept": 0, "marginal": 0, "reject": 0}
            tier_stats[tier]["total"] += 1
            if verdict == "ACCEPT":
                tier_stats[tier]["accept"] += 1
            elif verdict == "MARGINAL":
                tier_stats[tier]["marginal"] += 1
            elif verdict == "REJECT":
                tier_stats[tier]["reject"] += 1

        # Check thresholds
        gates: dict[str, Any] = {}
        overall_pass = True

        for tier, stats in tier_stats.items():
            total = stats["total"]
            if total == 0:
                continue
            accept_rate = (stats["accept"] + 0.5 * stats["marginal"]) / total

            if tier == "A":
                threshold = 0.90
            elif tier == "B":
                threshold = 0.70
            else:
                threshold = 0.0  # No threshold for C

            passed = accept_rate >= threshold
            if tier in ("A", "B") and not passed:
                overall_pass = False

            gates[tier] = {
                "accept_rate": round(accept_rate, 3),
                "threshold": threshold,
                "passed": passed,
                **stats,
            }

        result = {
            "status": "PASS" if overall_pass else "FAIL",
            "tier_gates": gates,
            "total_reviewed": len(results),
        }

        logger.info(f"Spot-check gate: {result['status']} — {gates}")
        return result

    def _find_original_image(self, class_name: str, stem: str) -> Path | None:
        """Find the original image file in the split directory."""
        class_dir = self.split_dir / class_name
        if not class_dir.exists():
            return None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = class_dir / f"{stem}{ext}"
            if p.exists():
                return p
        # Fuzzy search: stem might have been modified
        for p in class_dir.iterdir():
            if p.stem == stem:
                return p
        return None
