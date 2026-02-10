"""
V2_segmentation/pseudo_labels/class_sanity_checker.py
=====================================================
Disease-specific validation rules for pseudo-label masks.

Per Section 5.2 Layer 3 of the sprint plan, each disease class has domain-
specific rules about which segmentation channels should be active. Masks that
violate their class rules are downgraded one tier (A→B, B→C).

Universal rule: plant_ratio > 0.08 for ALL classes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from V2_segmentation.config import (
    NUM_SEG_CHANNELS,
)

logger = logging.getLogger(__name__)

# Per-class sanity rules: each returns True if the mask is VALID for this class.
# plant_ratio = Ch1 + Ch2 + Ch3 + Ch4 (any plant channel)
# anomaly_ratio = Ch2 / total
# surface_ratio = Ch3 / total
# degradation_ratio = Ch4 / total
UNIVERSAL_MIN_PLANT_RATIO = 0.08

CLASS_RULES: dict[str, dict[str, float]] = {
    "Healthy":              {"min_plant": 0.40, "max_disease": 0.05},
    "Red_rot":              {"min_degradation": 0.03, "alt_min_surface": 0.03},
    "Brown_spot":           {"min_surface": 0.03},
    "Mosaic":               {"min_surface": 0.03},
    "Smut":                 {"min_anomaly": 0.02},
    "Grassy_shoot_disease": {"min_anomaly": 0.02},
    "Wilt":                 {"min_degradation": 0.05},
    "Pokkah_boeng":         {"min_anomaly": 0.02, "alt_min_degradation": 0.02},
    "Leaf_flecking":        {"min_surface": 0.02},
    "Leaf_scorching":       {"min_degradation": 0.03},
    "Ring_spot":            {"min_surface": 0.03},
    "Yellow_leaf_Disease":  {"min_degradation": 0.03},
    "Black_stripe":         {"min_surface": 0.03},
}


def _compute_ratios(mask_5ch: np.ndarray) -> dict[str, float]:
    """Compute channel ratios from a 5-channel mask."""
    total = mask_5ch.shape[0] * mask_5ch.shape[1]
    if total == 0:
        return {"plant": 0.0, "anomaly": 0.0, "surface": 0.0, "degradation": 0.0}

    # Binary per channel
    ch = {}
    for i in range(NUM_SEG_CHANNELS):
        ch[i] = (mask_5ch[:, :, i] > 0.5).sum() / total

    return {
        "plant": ch[1] + ch[2] + ch[3] + ch[4],
        "healthy": ch[1],
        "anomaly": ch[2],
        "surface": ch[3],
        "degradation": ch[4],
        "background": ch[0],
    }


class ClassSanityChecker:
    """Validate pseudo-labels against disease-specific channel rules.

    Masks violating rules are downgraded one tier.
    """

    def __init__(self) -> None:
        self.rules = dict(CLASS_RULES)

    def check(
        self,
        mask_5ch: np.ndarray,
        class_name: str,
        current_tier: str,
    ) -> tuple[str, list[str]]:
        """Check a single mask against class-specific rules.

        Parameters
        ----------
        mask_5ch : (H, W, 5) float32
        class_name : str
        current_tier : str  ("A", "B", or "C")

        Returns
        -------
        new_tier : str
            Possibly downgraded tier.
        violations : list[str]
            List of violated rules (empty if valid).
        """
        violations: list[str] = []
        ratios = _compute_ratios(mask_5ch)

        # ── Universal rule ───────────────────────────────────────────
        if ratios["plant"] < UNIVERSAL_MIN_PLANT_RATIO:
            violations.append(
                f"Universal: plant_ratio={ratios['plant']:.3f} < {UNIVERSAL_MIN_PLANT_RATIO}"
            )

        # ── Class-specific rules ─────────────────────────────────────
        if class_name in self.rules:
            rule = self.rules[class_name]

            # Minimum plant ratio (for Healthy)
            if "min_plant" in rule and ratios["plant"] < rule["min_plant"]:
                violations.append(
                    f"{class_name}: plant={ratios['plant']:.3f} < {rule['min_plant']}"
                )

            # Maximum disease signal (for Healthy)
            if "max_disease" in rule:
                disease = ratios["anomaly"] + ratios["surface"] + ratios["degradation"]
                if disease > rule["max_disease"]:
                    violations.append(
                        f"{class_name}: disease_total={disease:.3f} > {rule['max_disease']}"
                    )

            # Minimum anomaly ratio (Ch2)
            if "min_anomaly" in rule and ratios["anomaly"] < rule["min_anomaly"]:
                # Check alternative: some diseases accept either anomaly OR degradation
                alt_ok = False
                if "alt_min_degradation" in rule:
                    alt_ok = ratios["degradation"] >= rule["alt_min_degradation"]
                if not alt_ok:
                    violations.append(
                        f"{class_name}: anomaly={ratios['anomaly']:.3f} < {rule['min_anomaly']}"
                    )

            # Minimum surface ratio (Ch3)
            if "min_surface" in rule and ratios["surface"] < rule["min_surface"]:
                alt_ok = False
                if "alt_min_surface" in rule:
                    alt_ok = ratios["surface"] >= rule["alt_min_surface"]
                if not alt_ok:
                    violations.append(
                        f"{class_name}: surface={ratios['surface']:.3f} < {rule['min_surface']}"
                    )

            # Minimum degradation ratio (Ch4)
            if "min_degradation" in rule and ratios["degradation"] < rule["min_degradation"]:
                alt_ok = False
                if "alt_min_surface" in rule:
                    alt_ok = ratios["surface"] >= rule["alt_min_surface"]
                if not alt_ok:
                    violations.append(
                        f"{class_name}: degradation={ratios['degradation']:.3f} < {rule['min_degradation']}"
                    )

        # ── Downgrade if violations found ────────────────────────────
        new_tier = current_tier
        if violations:
            downgrade_map = {"A": "B", "B": "C", "C": "C"}
            new_tier = downgrade_map[current_tier]
            if new_tier != current_tier:
                logger.debug(
                    f"Downgraded {class_name} mask: {current_tier}→{new_tier} "
                    f"({len(violations)} violations)"
                )

        return new_tier, violations

    def check_split(
        self,
        masks_dir: Path,
        tier_dir: Path,
    ) -> dict[str, Any]:
        """Check all masks in a split against class rules.

        Parameters
        ----------
        masks_dir : Path
            Directory with per-class *_mask.npy files.
        tier_dir : Path
            Directory with per-class *_tier.txt files (will be updated).

        Returns
        -------
        dict with violation stats.
        """
        from pathlib import Path
        masks_dir = Path(masks_dir)
        tier_dir = Path(tier_dir)

        stats: dict[str, Any] = {
            "total": 0, "violations": 0, "downgrades": 0,
            "per_class_violations": {},
        }

        for class_dir in sorted(masks_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            class_violations = 0

            for mask_path in sorted(class_dir.glob("*_mask.npy")):
                stem = mask_path.stem.replace("_mask", "")
                stats["total"] += 1

                mask_5ch = np.load(str(mask_path))

                # Read current tier
                tier_path = tier_dir / class_name / f"{stem}_tier.txt"
                current_tier = "B"  # default if missing
                if tier_path.exists():
                    current_tier = tier_path.read_text().strip()

                new_tier, violations = self.check(mask_5ch, class_name, current_tier)

                if violations:
                    stats["violations"] += 1
                    class_violations += 1

                if new_tier != current_tier:
                    stats["downgrades"] += 1
                    tier_path.parent.mkdir(parents=True, exist_ok=True)
                    tier_path.write_text(new_tier)

            stats["per_class_violations"][class_name] = class_violations

        logger.info(
            f"Sanity check: {stats['violations']}/{stats['total']} masks had violations, "
            f"{stats['downgrades']} downgrades applied"
        )
        return stats
