"""
V2_segmentation/validation/calibrate_gate.py
==============================================
Per-class validation gate threshold calibration from gold-labeled masks.

Per Phase 0.5 of the sprint plan:
  1. Load gold-labeled masks (200 images, ~15/class)
  2. Compute plant_ratio distribution per class
  3. Set per-class threshold = 5th percentile of gold distribution
  4. Validate: gate must pass ≥98% of real sugarcane images on gold set
  5. Store calibrated thresholds
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from V2_segmentation.config import (
    CLASS_NAMES, GATE_THRESHOLDS, GOLD_LABELS_DIR, NUM_SEG_CHANNELS,
)

logger = logging.getLogger(__name__)


class CalibrateGate:
    """Calibrate per-class validation gate thresholds using gold labels.

    Requires gold-labeled masks in gold_labels/{class_name}/ directory.
    Each mask: *_mask.png or *_mask.npy with 5 channels.
    """

    def __init__(
        self,
        gold_dir: Path | None = None,
        target_pass_rate: float = 0.98,
        percentile: float = 5.0,
    ) -> None:
        """
        Parameters
        ----------
        gold_dir : Path
            Directory with gold-labeled masks.
        target_pass_rate : float
            Minimum pass rate on gold set (default 98%).
        percentile : float
            Percentile of plant_ratio distribution to use as threshold.
        """
        self.gold_dir = gold_dir or GOLD_LABELS_DIR
        self.target_pass_rate = target_pass_rate
        self.percentile = percentile

    def compute_plant_ratios(self) -> dict[str, list[float]]:
        """Compute plant_ratio for each gold-labeled image.

        Returns dict: class_name → list of plant_ratio values.
        """
        ratios: dict[str, list[float]] = {}

        for class_name in CLASS_NAMES:
            class_dir = self.gold_dir / class_name
            if not class_dir.exists():
                logger.warning(f"No gold labels for {class_name}")
                ratios[class_name] = []
                continue

            class_ratios: list[float] = []
            for mask_path in sorted(class_dir.glob("*_mask.*")):
                try:
                    if mask_path.suffix == ".npy":
                        mask = np.load(str(mask_path))
                    else:
                        import cv2
                        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                        if mask_img is None:
                            continue
                        # Convert to normalized float
                        mask = mask_img.astype(np.float32) / 255.0
                        if mask.ndim == 2:
                            # Binary mask: treat as single-channel
                            plant_ratio = float((mask > 0.5).mean())
                            class_ratios.append(plant_ratio)
                            continue

                    # 5-channel mask: plant = Ch1 + Ch2 + Ch3 + Ch4
                    if mask.ndim == 3 and mask.shape[-1] >= NUM_SEG_CHANNELS:
                        plant = (mask[:, :, 1:NUM_SEG_CHANNELS] > 0.5).any(axis=2)
                    elif mask.ndim == 3 and mask.shape[0] >= NUM_SEG_CHANNELS:
                        plant = (mask[1:NUM_SEG_CHANNELS] > 0.5).any(axis=0)
                    else:
                        plant = mask > 0.5

                    plant_ratio = float(plant.mean())
                    class_ratios.append(plant_ratio)

                except Exception as e:
                    logger.warning(f"Failed to process gold mask {mask_path}: {e}")

            ratios[class_name] = class_ratios

        return ratios

    def calibrate(self) -> dict[str, Any]:
        """Run full calibration.

        Returns calibrated thresholds + statistics.
        """
        ratios = self.compute_plant_ratios()

        calibrated: dict[str, float] = {}
        per_class_stats: dict[str, dict[str, Any]] = {}

        for class_name in CLASS_NAMES:
            values = ratios.get(class_name, [])

            if len(values) < 3:
                # Too few gold samples: use default
                threshold = GATE_THRESHOLDS.get(class_name, 0.15)
                per_class_stats[class_name] = {
                    "n_gold": len(values),
                    "threshold": threshold,
                    "source": "default (insufficient gold)",
                }
            else:
                arr = np.array(values)
                threshold = float(np.percentile(arr, self.percentile))
                # Floor at a sensible minimum
                threshold = max(threshold, 0.02)

                per_class_stats[class_name] = {
                    "n_gold": len(values),
                    "threshold": round(threshold, 4),
                    "p5": round(float(np.percentile(arr, 5)), 4),
                    "p25": round(float(np.percentile(arr, 25)), 4),
                    "p50": round(float(np.percentile(arr, 50)), 4),
                    "mean": round(float(arr.mean()), 4),
                    "min": round(float(arr.min()), 4),
                    "max": round(float(arr.max()), 4),
                    "source": "calibrated",
                }

            calibrated[class_name] = threshold

        # Validate: check pass rate on gold set
        pass_counts = {"pass": 0, "fail": 0}
        for class_name, values in ratios.items():
            t = calibrated.get(class_name, 0.15)
            for v in values:
                if v >= t:
                    pass_counts["pass"] += 1
                else:
                    pass_counts["fail"] += 1

        total = pass_counts["pass"] + pass_counts["fail"]
        pass_rate = pass_counts["pass"] / max(total, 1)

        result = {
            "status": "PASS" if pass_rate >= self.target_pass_rate else "FAIL",
            "pass_rate": round(pass_rate, 4),
            "target_pass_rate": self.target_pass_rate,
            "total_gold_images": total,
            "calibrated_thresholds": calibrated,
            "per_class_stats": per_class_stats,
        }

        # Save
        self._save_results(result)

        logger.info(
            f"Calibration: pass_rate={pass_rate:.4f} "
            f"(target={self.target_pass_rate:.4f}), "
            f"status={result['status']}"
        )
        return result

    def _save_results(self, result: dict[str, Any]) -> None:
        """Save calibration results and thresholds."""
        output_dir = self.gold_dir.parent / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full report
        report_path = output_dir / "calibration_report.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        # Just thresholds (for config update)
        thresh_path = output_dir / "calibrated_thresholds.json"
        with open(thresh_path, "w") as f:
            json.dump(result["calibrated_thresholds"], f, indent=2)

        logger.info(f"Calibration saved to {report_path}")
