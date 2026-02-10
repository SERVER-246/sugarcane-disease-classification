"""
V2_segmentation/pseudo_labels/iterative_refiner.py
===================================================
Post-Phase-A self-training refinement of pseudo-labels.

After Phase A trains the segmentation head (on frozen backbone), the trained
model's own predictions are compared against the original pseudo-labels:
  - Where they agree → reinforce confidence (potentially B→A upgrade)
  - Where they disagree → flag for review

MAX_REFINEMENT_ROUNDS = 3
CONVERGENCE CHECK: If pixel change < 15% → converged
DIVERGENCE GUARD: If round N changes > round N-1 changes → stop, use N-1

Per Section 5.2 Layer 6 of the sprint plan.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


from V2_segmentation.config import (
    DEVICE, IMG_SIZE, MAX_REFINEMENT_ROUNDS, NUM_SEG_CHANNELS,
    REFINEMENT_CONVERGENCE_PCT,
)
from V2_segmentation.pseudo_labels.mask_quality_scorer import MaskQualityScorer

logger = logging.getLogger(__name__)


class IterativeRefiner:
    """Iteratively refine pseudo-labels using trained model predictions.

    After each Phase A training, run this to:
      1. Compare model predictions vs current pseudo-labels
      2. Where model agrees with high confidence → reinforce
      3. Where model disagrees → flag or update
      4. Re-score quality, potentially upgrade tiers

    Parameters
    ----------
    max_rounds : int
        Maximum refinement rounds (default 3).
    convergence_pct : float
        If pixel changes < this percentage, stop (default 15.0).
    agreement_threshold : float
        Model prediction confidence needed to reinforce label (default 0.7).
    """

    def __init__(
        self,
        max_rounds: int = MAX_REFINEMENT_ROUNDS,
        convergence_pct: float = REFINEMENT_CONVERGENCE_PCT,
        agreement_threshold: float = 0.7,
    ) -> None:
        self.max_rounds = max_rounds
        self.convergence_pct = convergence_pct
        self.agreement_threshold = agreement_threshold
        self.scorer = MaskQualityScorer()
        self.drift_log: list[dict[str, Any]] = []

    def refine_round(
        self,
        model: torch.nn.Module,
        masks_dir: Path,
        split_dir: Path,
        output_dir: Path,
        round_num: int,
    ) -> dict[str, Any]:
        """Run one round of iterative refinement.

        Parameters
        ----------
        model : nn.Module
            Trained V2 DualHeadModel (after Phase A).
        masks_dir : Path
            Current pseudo-label masks directory.
        split_dir : Path
            Image directory (e.g. split_dataset/train/).
        output_dir : Path
            Where to save refined masks.
        round_num : int
            Current round number (1-indexed).

        Returns
        -------
        dict with stats: total_pixels, changed_pixels, drift_pct,
             upgrades, downgrades, flagged.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        model.eval()

        total_pixels = 0
        changed_pixels = 0
        upgrades = 0
        downgrades = 0
        flagged_samples: list[dict[str, Any]] = []

        for class_dir in sorted(masks_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            out_class = output_dir / class_name
            out_class.mkdir(parents=True, exist_ok=True)

            for mask_path in sorted(class_dir.glob("*_mask.npy")):
                stem = mask_path.stem.replace("_mask", "")

                # Load current pseudo-label
                current_mask = np.load(str(mask_path))  # (H, W, 5)

                # Find original image
                img_path = self._find_image(split_dir / class_name, stem)
                if img_path is None:
                    # Copy as-is
                    np.save(str(out_class / f"{stem}_mask.npy"), current_mask)
                    continue

                # Get model prediction
                pred_mask = self._predict(model, img_path)  # (H, W, 5)
                if pred_mask is None:
                    np.save(str(out_class / f"{stem}_mask.npy"), current_mask)
                    continue

                # Compare: pixel-wise agreement
                n_pixels = current_mask.shape[0] * current_mask.shape[1]
                total_pixels += n_pixels

                current_class = current_mask.argmax(axis=2)
                pred_class = pred_mask.argmax(axis=2)
                pred_conf = pred_mask.max(axis=2)

                agree = current_class == pred_class
                disagree = ~agree

                # ── Update logic ──────────────────────────────────────
                refined_mask = current_mask.copy()

                # Where model agrees with high confidence → reinforce
                reinforce = agree & (pred_conf > self.agreement_threshold)
                for ch in range(NUM_SEG_CHANNELS):
                    ch_mask = (pred_class == ch) & reinforce
                    refined_mask[ch_mask, :] = 0.0
                    refined_mask[ch_mask, ch] = 1.0

                # Where model disagrees with high confidence → update
                high_conf_disagree = disagree & (pred_conf > 0.8)
                for ch in range(NUM_SEG_CHANNELS):
                    ch_mask = (pred_class == ch) & high_conf_disagree
                    refined_mask[ch_mask, :] = 0.0
                    refined_mask[ch_mask, ch] = 1.0

                # Count changes
                refined_class = refined_mask.argmax(axis=2)
                n_changed = int((current_class != refined_class).sum())
                changed_pixels += n_changed

                # Flag ambiguous cases (moderate disagreement)
                ambiguous = disagree & (pred_conf > 0.5) & (pred_conf <= 0.8)
                n_ambiguous = int(ambiguous.sum())
                if n_ambiguous > n_pixels * 0.1:
                    flagged_samples.append({
                        "class": class_name,
                        "stem": stem,
                        "ambiguous_pct": round(n_ambiguous / n_pixels * 100, 2),
                    })

                # Save refined mask
                np.save(str(out_class / f"{stem}_mask.npy"), refined_mask)

                # Re-score and update tier
                conf = refined_mask.max(axis=2)
                result = self.scorer.score(refined_mask, conf)
                old_tier_path = masks_dir / class_name / f"{stem}_tier.txt"
                old_tier = "B"
                if old_tier_path.exists():
                    old_tier = old_tier_path.read_text().strip()
                new_tier = result["tier"]

                if new_tier < old_tier:  # "A" < "B" in string comparison
                    upgrades += 1
                elif new_tier > old_tier:
                    downgrades += 1

                # Save new tier
                (out_class / f"{stem}_tier.txt").write_text(new_tier)
                # Save new confidence
                np.save(str(out_class / f"{stem}_conf.npy"), conf)

        # Compute drift
        drift_pct = (changed_pixels / max(total_pixels, 1)) * 100

        round_stats = {
            "round": round_num,
            "total_pixels": total_pixels,
            "changed_pixels": changed_pixels,
            "drift_pct": round(drift_pct, 2),
            "upgrades": upgrades,
            "downgrades": downgrades,
            "flagged_count": len(flagged_samples),
        }
        self.drift_log.append(round_stats)

        logger.info(
            f"Refinement round {round_num}: drift={drift_pct:.2f}%, "
            f"upgrades={upgrades}, downgrades={downgrades}, "
            f"flagged={len(flagged_samples)}"
        )

        # Save flagged samples
        if flagged_samples:
            flagged_path = output_dir / f"flagged_round{round_num}.json"
            with open(flagged_path, "w") as f:
                json.dump(flagged_samples, f, indent=2)

        return round_stats

    def refine_all(
        self,
        model: torch.nn.Module,
        initial_masks_dir: Path,
        split_dir: Path,
        output_base: Path,
    ) -> dict[str, Any]:
        """Run all refinement rounds with convergence/divergence checks.

        Returns final stats including all round results.
        """
        current_masks = initial_masks_dir
        self.drift_log = []

        for round_num in range(1, self.max_rounds + 1):
            output_dir = output_base / f"round_{round_num}"
            stats = self.refine_round(
                model, current_masks, split_dir, output_dir, round_num
            )

            drift = stats["drift_pct"]

            # ── Convergence check ─────────────────────────────────────
            if drift < self.convergence_pct:
                logger.info(
                    f"Converged at round {round_num} (drift={drift:.2f}% "
                    f"< {self.convergence_pct}%)"
                )
                break

            # ── Oscillation blocking gate (round ≥ 3) ────────────────
            if round_num >= 3:
                prev_drift = self.drift_log[-2]["drift_pct"]
                if drift > prev_drift:
                    logger.warning(
                        f"DIVERGENCE at round {round_num}: "
                        f"drift={drift:.2f}% > prev={prev_drift:.2f}%. "
                        f"Using round {round_num - 1} labels."
                    )
                    # Revert to previous round
                    current_masks = output_base / f"round_{round_num - 1}"
                    break

                if drift > self.convergence_pct:
                    logger.error(
                        f"BLOCKING: Round {round_num} drift={drift:.2f}% "
                        f"> {self.convergence_pct}%. Refinement NOT converging."
                    )
                    break

            current_masks = output_dir

        # Save drift log
        drift_path = output_base / "refinement_drift.json"
        with open(drift_path, "w") as f:
            json.dump(self.drift_log, f, indent=2)

        return {
            "rounds_completed": len(self.drift_log),
            "final_drift_pct": self.drift_log[-1]["drift_pct"] if self.drift_log else 0,
            "final_masks_dir": str(current_masks),
            "drift_log": self.drift_log,
        }

    def _predict(
        self,
        model: torch.nn.Module,
        image_path: Path,
    ) -> np.ndarray | None:
        """Run model inference on a single image, return seg mask."""
        try:
            from PIL import Image
            from torchvision.transforms import functional as TF

            img = Image.open(str(image_path)).convert("RGB")
            img_t = TF.resize(img, [IMG_SIZE, IMG_SIZE])  # type: ignore[arg-type]
            img_t = TF.to_tensor(img_t)  # type: ignore[arg-type]
            img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_t = img_t.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(img_t)
                seg_logits = outputs.get("seg_logits")
                if seg_logits is None:
                    return None
                seg_probs = F.softmax(seg_logits, dim=1)  # (1, 5, H', W')
                seg_probs = F.interpolate(
                    seg_probs, size=(IMG_SIZE, IMG_SIZE),
                    mode="bilinear", align_corners=False,
                )
                # (H, W, 5)
                return seg_probs[0].permute(1, 2, 0).cpu().numpy()
        except Exception as e:
            logger.warning(f"Prediction failed for {image_path}: {e}")
            return None

    def _find_image(self, class_dir: Path, stem: str) -> Path | None:
        """Find image file by stem in a class directory."""
        if not class_dir.exists():
            return None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = class_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None
