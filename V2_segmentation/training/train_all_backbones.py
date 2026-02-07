"""
V2_segmentation/training/train_all_backbones.py
=================================================
Wave orchestrator: trains all 15 backbones sequentially, grouped by memory tier.

Wave order (lowest→highest VRAM):
  Wave 1 — LIGHT  (~45 min each): EfficientNetV4 → DenseNetHybrid → InceptionV4 → MobileOne → GhostNetV2
  Wave 2 — MEDIUM (~1.5h each):   ConvNeXt → ResNetMish → RegNet
  Wave 3 — HIGH   (~2.5h each):   CSPDarkNet → DynamicConvNet
  Wave 4 — HEAVY  (~4h each):     DeiTStyle → SwinTransformer → MaxViT → CoAtNet → ViTHybrid

Between waves: GPU cleanup + free memory assertion.
OOM recovery: halve BS, double grad_accum, retry from checkpoint.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import torch

from V2_segmentation.config import (
    BACKBONE_PROFILES,
    CKPT_V2_DIR,
    MemoryTier,
)
from V2_segmentation.training.memory_manager import MemoryManager
from V2_segmentation.training.train_v2_backbone import train_v2_backbone

logger = logging.getLogger(__name__)


# ============================================================================
#  Wave definitions
# ============================================================================

WAVES: list[dict[str, Any]] = [
    {
        "name": "Wave 1 — LIGHT",
        "tier": MemoryTier.LIGHT,
        "backbones": [
            "CustomEfficientNetV4",
            "CustomDenseNetHybrid",
            "CustomInceptionV4",
            "CustomMobileOne",
            "CustomGhostNetV2",
        ],
    },
    {
        "name": "Wave 2 — MEDIUM",
        "tier": MemoryTier.MEDIUM,
        "backbones": [
            "CustomConvNeXt",
            "CustomResNetMish",
            "CustomRegNet",
        ],
    },
    {
        "name": "Wave 3 — HIGH",
        "tier": MemoryTier.HIGH,
        "backbones": [
            "CustomCSPDarkNet",
            "CustomDynamicConvNet",
        ],
    },
    {
        "name": "Wave 4 — HEAVY",
        "tier": MemoryTier.HEAVY,
        "backbones": [
            "CustomDeiTStyle",
            "CustomSwinTransformer",
            "CustomMaxViT",
            "CustomCoAtNet",
            "CustomViTHybrid",
        ],
    },
]


# ============================================================================
#  GPU safety
# ============================================================================

def _gpu_cleanup() -> None:
    """Full GPU cleanup between backbone runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _assert_gpu_headroom(min_free_gb: float = 8.0) -> bool:
    """Assert that GPU has enough free memory before starting a new backbone.

    Returns True if safe to proceed.
    """
    if not torch.cuda.is_available():
        return True

    usage = MemoryManager.get_vram_usage()
    free = usage["free"]
    if free < min_free_gb:
        logger.warning(
            f"  ⚠ Low VRAM: {free:.2f} GB free (need {min_free_gb:.1f} GB). "
            f"Running emergency cleanup..."
        )
        _gpu_cleanup()
        usage = MemoryManager.get_vram_usage()
        free = usage["free"]
        if free < min_free_gb:
            logger.error(
                f"  ✗ VRAM still low after cleanup: {free:.2f} GB free. "
                f"Skipping backbone."
            )
            return False
    return True


# ============================================================================
#  Result tracking
# ============================================================================

class TrainingReport:
    """Accumulate and persist training results for all backbones."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.report_path = CKPT_V2_DIR / "training_report.json"

    def add(self, backbone_name: str, result: dict[str, Any]) -> None:
        self.results.append({
            "backbone": backbone_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result,
        })
        self._save()

    def add_skip(self, backbone_name: str, reason: str) -> None:
        self.results.append({
            "backbone": backbone_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "skipped": True,
            "skip_reason": reason,
        })
        self._save()

    def _save(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable objects
        def _make_serializable(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_serializable(v) for v in obj]
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            try:
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
            except ImportError:
                pass
            return str(obj)

        with open(self.report_path, "w") as f:
            json.dump(_make_serializable(self.results), f, indent=2)

    def summary(self) -> str:
        """Print a summary table."""
        lines = ["\n" + "=" * 70, "  V2 Training Summary", "=" * 70]
        for r in self.results:
            name = r.get("backbone", "?")
            if r.get("skipped"):
                lines.append(f"  ✗ {name:30s} — SKIPPED: {r.get('skip_reason')}")
            elif r.get("rollback"):
                lines.append(f"  ↩ {name:30s} — ROLLBACK")
            else:
                acc = 0.0
                miou = 0.0
                t = r.get("total_time_s", 0)
                phases = r.get("phases", {})
                # Use Phase C metrics if available, else Phase B
                final = phases.get("C", phases.get("B", {}))
                acc = final.get("cls_accuracy", 0)
                miou = final.get("mean_iou", 0)
                lines.append(
                    f"  ✓ {name:30s} — "
                    f"acc={acc:.4f}  mIoU={miou:.4f}  "
                    f"time={t / 60:.1f}min"
                )
        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
#  Main orchestrator
# ============================================================================

def train_all_backbones(
    waves: list[dict[str, Any]] | None = None,
    num_workers: int = 4,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> TrainingReport:
    """Train all 15 backbones in wave order.

    Parameters
    ----------
    waves : list or None
        Override wave definitions. If None, use default WAVES.
    num_workers : int
        DataLoader workers.
    dry_run : bool
        1 epoch per phase.
    skip_existing : bool
        If True, skip backbones that already have a final V2 checkpoint.

    Returns
    -------
    TrainingReport with all results.
    """
    t_global = time.time()
    wave_defs = waves or WAVES
    report = TrainingReport()

    total_backbones = sum(len(w["backbones"]) for w in wave_defs)
    completed = 0

    for wave in wave_defs:
        wave_name = wave["name"]
        logger.info(f"\n{'#'*70}")
        logger.info(f"  {wave_name}")
        logger.info(f"{'#'*70}\n")

        for backbone_name in wave["backbones"]:
            completed += 1
            logger.info(
                f"\n--- [{completed}/{total_backbones}] {backbone_name} "
                f"({wave_name}) ---\n"
            )

            # ── Skip if already trained ──────────────────────────────
            if skip_existing:
                final_ckpt = CKPT_V2_DIR / f"{backbone_name}_v2_final.pth"
                if final_ckpt.exists():
                    logger.info(f"  Skipping {backbone_name} — final checkpoint exists")
                    report.add_skip(backbone_name, "final checkpoint already exists")
                    continue

            # ── GPU headroom check ───────────────────────────────────
            _gpu_cleanup()
            min_free = _get_min_free_gb(wave["tier"])
            if not _assert_gpu_headroom(min_free):
                report.add_skip(backbone_name, f"insufficient VRAM (need {min_free}GB free)")
                continue

            # ── Train ────────────────────────────────────────────────
            try:
                result = train_v2_backbone(
                    backbone_name=backbone_name,
                    num_workers=num_workers,
                    dry_run=dry_run,
                )
                report.add(backbone_name, result)

            except torch.cuda.OutOfMemoryError:
                logger.error(f"  ✗ OOM for {backbone_name} — attempting recovery...")
                MemoryManager.emergency_cleanup()

                # Retry with halved batch size
                try:
                    result = _retry_with_reduced_memory(
                        backbone_name, num_workers, dry_run,
                    )
                    report.add(backbone_name, result)
                except Exception as e2:
                    logger.exception(f"  ✗ Retry failed for {backbone_name}: {e2}")
                    report.add_skip(backbone_name, f"OOM + retry failed: {e2}")

            except Exception as e:
                logger.exception(f"  ✗ Error training {backbone_name}: {e}")
                report.add_skip(backbone_name, str(e))
                _gpu_cleanup()

        logger.info(f"\n  {wave_name} complete.\n")

    elapsed = time.time() - t_global
    summary = report.summary()
    logger.info(summary)
    logger.info(f"\n  Total V2 training time: {elapsed / 3600:.1f} hours\n")

    return report


def _get_min_free_gb(tier: str) -> float:
    """Minimum free VRAM needed to start a backbone of this tier."""
    tier_mins = {
        MemoryTier.LIGHT: 4.0,
        MemoryTier.MEDIUM: 6.0,
        MemoryTier.HIGH: 10.0,
        MemoryTier.HEAVY: 14.0,
    }
    return tier_mins.get(tier, 8.0)


def _retry_with_reduced_memory(
    backbone_name: str,
    num_workers: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Retry training with halved batch size and doubled accumulation.

    Modifies BACKBONE_PROFILES temporarily.
    """
    profile = BACKBONE_PROFILES[backbone_name]
    original_bs = profile["batch_size"]
    original_accum = profile["grad_accum"]

    try:
        profile["batch_size"] = max(original_bs // 2, 1)
        profile["grad_accum"] = original_accum * 2
        logger.warning(
            f"  Retrying {backbone_name} with BS={profile['batch_size']}, "
            f"grad_accum={profile['grad_accum']}"
        )
        return train_v2_backbone(
            backbone_name=backbone_name,
            num_workers=num_workers,
            dry_run=dry_run,
        )
    finally:
        # Restore original profile
        profile["batch_size"] = original_bs
        profile["grad_accum"] = original_accum


# ============================================================================
#  CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train all V2 backbones in waves")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="1 epoch per phase (smoke test)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Don't skip backbones with existing checkpoints",
    )
    parser.add_argument(
        "--wave", type=int, default=None,
        help="Train only this wave number (1-4)",
    )
    parser.add_argument(
        "--backbone", type=str, default=None,
        help="Train only this specific backbone",
    )
    args = parser.parse_args()

    # Filter waves/backbones
    wave_defs = WAVES
    if args.wave is not None:
        if 1 <= args.wave <= len(WAVES):
            wave_defs = [WAVES[args.wave - 1]]
        else:
            parser.error(f"Wave must be 1-{len(WAVES)}")

    if args.backbone is not None:
        if args.backbone not in BACKBONE_PROFILES:
            parser.error(f"Unknown backbone: {args.backbone}")
        # Create a single-backbone wave
        tier = BACKBONE_PROFILES[args.backbone]["tier"]
        wave_defs = [{
            "name": f"Single — {args.backbone}",
            "tier": tier,
            "backbones": [args.backbone],
        }]

    report = train_all_backbones(
        waves=wave_defs,
        num_workers=args.workers,
        dry_run=args.dry_run,
        skip_existing=not args.no_skip,
    )
    print(report.summary())
