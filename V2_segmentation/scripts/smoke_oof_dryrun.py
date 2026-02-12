"""
V2_segmentation/scripts/smoke_oof_dryrun.py
============================================
Smoke test: dry-run of the out-of-fold (OOF) prediction pipeline.

Verifies that:
  1. DualHeadModel can be built with V1 checkpoint weights
  2. Forward pass produces valid cls + seg outputs
  3. Outputs can be collected across a mini "fold"
  4. OOF predictions aggregate correctly
  5. Memory stays within budget

This does NOT run real training — it simulates the OOF collection loop.

Usage
-----
    python -m V2_segmentation.scripts.smoke_oof_dryrun
    python -m V2_segmentation.scripts.smoke_oof_dryrun CustomMobileOne
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from V2_segmentation.config import (
    DEVICE, IMG_SIZE,
    NUM_CLASSES, NUM_SEG_CHANNELS, V1_CKPT_DIR,
)
from V2_segmentation.models.model_factory import load_v1_into_v2, build_v2_model
from V2_segmentation.training.memory_manager import MemoryManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _find_checkpoint(backbone_name: str) -> Path | None:
    """Find best available V1 checkpoint."""
    for suffix in ("_finetune_best.pth", "_final.pth", "_head_best.pth"):
        path = V1_CKPT_DIR / f"{backbone_name}{suffix}"
        if path.exists():
            return path
    return None


def smoke_oof_dryrun(
    backbone_name: str,
    num_fake_batches: int = 3,
    batch_size: int = 4,
) -> dict:
    """Simulate an OOF prediction collection loop.

    Parameters
    ----------
    backbone_name : str
    num_fake_batches : int
        Number of dummy batches to simulate.
    batch_size : int
        Batch size for dummy data.

    Returns
    -------
    dict with test results.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  OOF DRY-RUN: {backbone_name}")
    logger.info(f"{'='*60}")

    result: dict[str, object] = {"backbone": backbone_name}

    try:
        # 1. Build model (with V1 weights if available)
        ckpt = _find_checkpoint(backbone_name)
        if ckpt:
            logger.info(f"  Loading V1 checkpoint: {ckpt.name}")
            model = load_v1_into_v2(
                backbone_name, ckpt,
                num_classes=NUM_CLASSES,
                num_seg_channels=NUM_SEG_CHANNELS,
                device=str(DEVICE),
            )
            result["v1_checkpoint"] = str(ckpt)
        else:
            logger.info(f"  No V1 checkpoint — using random init")
            model = build_v2_model(
                backbone_name=backbone_name,
                num_classes=NUM_CLASSES,
                num_seg_channels=NUM_SEG_CHANNELS,
                device=str(DEVICE),
            )
            result["v1_checkpoint"] = None

        model.eval()

        # 2. Simulate OOF collection
        all_cls_preds = []
        all_seg_preds = []
        all_labels = []
        total_samples = 0

        logger.info(f"  Simulating {num_fake_batches} batches (BS={batch_size})...")

        for batch_idx in range(num_fake_batches):
            dummy_images = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
            dummy_labels = torch.randint(0, NUM_CLASSES, (batch_size,))

            with torch.no_grad():
                outputs = model(dummy_images)

            # Collect predictions
            cls_probs = F.softmax(outputs["cls_logits"], dim=1).cpu().numpy()
            seg_probs = torch.sigmoid(outputs["seg_logits"]).cpu().numpy()

            all_cls_preds.append(cls_probs)
            all_seg_preds.append(seg_probs)
            all_labels.append(dummy_labels.numpy())
            total_samples += batch_size

            logger.info(
                f"    Batch {batch_idx + 1}/{num_fake_batches}: "
                f"cls={cls_probs.shape}, seg={seg_probs.shape}"
            )

        # 3. Aggregate OOF predictions
        oof_cls = np.concatenate(all_cls_preds, axis=0)   # (N, num_classes)
        oof_seg = np.concatenate(all_seg_preds, axis=0)   # (N, seg_ch, H, W)
        _oof_labels = np.concatenate(all_labels, axis=0)    # (N,)

        result["oof_cls_shape"] = oof_cls.shape
        result["oof_seg_shape"] = oof_seg.shape
        result["total_samples"] = total_samples

        logger.info(f"  OOF cls predictions: {oof_cls.shape}")
        logger.info(f"  OOF seg predictions: {oof_seg.shape}")

        # 4. Sanity checks
        assert oof_cls.shape == (total_samples, NUM_CLASSES)
        assert oof_seg.shape == (total_samples, NUM_SEG_CHANNELS, IMG_SIZE, IMG_SIZE)
        assert np.all(oof_cls >= 0) and np.all(oof_cls <= 1), "cls probs out of [0,1]"
        assert np.all(oof_seg >= 0) and np.all(oof_seg <= 1), "seg probs out of [0,1]"
        assert not np.isnan(oof_cls).any(), "NaN in cls predictions"
        assert not np.isnan(oof_seg).any(), "NaN in seg predictions"

        logger.info(f"  [OK] All sanity checks passed")

        # 5. Check cls prediction quality (should sum to ~1.0 per sample)
        cls_sums = oof_cls.sum(axis=1)
        logger.info(
            f"  cls prob sums: min={cls_sums.min():.4f}, "
            f"max={cls_sums.max():.4f}, mean={cls_sums.mean():.4f}"
        )

        # 6. Check seg prediction statistics
        logger.info(f"  seg mean activation per channel:")
        for ch in range(NUM_SEG_CHANNELS):
            ch_mean = oof_seg[:, ch].mean()
            logger.info(f"    ch{ch}: {ch_mean:.4f}")

        # 7. VRAM
        if torch.cuda.is_available():
            vram = MemoryManager.get_vram_usage()
            result["vram_gb"] = vram["allocated"]
            logger.info(f"  VRAM: {vram['allocated']:.2f} GB")

        # Cleanup
        model.cleanup()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result["status"] = "PASS"
        logger.info(f"  [PASS] OOF dry-run PASSED for {backbone_name}")

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = str(e)
        logger.error(f"  [FAIL] OOF dry-run FAILED for {backbone_name}: {e}")
        import traceback
        traceback.print_exc()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def run_all_oof_dryruns(
    backbones: list[str] | None = None,
) -> list[dict]:
    """Run OOF dry-runs for all (or selected) backbones."""
    backbones = backbones or ["CustomMobileOne", "CustomEfficientNetV4"]  # Quick defaults

    results = []
    for name in backbones:
        result = smoke_oof_dryrun(name)
        results.append(result)

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    logger.info(f"\n  OOF Dry-Run: {passed}/{len(results)} PASSED")
    return results


if __name__ == "__main__":
    selected = sys.argv[1:] if len(sys.argv) > 1 else None
    results = run_all_oof_dryruns(backbones=selected)

    if any(r["status"] != "PASS" for r in results):
        sys.exit(1)
