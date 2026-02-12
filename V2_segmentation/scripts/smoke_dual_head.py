"""
V2_segmentation/scripts/smoke_dual_head.py
==========================================
Smoke test: verify DualHeadModel forward pass for all 15 backbones.

For each backbone:
  1. Create V1 backbone (random init — no checkpoint needed)
  2. Wrap in DualHeadModel with tier-appropriate decoder
  3. Forward pass with dummy input
  4. Verify output shapes (cls_logits and seg_logits)
  5. Backward pass (verify gradients flow)
  6. Report VRAM usage

Usage
-----
    python -m V2_segmentation.scripts.smoke_dual_head
    python -m V2_segmentation.scripts.smoke_dual_head CustomEfficientNetV4  # single backbone
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from V2_segmentation.config import (
    BACKBONE_PROFILES, BACKBONES, DEVICE, IMG_SIZE,
    NUM_CLASSES, NUM_SEG_CHANNELS,
)
from V2_segmentation.models.model_factory import build_v2_model
from V2_segmentation.training.memory_manager import MemoryManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def smoke_test_backbone(backbone_name: str, device: torch.device = DEVICE) -> dict:
    """Run smoke test for a single backbone.

    Returns
    -------
    dict with test results.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  SMOKE TEST: {backbone_name}")
    logger.info(f"{'='*60}")

    profile = BACKBONE_PROFILES.get(backbone_name, {})
    tier = profile.get("tier", "UNKNOWN")
    decoder_ch = profile.get("decoder_channels", 256)
    skip_ch = profile.get("skip_channels", 48)

    result = {
        "backbone": backbone_name,
        "tier": tier,
        "decoder_channels": decoder_ch,
    }

    try:
        # 1. Build model (random init — no checkpoint)
        logger.info(f"  Building DualHeadModel (decoder={decoder_ch}ch)...")
        t0 = time.time()
        model = build_v2_model(
            backbone_name=backbone_name,
            num_classes=NUM_CLASSES,
            num_seg_channels=NUM_SEG_CHANNELS,
            decoder_channels=decoder_ch,
            skip_channels=skip_ch,
            device=str(device),
        )
        build_time = time.time() - t0
        result["build_time_s"] = round(build_time, 2)
        result["total_params"] = model.total_parameter_count()
        result["seg_params"] = model.seg_parameter_count()
        logger.info(
            f"  Built in {build_time:.1f}s — "
            f"total={result['total_params']:,} params, "
            f"seg_decoder={result['seg_params']:,} params"
        )

        # 2. Forward pass
        logger.info(f"  Forward pass (BS=2)...")
        dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=device)
        model.eval()
        with torch.no_grad():
            outputs = model(dummy)

        cls_shape = tuple(outputs["cls_logits"].shape)
        seg_shape = tuple(outputs["seg_logits"].shape)
        result["cls_logits_shape"] = cls_shape
        result["seg_logits_shape"] = seg_shape

        expected_cls = (2, NUM_CLASSES)
        expected_seg = (2, NUM_SEG_CHANNELS, IMG_SIZE, IMG_SIZE)

        assert cls_shape == expected_cls, (
            f"cls_logits shape mismatch: {cls_shape} != {expected_cls}"
        )
        assert seg_shape == expected_seg, (
            f"seg_logits shape mismatch: {seg_shape} != {expected_seg}"
        )
        logger.info(f"  [OK] cls_logits: {cls_shape}")
        logger.info(f"  [OK] seg_logits: {seg_shape}")

        # 3. Check for NaN/Inf
        assert not torch.isnan(outputs["cls_logits"]).any(), "NaN in cls_logits!"
        assert not torch.isnan(outputs["seg_logits"]).any(), "NaN in seg_logits!"
        assert not torch.isinf(outputs["cls_logits"]).any(), "Inf in cls_logits!"
        assert not torch.isinf(outputs["seg_logits"]).any(), "Inf in seg_logits!"
        logger.info(f"  [OK] No NaN/Inf in outputs")

        # 4. Backward pass (verify gradients flow through both heads)
        logger.info(f"  Backward pass...")
        model.train()
        model.unfreeze_backbone()
        model.unfreeze_seg_head()

        dummy_grad = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=device)
        outputs_grad = model(dummy_grad)

        # Combined loss (cls + seg)
        cls_loss = nn.functional.cross_entropy(
            outputs_grad["cls_logits"],
            torch.randint(0, NUM_CLASSES, (2,), device=device),
        )
        seg_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs_grad["seg_logits"],
            torch.rand(2, NUM_SEG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device),
        )
        loss = cls_loss + seg_loss
        loss.backward()

        # Check gradients exist
        has_backbone_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.backbone.parameters()
            if p.requires_grad
        )
        has_seg_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.seg_decoder.parameters()
            if p.requires_grad
        )
        assert has_backbone_grad, "No gradients in backbone!"
        assert has_seg_grad, "No gradients in seg decoder!"
        logger.info(f"  [OK] Gradients flow to backbone and seg decoder")
        result["gradients_ok"] = True

        # 5. VRAM usage
        if torch.cuda.is_available():
            vram = MemoryManager.get_vram_usage()
            result["vram_allocated_gb"] = vram["allocated"]
            logger.info(f"  VRAM: {vram['allocated']:.2f} GB allocated")

        # 6. Test freeze/unfreeze
        model.freeze_backbone()
        model.freeze_cls_head()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Phase A (backbone+cls frozen): {trainable:,} trainable params")
        # Seg decoder params should be the only trainable ones
        assert trainable > 0, "No trainable params after Phase A freeze!"
        assert trainable <= result["seg_params"], (
            f"More trainable params than seg decoder: {trainable} > {result['seg_params']}"
        )
        logger.info(f"  [OK] Freeze/unfreeze works correctly")

        # Cleanup
        model.cleanup()
        del model, dummy, dummy_grad, outputs, outputs_grad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result["status"] = "PASS"
        logger.info(f"  [PASS] {backbone_name} PASSED")

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = str(e)
        logger.error(f"  [FAIL] {backbone_name} FAILED: {e}")
        import traceback
        traceback.print_exc()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def run_all_smoke_tests(
    backbones: list[str] | None = None,
) -> list[dict]:
    """Run smoke tests for all (or selected) backbones.

    Parameters
    ----------
    backbones : list of backbone names.  None = all 15.

    Returns
    -------
    list of result dicts.
    """
    backbones = backbones or BACKBONES
    results = []

    for name in backbones:
        result = smoke_test_backbone(name)
        results.append(result)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("  SMOKE TEST SUMMARY")
    logger.info(f"{'='*60}")
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    for r in results:
        status = "✅" if r["status"] == "PASS" else "❌"
        params = r.get("total_params", "?")
        seg = r.get("seg_params", "?")
        vram = r.get("vram_allocated_gb", "?")
        logger.info(
            f"  {status} {r['backbone']:30s}  "
            f"tier={r.get('tier', '?'):6s}  "
            f"params={params}  seg={seg}  vram={vram}GB"
        )

    logger.info(f"\n  {passed}/{total} PASSED")
    return results


if __name__ == "__main__":
    selected = sys.argv[1:] if len(sys.argv) > 1 else None
    results = run_all_smoke_tests(backbones=selected)

    # Exit with non-zero if any test failed
    if any(r["status"] != "PASS" for r in results):
        sys.exit(1)
