"""
V2_segmentation/scripts/smoke_training_pipeline.py
====================================================
Comprehensive end-to-end smoke test for the FULL V2 training pipeline.

Tests for EACH of all 15 backbones:
  1. Model creation (build_v2_model)
  2. V1 checkpoint loading (load_v1_checkpoint)
  3. Data loading (SegmentationDataset + DataLoader with real split_dataset)
  4. Augmentations (JointTransform on real images + fallback zero masks)
  5. Joint loss computation (forward pass through JointLoss)
  6. Training step (forward + backward + optimizer step with AMP + grad accum)
  7. Validation step (forward pass only, no grad)
  8. Checkpoint save/load (CheckpointManager save + restore)
  9. Freeze/unfreeze for all 3 phases (A/B/C)
  10. MetricTracker accumulation + compute

Run:
    python -m V2_segmentation.scripts.smoke_training_pipeline

Or single backbone:
    python -m V2_segmentation.scripts.smoke_training_pipeline --backbone CustomMobileOne
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
import traceback
from typing import Any

import torch
import torch.nn as nn
from torch import amp  # type: ignore[attr-defined]

from V2_segmentation.config import (
    AMP_ENABLED,
    BACKBONE_PROFILES,
    BACKBONES,
    DEVICE,
    IMG_SIZE,
    NUM_CLASSES,
    NUM_SEG_CHANNELS,
)
from pathlib import Path

from V2_segmentation.data.seg_dataset import create_seg_dataloaders
from V2_segmentation.losses.joint_loss import JointLoss
from V2_segmentation.models.model_factory import build_v2_model
from V2_segmentation.training.checkpoint_manager import CheckpointManager
from V2_segmentation.training.memory_manager import MemoryManager
from V2_segmentation.training.metrics import MetricTracker
from V2_segmentation.training.train_v2_backbone import load_v1_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
#  Comprehensive per-backbone test
# ============================================================================

def test_backbone_pipeline(
    backbone_name: str,
    loaders: dict[str, Any],
) -> dict[str, Any]:
    """Run all 10 checks for one backbone.

    Returns dict with backbone_name, passed (bool), checks (list of results).
    """
    profile = BACKBONE_PROFILES[backbone_name]
    checks: list[dict[str, Any]] = []
    model = None

    def _check(name: str, fn):
        """Run a check and record pass/fail."""
        t0 = time.time()
        try:
            result = fn()
            elapsed = time.time() - t0
            checks.append({"name": name, "passed": True, "time": elapsed})
            logger.info(f"    [OK] {name} ({elapsed:.2f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - t0
            checks.append({"name": name, "passed": False, "error": str(e), "time": elapsed})
            logger.error(f"    [FAIL] {name} FAILED: {e}")
            traceback.print_exc()
            return None

    logger.info(f"\n{'='*60}")
    logger.info(f"  TESTING: {backbone_name} (tier={profile['tier']})")
    logger.info(f"{'='*60}")

    # ── 1. Model creation ────────────────────────────────────────────
    def check_model_creation():
        nonlocal model
        model = build_v2_model(
            backbone_name=backbone_name,
            num_classes=NUM_CLASSES,
            num_seg_channels=NUM_SEG_CHANNELS,
            decoder_channels=profile["decoder_channels"],
            skip_channels=profile["skip_channels"],
            device="cpu",
        )
        total_params = sum(p.numel() for p in model.parameters())
        seg_params = model.seg_parameter_count()
        logger.info(f"      params={total_params:,}  seg_decoder={seg_params:,}")
        return model

    _check("1. Model creation", check_model_creation)
    if model is None:
        return {"backbone": backbone_name, "passed": False, "checks": checks}

    # ── 2. V1 checkpoint loading ─────────────────────────────────────
    def check_v1_loading():
        assert model is not None
        load_v1_checkpoint(model, backbone_name)

    _check("2. V1 checkpoint loading", check_v1_loading)

    # Move to device
    model.to(DEVICE)

    # ── 3. Data loading (one batch from train + val) ─────────────────
    train_batch = None
    val_batch = None

    def check_data_loading():
        nonlocal train_batch, val_batch
        train_batch = next(iter(loaders["train"]))
        val_batch = next(iter(loaders["val"]))
        # Validate shapes
        bs = train_batch["image"].shape[0]
        assert train_batch["image"].shape == (bs, 3, IMG_SIZE, IMG_SIZE), \
            f"image shape: {train_batch['image'].shape}"
        assert train_batch["mask"].shape == (bs, NUM_SEG_CHANNELS, IMG_SIZE, IMG_SIZE), \
            f"mask shape: {train_batch['mask'].shape}"
        assert train_batch["confidence"].shape[0] == bs, \
            f"confidence batch: {train_batch['confidence'].shape[0]}"
        assert train_batch["label"].shape == (bs,), \
            f"label shape: {train_batch['label'].shape}"
        logger.info(
            f"      train_batch: imgs={train_batch['image'].shape}, "
            f"masks={train_batch['mask'].shape}, "
            f"labels={train_batch['label'].shape}"
        )
        logger.info(
            f"      val_batch:   imgs={val_batch['image'].shape}, "
            f"masks={val_batch['mask'].shape}"
        )

    _check("3. Data loading + augmentations", check_data_loading)
    if train_batch is None:
        return {"backbone": backbone_name, "passed": False, "checks": checks}

    # ── 4. Forward pass ──────────────────────────────────────────────
    outputs = None

    def check_forward():
        nonlocal outputs
        assert model is not None
        assert train_batch is not None
        images = train_batch["image"].to(DEVICE)
        model.eval()
        with torch.no_grad():
            outputs = model(images)
        assert "cls_logits" in outputs, "Missing cls_logits"
        assert "seg_logits" in outputs, "Missing seg_logits"
        cls_shape = outputs["cls_logits"].shape
        seg_shape = outputs["seg_logits"].shape
        assert cls_shape[1] == NUM_CLASSES, f"cls classes: {cls_shape[1]}"
        assert seg_shape[1] == NUM_SEG_CHANNELS, f"seg channels: {seg_shape[1]}"
        assert seg_shape[2] == IMG_SIZE and seg_shape[3] == IMG_SIZE, \
            f"seg spatial: {seg_shape[2:]}"
        assert not torch.isnan(outputs["cls_logits"]).any(), "NaN in cls_logits"
        assert not torch.isnan(outputs["seg_logits"]).any(), "NaN in seg_logits"
        logger.info(f"      cls_logits={cls_shape}, seg_logits={seg_shape}")

    _check("4. Forward pass", check_forward)

    # ── 5. Joint loss computation ────────────────────────────────────
    loss_dict = None

    def check_loss():
        nonlocal loss_dict
        assert model is not None
        assert train_batch is not None
        criterion = JointLoss()
        criterion.set_phase("B")  # Joint phase: both losses active
        images = train_batch["image"].to(DEVICE)
        labels = train_batch["label"].to(DEVICE)
        masks = train_batch["mask"].to(DEVICE)
        confidence = train_batch["confidence"].to(DEVICE)
        model.train()
        with amp.autocast("cuda", enabled=AMP_ENABLED):  # type: ignore[attr-defined]
            out = model(images)
            loss_dict = criterion(
                cls_logits=out["cls_logits"],
                seg_logits=out["seg_logits"],
                cls_targets=labels,
                seg_targets=masks,
                confidence=confidence,
            )
        loss_val = loss_dict["loss"].item()
        assert not torch.isnan(loss_dict["loss"]), "NaN loss"
        assert loss_val > 0, f"Loss should be > 0, got {loss_val}"
        logger.info(
            f"      loss={loss_val:.4f} "
            f"(seg={loss_dict['loss_seg'].item():.4f}, "
            f"cls={loss_dict['loss_cls'].item():.4f})"
        )

    _check("5. Joint loss computation", check_loss)

    # ── 6. Training step (backward + optimizer + AMP) ────────────────
    def check_training_step():
        assert model is not None
        assert train_batch is not None
        # Clear residual grads from V1 backbone arch verification
        model.zero_grad(set_to_none=True)

        criterion = JointLoss()
        criterion.set_phase("A")  # Seg-only phase
        # Order matters: freeze/unfreeze methods set eval/train per-component
        model.freeze_backbone()
        model.freeze_cls_head()
        model.unfreeze_seg_head()

        param_groups = model.get_param_groups(
            backbone_lr=0.0, seg_head_lr=1e-3, cls_head_lr=0.0,
        )
        optimizer = torch.optim.AdamW(param_groups, lr=1e-3)
        scaler = amp.GradScaler("cuda", enabled=AMP_ENABLED)  # type: ignore[attr-defined]
        grad_accum = profile["grad_accum"]

        images = train_batch["image"].to(DEVICE)
        labels = train_batch["label"].to(DEVICE)
        masks = train_batch["mask"].to(DEVICE)
        confidence = train_batch["confidence"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=AMP_ENABLED):  # type: ignore[attr-defined]
            out = model(images)
            ld = criterion(
                cls_logits=out["cls_logits"],
                seg_logits=out["seg_logits"],
                cls_targets=labels,
                seg_targets=masks,
                confidence=confidence,
            )

        scaled = MemoryManager.scale_loss(ld["loss"], grad_accum)
        scaler.scale(scaled).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Verify gradients flowed to seg decoder only
        seg_grads = sum(
            1 for p in model.seg_decoder.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        backbone_grads = sum(
            1 for p in model.backbone.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert seg_grads > 0, "No gradients in seg decoder"
        assert backbone_grads == 0, f"Backbone should be frozen but has {backbone_grads} grads"
        logger.info(f"      seg_grads={seg_grads}, backbone_grads={backbone_grads} (frozen)")

    _check("6. Training step (AMP + grad accum)", check_training_step)

    # ── 7. Validation step ───────────────────────────────────────────
    def check_validation_step():
        assert model is not None
        assert val_batch is not None
        model.eval()
        tracker = MetricTracker()
        tracker.reset()

        images = val_batch["image"].to(DEVICE)
        labels = val_batch["label"].to(DEVICE)
        masks = val_batch["mask"].to(DEVICE)

        with torch.no_grad(), amp.autocast("cuda", enabled=AMP_ENABLED):  # type: ignore[attr-defined]
            out = model(images)

        tracker.update(
            cls_logits=out["cls_logits"],
            cls_labels=labels,
            seg_logits=out["seg_logits"],
            seg_targets=masks,
        )
        summary = tracker.compute()
        assert "cls_accuracy" in summary, "Missing cls_accuracy"
        assert "mean_iou" in summary, "Missing mean_iou"
        logger.info(
            f"      val_acc={summary['cls_accuracy']:.4f}, "
            f"val_mIoU={summary['mean_iou']:.4f}"
        )

    _check("7. Validation step + MetricTracker", check_validation_step)

    # ── 8. Checkpoint save + load ────────────────────────────────────
    def check_checkpoint():
        import tempfile
        import shutil

        assert model is not None
        tmp_dir = tempfile.mkdtemp(prefix="v2_smoke_ckpt_")
        try:
            mgr = CheckpointManager(backbone_name, ckpt_dir=Path(tmp_dir))
            dummy_metrics = {"cls_accuracy": 0.5, "mean_iou": 0.3}

            # Save
            ckpt_path = mgr.save(
                model,
                torch.optim.AdamW(model.parameters(), lr=1e-4),
                None,  # no scheduler for smoke
                epoch=1, phase="A",
                metrics=dummy_metrics,
                suffix="phaseA_best",
            )
            assert ckpt_path.exists(), f"Checkpoint not created: {ckpt_path}"

            # Load into fresh model
            model2 = build_v2_model(
                backbone_name=backbone_name,
                num_classes=NUM_CLASSES,
                num_seg_channels=NUM_SEG_CHANNELS,
                decoder_channels=profile["decoder_channels"],
                skip_channels=profile["skip_channels"],
                device="cpu",
            )
            restored = mgr.load("phaseA_best", model2)
            assert restored["epoch"] == 1
            assert restored["phase"] == "A"
            assert restored["backbone_name"] == backbone_name
            logger.info(f"      saved+loaded checkpoint OK (keys: {list(restored.keys())[:5]}...)")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    _check("8. Checkpoint save + load", check_checkpoint)

    # ── 9. Phase freeze/unfreeze cycles ──────────────────────────────
    def check_phases():
        assert model is not None
        # Phase A: backbone+cls frozen, seg trainable
        model.freeze_backbone()
        model.freeze_cls_head()
        model.unfreeze_seg_head()
        a_trainable = sum(1 for p in model.parameters() if p.requires_grad)

        # Phase B: everything unfrozen
        model.unfreeze_backbone()
        model.unfreeze_cls_head()
        model.unfreeze_seg_head()
        b_trainable = sum(1 for p in model.parameters() if p.requires_grad)

        # Phase C: seg frozen, backbone+cls trainable
        model.unfreeze_backbone()
        model.unfreeze_cls_head()
        model.freeze_seg_head()
        c_trainable = sum(1 for p in model.parameters() if p.requires_grad)

        total = sum(1 for _ in model.parameters())
        assert a_trainable < b_trainable, \
            f"Phase A ({a_trainable}) should have fewer trainable than B ({b_trainable})"
        assert c_trainable < b_trainable, \
            f"Phase C ({c_trainable}) should have fewer trainable than B ({b_trainable})"
        logger.info(
            f"      Phase A={a_trainable}, B={b_trainable}, C={c_trainable} trainable (total={total})"
        )

    _check("9. Phase freeze/unfreeze (A/B/C)", check_phases)

    # ── 10. Memory cleanup ───────────────────────────────────────────
    def check_cleanup():
        nonlocal model
        assert model is not None
        model.cleanup()
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vram = MemoryManager.get_vram_usage()
        logger.info(f"      VRAM after cleanup: {vram['allocated']:.2f}GB alloc, {vram['free']:.2f}GB free")

    _check("10. GPU memory cleanup", check_cleanup)

    # ── Result ───────────────────────────────────────────────────────
    all_passed = all(c["passed"] for c in checks)
    return {"backbone": backbone_name, "passed": all_passed, "checks": checks}


# ============================================================================
#  Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Comprehensive V2 training pipeline smoke test")
    parser.add_argument(
        "--backbone", type=str, default=None,
        help="Test only this backbone. If omitted, tests all 15.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for smoke test DataLoaders (default: 4)",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="DataLoader workers (default: 2)",
    )
    args = parser.parse_args()

    backbones_to_test = [args.backbone] if args.backbone else BACKBONES

    logger.info(f"\n{'#'*60}")
    logger.info(f"  V2 TRAINING PIPELINE — COMPREHENSIVE SMOKE TEST")
    logger.info(f"  Backbones: {len(backbones_to_test)}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  AMP: {AMP_ENABLED}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"{'#'*60}\n")

    # ── Create DataLoaders ONCE (shared across all backbones) ────────
    logger.info("Creating DataLoaders (shared for all backbones)...")
    t0 = time.time()
    loaders = create_seg_dataloaders(
        batch_size=args.batch_size,
        img_size=IMG_SIZE,
        num_workers=args.workers,
    )
    logger.info(f"DataLoaders ready in {time.time() - t0:.1f}s\n")

    # ── Run per-backbone tests ───────────────────────────────────────
    results: list[dict[str, Any]] = []
    for backbone_name in backbones_to_test:
        result = test_backbone_pipeline(backbone_name, loaders)
        results.append(result)

    # ── Summary ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"  COMPREHENSIVE SMOKE TEST SUMMARY")
    logger.info(f"{'='*60}")

    passed_count = 0
    for r in results:
        status = "✅ PASSED" if r["passed"] else "❌ FAILED"
        check_summary = f"{sum(1 for c in r['checks'] if c['passed'])}/{len(r['checks'])} checks"
        tier = BACKBONE_PROFILES.get(r["backbone"], {}).get("tier", "?")
        logger.info(f"  {status}  {r['backbone']:30s}  tier={tier:6s}  {check_summary}")
        if not r["passed"]:
            for c in r["checks"]:
                if not c["passed"]:
                    logger.info(f"         +-- [FAIL] {c['name']}: {c.get('error', '?')}")
        if r["passed"]:
            passed_count += 1

    logger.info(f"\n  {passed_count}/{len(results)} PASSED\n")

    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
