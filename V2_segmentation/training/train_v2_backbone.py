"""
V2_segmentation/training/train_v2_backbone.py
===============================================
Core 3-phase trainer for a single V2 dual-head backbone.

Phase A — Segmentation head training (backbone FROZEN)
Phase B — Joint fine-tuning (backbone + both heads)
Phase C — Classification refinement (seg head FROZEN)

After Phase B a **rollback gate** checks:
  - val_acc drop ≥ 0.5 pp vs V1 baseline  → REVERT
  - mean_IoU < 0.50                        → REVERT
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import amp  # type: ignore[attr-defined]
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from V2_segmentation.config import (
    AMP_ENABLED,
    BACKBONE_PROFILES,
    DEVICE,
    IMG_SIZE,
    NUM_CLASSES,
    NUM_SEG_CHANNELS,
    PHASE_A,
    PHASE_B,
    PHASE_C,
    ROLLBACK_MEAN_IOU_THRESHOLD,
    ROLLBACK_VAL_ACC_DROP_THRESHOLD,
    RUN_SEED,
    WARMUP_EPOCHS,
)
from V2_segmentation.data.hard_example_sampler import HardExampleSampler
from V2_segmentation.data.seg_dataset import create_seg_dataloaders
from V2_segmentation.losses.joint_loss import JointLoss
from V2_segmentation.models.dual_head import DualHeadModel
from V2_segmentation.models.model_factory import build_v2_model, load_v1_weights
from V2_segmentation.training.checkpoint_manager import CheckpointManager, RollbackLogger
from V2_segmentation.training.memory_manager import MemoryManager
from V2_segmentation.training.metrics import MetricTracker

logger = logging.getLogger(__name__)


# ============================================================================
#  Helpers
# ============================================================================

def load_v1_checkpoint(model: DualHeadModel, backbone_name: str) -> None:
    """Find and load V1 checkpoint weights into the backbone of a DualHeadModel."""
    from V2_segmentation.config import V1_CKPT_DIR

    # Try checkpoint suffixes in priority order
    for suffix in ("_finetune_best.pth", "_final.pth", "_head_best.pth"):
        ckpt_path = V1_CKPT_DIR / f"{backbone_name}{suffix}"
        if ckpt_path.exists():
            logger.info(f"  Loading V1 weights from {ckpt_path.name}")
            load_v1_weights(model.backbone, ckpt_path, strict=False)
            return

    logger.warning(
        f"  No V1 checkpoint found for {backbone_name} in {V1_CKPT_DIR} — "
        f"using random initialization"
    )


def _seed_everything(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = WARMUP_EPOCHS,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create CosineAnnealingWarmRestarts with linear warmup."""
    warmup = LinearLR(
        optimizer, start_factor=0.01, total_iters=max(warmup_epochs, 1),
    )
    cosine = CosineAnnealingWarmRestarts(
        optimizer, T_0=max(total_epochs - warmup_epochs, 1), T_mult=1,
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
    )
    return scheduler


# ============================================================================
#  Single-epoch training
# ============================================================================

def train_one_epoch(
    model: DualHeadModel,
    loader: DataLoader,
    criterion: JointLoss,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,  # type: ignore[name-defined]
    tracker: MetricTracker,
    grad_accum_steps: int = 1,
    hard_sampler: HardExampleSampler | None = None,
) -> dict[str, Any]:
    """Run one training epoch.

    Returns
    -------
    Summary dict from MetricTracker.compute().
    """
    model.train()
    tracker.reset()
    optimizer.zero_grad(set_to_none=True)

    sample_losses: list[float] = []

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)
        confidence = batch["confidence"].to(DEVICE, non_blocking=True)

        with amp.autocast("cuda", enabled=AMP_ENABLED):  # type: ignore[attr-defined]
            outputs = model(images)
            loss_dict = criterion(
                cls_logits=outputs["cls_logits"],
                seg_logits=outputs["seg_logits"],
                cls_targets=labels,
                seg_targets=masks,
                confidence=confidence,
            )
            loss = loss_dict["loss"]

        # Scale loss for gradient accumulation
        scaled_loss = MemoryManager.scale_loss(loss, grad_accum_steps)
        scaler.scale(scaled_loss).backward()

        # Track per-sample loss for hard-example mining
        sample_losses.append(loss.item())

        # Optimizer step at accumulation boundary
        if MemoryManager.should_step(batch_idx, grad_accum_steps):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Track metrics
        tracker.update(
            cls_logits=outputs["cls_logits"],
            cls_labels=labels,
            seg_logits=outputs["seg_logits"],
            seg_targets=masks,
            loss_dict=loss_dict,
        )

    # Update hard-example sampler with per-batch mean losses
    if hard_sampler is not None and sample_losses:
        # Approximate: broadcast batch loss across batch indices
        losses_arr = np.array(sample_losses, dtype=np.float64)
        # Replicate to match dataset size (approximate)
        tiled = np.tile(losses_arr, (hard_sampler.num_samples // max(len(losses_arr), 1) + 1))
        hard_sampler.update_weights(tiled[: hard_sampler.num_samples])

    return tracker.compute()


# ============================================================================
#  Validation epoch
# ============================================================================

@torch.no_grad()
def validate_one_epoch(
    model: DualHeadModel,
    loader: DataLoader,
    criterion: JointLoss,
    tracker: MetricTracker,
) -> dict[str, Any]:
    """Run one validation epoch.

    Returns
    -------
    Summary dict from MetricTracker.compute().
    """
    model.eval()
    tracker.reset()

    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)
        confidence = batch["confidence"].to(DEVICE, non_blocking=True)

        with amp.autocast("cuda", enabled=AMP_ENABLED):  # type: ignore[attr-defined]
            outputs = model(images)
            loss_dict = criterion(
                cls_logits=outputs["cls_logits"],
                seg_logits=outputs["seg_logits"],
                cls_targets=labels,
                seg_targets=masks,
                confidence=confidence,
            )

        tracker.update(
            cls_logits=outputs["cls_logits"],
            cls_labels=labels,
            seg_logits=outputs["seg_logits"],
            seg_targets=masks,
            loss_dict=loss_dict,
        )

    return tracker.compute()


# ============================================================================
#  Phase runner
# ============================================================================

def _run_phase(
    phase_name: str,
    phase_cfg: dict[str, Any],
    model: DualHeadModel,
    loaders: dict[str, DataLoader],
    ckpt_mgr: CheckpointManager,
    grad_accum_steps: int = 1,
    hard_sampler: HardExampleSampler | None = None,
) -> dict[str, Any]:
    """Train a single phase (A, B, or C).

    Returns
    -------
    Best validation metrics dict for this phase.
    """
    logger.info(f"{'='*60}")
    logger.info(f"  Phase {phase_name} — {phase_cfg.get('_desc', '')}")
    logger.info(f"{'='*60}")

    epochs = phase_cfg["epochs"]
    patience = phase_cfg["patience"]

    # ── Freeze/unfreeze per phase ────────────────────────────────────
    if phase_name == "A":
        model.freeze_backbone()
        model.freeze_cls_head()
        model.unfreeze_seg_head()
    elif phase_name == "B":
        model.unfreeze_backbone()
        model.unfreeze_seg_head()
        model.unfreeze_cls_head()
    elif phase_name == "C":
        model.unfreeze_backbone()
        model.freeze_seg_head()
        model.unfreeze_cls_head()

    # ── Loss + optimizer ─────────────────────────────────────────────
    criterion = JointLoss()
    criterion.set_phase(phase_name)

    param_groups = model.get_param_groups(
        backbone_lr=phase_cfg["backbone_lr"],
        seg_head_lr=phase_cfg["seg_head_lr"],
        cls_head_lr=phase_cfg["cls_head_lr"],
        weight_decay=phase_cfg["weight_decay"],
    )
    if not param_groups:
        logger.warning(f"  Phase {phase_name}: no trainable parameters — skipping")
        return {}

    optimizer = AdamW(param_groups)
    scheduler = _create_scheduler(optimizer, epochs)
    scaler = amp.GradScaler("cuda", enabled=AMP_ENABLED)  # type: ignore[attr-defined]

    train_tracker = MetricTracker()
    val_tracker = MetricTracker()

    best_val_metrics: dict[str, Any] = {}
    best_metric_val = -float("inf")
    # Metric to track: mIoU for Phase A, cls_accuracy for B/C
    tracked_key = "mean_iou" if phase_name == "A" else "cls_accuracy"
    epochs_no_improve = 0

    config_snapshot = {
        "phase": phase_name,
        "epochs": epochs,
        "backbone_lr": phase_cfg["backbone_lr"],
        "seg_head_lr": phase_cfg["seg_head_lr"],
        "cls_head_lr": phase_cfg["cls_head_lr"],
        "grad_accum": grad_accum_steps,
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler,
            train_tracker, grad_accum_steps, hard_sampler,
        )

        # ── Validate ─────────────────────────────────────────────────
        val_metrics = validate_one_epoch(
            model, loaders["val"], criterion, val_tracker,
        )

        scheduler.step()
        elapsed = time.time() - t0

        # ── Log ──────────────────────────────────────────────────────
        train_loss = train_metrics.get("avg_loss", 0)
        val_loss = val_metrics.get("avg_loss", 0)
        val_acc = val_metrics.get("cls_accuracy", 0)
        val_miou = val_metrics.get("mean_iou", 0)
        logger.info(
            f"  Phase {phase_name} epoch {epoch}/{epochs} "
            f"[{elapsed:.1f}s] — "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  val_mIoU={val_miou:.4f}"
        )

        # ── Best checkpoint ──────────────────────────────────────────
        current_val = val_metrics.get(tracked_key, 0)
        if current_val > best_metric_val:
            best_metric_val = current_val
            best_val_metrics = val_metrics.copy()
            epochs_no_improve = 0
            ckpt_mgr.save(
                model, optimizer, scheduler, epoch, phase_name,
                val_metrics, config=config_snapshot,
                suffix=f"phase{phase_name}_best",
            )
        else:
            epochs_no_improve += 1

        # ── Early stopping ───────────────────────────────────────────
        if epochs_no_improve >= patience:
            logger.info(
                f"  Early stopping Phase {phase_name} at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    # Restore best checkpoint for this phase
    restored = ckpt_mgr.load_latest(phase_name, model)
    if restored:
        logger.info(f"  Restored Phase {phase_name} best checkpoint")

    MemoryManager.log_vram(f"after Phase {phase_name}")
    return best_val_metrics


# ============================================================================
#  Rollback gate (after Phase B)
# ============================================================================

def _check_rollback(
    backbone_name: str,
    v2_metrics: dict[str, Any],
    ckpt_mgr: CheckpointManager,
    rollback_logger: RollbackLogger,
) -> bool:
    """Check if backbone should revert to V1.

    Returns True if training should STOP (rollback triggered).
    """
    v2_acc = v2_metrics.get("cls_accuracy", 0.0)
    v2_miou = v2_metrics.get("mean_iou", 0.0)
    v1_acc = ckpt_mgr.get_v1_accuracy()

    # Gate 1: mIoU too low
    if v2_miou < ROLLBACK_MEAN_IOU_THRESHOLD:
        rollback_logger.log_rollback(
            backbone_name,
            f"mean_IoU={v2_miou:.4f} < {ROLLBACK_MEAN_IOU_THRESHOLD}",
            v1_acc, v2_acc, v2_miou,
        )
        return True

    # Gate 2: accuracy regression
    if v1_acc is not None:
        acc_drop = v1_acc - v2_acc
        if acc_drop >= ROLLBACK_VAL_ACC_DROP_THRESHOLD:
            rollback_logger.log_rollback(
                backbone_name,
                f"val_acc dropped {acc_drop:.4f} (V1={v1_acc:.4f} → V2={v2_acc:.4f})",
                v1_acc, v2_acc, v2_miou,
            )
            return True

    # Passed
    rollback_logger.log_pass(backbone_name, v1_acc, v2_acc, v2_miou)
    logger.info(
        f"  ✓ Rollback gate PASSED: "
        f"V1_acc={v1_acc}, V2_acc={v2_acc:.4f}, mIoU={v2_miou:.4f}"
    )
    return False


# ============================================================================
#  Full training orchestrator for ONE backbone
# ============================================================================

def train_v2_backbone(
    backbone_name: str,
    num_workers: int = 4,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train a single backbone through all 3 phases.

    Parameters
    ----------
    backbone_name : str
        e.g. "CustomMobileOne".
    num_workers : int
        DataLoader workers.
    dry_run : bool
        If True, run only 1 epoch per phase (for smoke tests).

    Returns
    -------
    dict with training results, or empty dict on rollback.
    """
    t_start = time.time()
    _seed_everything(RUN_SEED)

    logger.info(f"\n{'#'*70}")
    logger.info(f"  V2 Training: {backbone_name}")
    logger.info(f"{'#'*70}\n")

    # ── Memory budget ────────────────────────────────────────────────
    mm = MemoryManager()
    budget = mm.get_budget(backbone_name)
    mm.log_budget(backbone_name)

    # ── Build model (V1 → V2) ───────────────────────────────────────
    profile = BACKBONE_PROFILES[backbone_name]
    model = build_v2_model(
        backbone_name=backbone_name,
        num_classes=NUM_CLASSES,
        num_seg_channels=NUM_SEG_CHANNELS,
        decoder_channels=profile["decoder_channels"],
        skip_channels=profile["skip_channels"],
    )
    load_v1_checkpoint(model, backbone_name)
    model.to(DEVICE)

    # Gradient checkpointing for HEAVY tier
    mm.apply_grad_checkpoint(model, backbone_name)

    MemoryManager.log_vram("after model load")

    # ── Data loaders ─────────────────────────────────────────────────
    loaders = create_seg_dataloaders(
        batch_size=budget.batch_size,
        img_size=IMG_SIZE,
        num_workers=num_workers,
    )
    train_size = len(loaders["train"].dataset)  # type: ignore[arg-type]

    # Hard-example sampler (activated after first epoch)
    hard_sampler = HardExampleSampler(num_samples=train_size)

    # ── Checkpoint manager ───────────────────────────────────────────
    ckpt_mgr = CheckpointManager(backbone_name)
    rollback_log = RollbackLogger()

    # ── Override epochs for dry-run ──────────────────────────────────
    phase_a_cfg = {**PHASE_A, "_desc": "Seg head training (backbone frozen)"}
    phase_b_cfg = {**PHASE_B, "_desc": "Joint fine-tuning (all unfrozen)"}
    phase_c_cfg = {**PHASE_C, "_desc": "Cls refinement (seg frozen)"}

    if dry_run:
        for cfg in (phase_a_cfg, phase_b_cfg, phase_c_cfg):
            cfg["epochs"] = 1
            cfg["patience"] = 1

    results: dict[str, Any] = {
        "backbone_name": backbone_name,
        "tier": budget.tier,
        "batch_size": budget.batch_size,
        "grad_accum": budget.grad_accum_steps,
        "phases": {},
    }

    # ================================================================
    #  PHASE A — Segmentation head training
    # ================================================================
    phase_a_metrics = _run_phase(
        "A", phase_a_cfg, model, loaders, ckpt_mgr,
        budget.grad_accum_steps, hard_sampler,
    )
    results["phases"]["A"] = phase_a_metrics

    # ================================================================
    #  PHASE B — Joint fine-tuning
    # ================================================================
    phase_b_metrics = _run_phase(
        "B", phase_b_cfg, model, loaders, ckpt_mgr,
        budget.grad_accum_steps, hard_sampler,
    )
    results["phases"]["B"] = phase_b_metrics

    # ── Rollback gate ────────────────────────────────────────────────
    should_rollback = _check_rollback(
        backbone_name, phase_b_metrics, ckpt_mgr, rollback_log,
    )
    if should_rollback:
        logger.warning(
            f"  ✗ ROLLBACK: {backbone_name} reverted to V1 — "
            f"Phase C skipped"
        )
        results["rollback"] = True
        _cleanup_gpu()
        return results

    # ================================================================
    #  PHASE C — Classification refinement
    # ================================================================
    phase_c_metrics = _run_phase(
        "C", phase_c_cfg, model, loaders, ckpt_mgr,
        budget.grad_accum_steps,
    )
    results["phases"]["C"] = phase_c_metrics

    # ── Save final model ─────────────────────────────────────────────
    final_metrics = phase_c_metrics or phase_b_metrics
    ckpt_mgr.save_final(model, final_metrics)
    results["rollback"] = False

    elapsed = time.time() - t_start
    results["total_time_s"] = round(elapsed, 1)
    logger.info(
        f"\n  ✓ {backbone_name} V2 training complete in {elapsed / 60:.1f} min "
        f"(acc={final_metrics.get('cls_accuracy', 0):.4f}, "
        f"mIoU={final_metrics.get('mean_iou', 0):.4f})\n"
    )

    _cleanup_gpu()
    return results


def _cleanup_gpu() -> None:
    """Clean up GPU memory between backbone runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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

    parser = argparse.ArgumentParser(description="Train one V2 dual-head backbone")
    parser.add_argument(
        "backbone", type=str,
        help=f"Backbone name. One of: {', '.join(BACKBONE_PROFILES)}",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 1 epoch per phase (smoke test)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader workers (default: 4)",
    )
    args = parser.parse_args()

    result = train_v2_backbone(
        backbone_name=args.backbone,
        num_workers=args.workers,
        dry_run=args.dry_run,
    )
    logger.info(f"Result: {result}")
