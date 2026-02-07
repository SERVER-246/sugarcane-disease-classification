"""
V2_segmentation/training/checkpoint_manager.py
================================================
V2 checkpoint save/load/resume with full metadata for reproducibility.

Every checkpoint includes:
  - model_state_dict, optimizer_state_dict, scheduler_state_dict
  - epoch, phase, best metrics (val_acc, mean_iou)
  - training config (batch_size, grad_accum, lr)
  - pseudo-label tier distribution
  - reproducibility metadata (run_seed, git_hash, pip_freeze_hash, timestamp)
  - pytorch_version, cuda_version
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from V2_segmentation.config import CKPT_V2_DIR, RUN_SEED

logger = logging.getLogger(__name__)


def _get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_pip_freeze_hash() -> str:
    """Get SHA-256 of pip freeze output for environment fingerprinting."""
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return hashlib.sha256(result.stdout.encode()).hexdigest()[:16]
    except Exception:
        pass
    return "unknown"


class CheckpointManager:
    """Manage V2 model checkpoints with full metadata.

    Usage
    -----
    >>> ckpt_mgr = CheckpointManager("CustomMobileOne")
    >>> # Save after each epoch:
    >>> ckpt_mgr.save(model, optimizer, scheduler, epoch, phase, metrics, config)
    >>> # Resume:
    >>> state = ckpt_mgr.load_latest(model, optimizer, scheduler)
    """

    def __init__(
        self,
        backbone_name: str,
        ckpt_dir: Path | None = None,
    ) -> None:
        self.backbone_name = backbone_name
        self.ckpt_dir = Path(ckpt_dir or CKPT_V2_DIR)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Cache expensive env queries
        self._git_hash = _get_git_hash()
        self._pip_hash = _get_pip_freeze_hash()

    def _make_path(self, suffix: str) -> Path:
        """Create checkpoint path: {backbone}_v2_{suffix}.pth"""
        return self.ckpt_dir / f"{self.backbone_name}_v2_{suffix}.pth"

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        phase: str,
        metrics: dict[str, Any],
        config: dict[str, Any] | None = None,
        tier_distribution: dict[str, int] | None = None,
        suffix: str | None = None,
    ) -> Path:
        """Save a full checkpoint with metadata.

        Parameters
        ----------
        model : nn.Module
            The DualHeadModel.
        optimizer : Optimizer
        scheduler : LR scheduler
        epoch : int
            Current epoch number.
        phase : str
            "A", "B", or "C".
        metrics : dict
            Current epoch metrics (val_acc, mean_iou, etc.).
        config : dict
            Training configuration for this run.
        tier_distribution : dict
            Count of Tier A/B/C samples used.
        suffix : str
            Checkpoint filename suffix. If None, uses phase + "epoch{N}".

        Returns
        -------
        Path to saved checkpoint.
        """
        if suffix is None:
            suffix = f"phase{phase}_epoch{epoch}"

        ckpt_path = self._make_path(suffix)

        checkpoint = {
            # Model + optimizer state
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,

            # Training progress
            "epoch": epoch,
            "phase": phase,
            "backbone_name": self.backbone_name,

            # Metrics
            "metrics": metrics,
            "best_val_acc": metrics.get("cls_accuracy", 0.0),
            "best_val_iou": metrics.get("mean_iou", 0.0),

            # Configuration
            "training_config": config or {},
            "pseudo_label_tier_distribution": tier_distribution or {},

            # Reproducibility
            "run_seed": RUN_SEED,
            "git_hash": self._git_hash,
            "pip_freeze_hash": self._pip_hash,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",  # type: ignore[attr-defined]
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        torch.save(checkpoint, str(ckpt_path))
        logger.info(
            f"  Checkpoint saved: {ckpt_path.name} "
            f"(phase={phase}, epoch={epoch}, "
            f"acc={metrics.get('cls_accuracy', 0):.4f}, "
            f"mIoU={metrics.get('mean_iou', 0):.4f})"
        )
        return ckpt_path

    def save_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        phase: str,
        metrics: dict[str, Any],
        metric_key: str = "cls_accuracy",
        config: dict[str, Any] | None = None,
        tier_distribution: dict[str, int] | None = None,
    ) -> Path | None:
        """Save checkpoint only if the tracked metric improved.

        Returns the path if saved, None otherwise.
        """
        current_val = metrics.get(metric_key, 0.0)
        best_attr = f"_best_{phase}_{metric_key}"

        prev_best = getattr(self, best_attr, -float("inf"))
        if current_val > prev_best:
            setattr(self, best_attr, current_val)
            suffix = f"phase{phase}_best"
            return self.save(
                model, optimizer, scheduler, epoch, phase,
                metrics, config, tier_distribution, suffix=suffix,
            )
        return None

    def save_final(
        self,
        model: nn.Module,
        metrics: dict[str, Any],
    ) -> Path:
        """Save the final model (copy of best Phase C or last phase)."""
        ckpt_path = self._make_path("final")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "backbone_name": self.backbone_name,
            "metrics": metrics,
            "run_seed": RUN_SEED,
            "git_hash": self._git_hash,
            "pip_freeze_hash": self._pip_hash,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",  # type: ignore[attr-defined]
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(checkpoint, str(ckpt_path))
        logger.info(f"  Final checkpoint saved: {ckpt_path.name}")
        return ckpt_path

    def load(
        self,
        suffix: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        map_location: str = "cpu",
    ) -> dict[str, Any]:
        """Load a specific checkpoint.

        Returns the full checkpoint dict (including metadata).
        """
        ckpt_path = self._make_path(suffix)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(
            f"  Loaded checkpoint: {ckpt_path.name} "
            f"(phase={checkpoint.get('phase')}, epoch={checkpoint.get('epoch')})"
        )
        return checkpoint

    def load_latest(
        self,
        phase: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
    ) -> dict[str, Any] | None:
        """Load the latest (best) checkpoint for a given phase.

        Returns None if no checkpoint found for this phase.
        """
        best_path = self._make_path(f"phase{phase}_best")
        if best_path.exists():
            return self.load(f"phase{phase}_best", model, optimizer, scheduler)
        return None

    def has_checkpoint(self, suffix: str) -> bool:
        """Check if a checkpoint exists."""
        return self._make_path(suffix).exists()

    def list_checkpoints(self) -> list[Path]:
        """List all checkpoints for this backbone."""
        pattern = f"{self.backbone_name}_v2_*.pth"
        return sorted(self.ckpt_dir.glob(pattern))

    def get_v1_accuracy(self) -> float | None:
        """Load V1 best validation accuracy for rollback comparison.

        Reads from V1 checkpoint metadata if available.
        """
        from V2_segmentation.config import V1_CKPT_DIR

        for suffix in ("_finetune_best.pth", "_final.pth", "_head_best.pth"):
            ckpt_path = V1_CKPT_DIR / f"{self.backbone_name}{suffix}"
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                    if isinstance(ckpt, dict):
                        for key in ("best_val_acc", "val_acc", "accuracy"):
                            if key in ckpt:
                                return float(ckpt[key])
                except Exception:
                    pass
        return None


# ============================================================================
#  Rollback log
# ============================================================================

class RollbackLogger:
    """Track rollback decisions for backbones that fail Phase B gate."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = Path(log_path or CKPT_V2_DIR / "rollback_log.json")
        self._entries: list[dict[str, Any]] = []
        if self.log_path.exists():
            with open(self.log_path) as f:
                self._entries = json.load(f)

    def log_rollback(
        self,
        backbone_name: str,
        reason: str,
        v1_acc: float | None,
        v2_acc: float,
        mean_iou: float,
    ) -> None:
        """Record a rollback decision."""
        entry = {
            "backbone": backbone_name,
            "reason": reason,
            "v1_val_acc": v1_acc,
            "v2_val_acc": v2_acc,
            "v2_mean_iou": mean_iou,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._entries.append(entry)
        self._save()
        logger.warning(
            f"  ROLLBACK: {backbone_name} — {reason} "
            f"(V1={v1_acc}, V2={v2_acc:.4f}, mIoU={mean_iou:.4f})"
        )

    def log_pass(
        self,
        backbone_name: str,
        v1_acc: float | None,
        v2_acc: float,
        mean_iou: float,
    ) -> None:
        """Record a successful pass (no rollback needed)."""
        entry = {
            "backbone": backbone_name,
            "reason": "PASSED",
            "v1_val_acc": v1_acc,
            "v2_val_acc": v2_acc,
            "v2_mean_iou": mean_iou,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._entries.append(entry)
        self._save()

    def _save(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self._entries, f, indent=2)

    def get_reverted_backbones(self) -> list[str]:
        """Return names of backbones that were rolled back."""
        return [
            e["backbone"] for e in self._entries
            if e["reason"] != "PASSED"
        ]
