"""
V2_segmentation/training/memory_manager.py
==========================================
GPU memory management for dual-head training.

Per-backbone memory profiles ensure no OOM:
  - Tier-specific batch size & gradient accumulation
  - Gradient checkpointing for HEAVY tier
  - Dynamic VRAM monitoring + emergency OOM recovery
  - Effective batch size stays constant (32) across all tiers
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt_util

from V2_segmentation.config import (
    BACKBONE_PROFILES,
    VRAM_SAFE_LIMIT_GB,
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryBudget:
    """Memory budget for one backbone."""
    backbone_name: str
    tier: str
    batch_size: int
    grad_accum_steps: int
    grad_checkpoint: bool
    decoder_channels: int
    estimated_vram_gb: float


class MemoryManager:
    """Manage GPU memory for dual-head training.

    Provides per-backbone memory budgets and OOM recovery.

    Usage
    -----
    >>> mm = MemoryManager()
    >>> budget = mm.get_budget("CustomEfficientNetV4")
    >>> budget.batch_size       # 32
    >>> budget.grad_accum_steps # 1
    >>> mm.apply_grad_checkpoint(model, "CustomViTHybrid")  # enables checkpointing
    """

    def __init__(self) -> None:
        self._budgets: dict[str, MemoryBudget] = {}
        self._build_budgets()

    def _build_budgets(self) -> None:
        """Pre-compute memory budgets from config profiles."""
        for name, profile in BACKBONE_PROFILES.items():
            # V2 adds ~20-40% overhead for seg decoder on top of V1 training memory
            v1_gb = profile["v1_bs32_gb"]
            # Scale V1 memory to the tier's batch size
            tier_bs = profile["batch_size"]
            scaled_v1 = v1_gb * (tier_bs / 32.0)
            # Decoder overhead estimate: ~1-3 GB depending on channels
            decoder_overhead = profile["decoder_channels"] * 0.005  # rough heuristic
            estimated = scaled_v1 + decoder_overhead

            self._budgets[name] = MemoryBudget(
                backbone_name=name,
                tier=profile["tier"],
                batch_size=tier_bs,
                grad_accum_steps=profile["grad_accum"],
                grad_checkpoint=profile["grad_checkpoint"],
                decoder_channels=profile["decoder_channels"],
                estimated_vram_gb=round(estimated, 2),
            )

    def get_budget(self, backbone_name: str) -> MemoryBudget:
        """Get the memory budget for a backbone."""
        if backbone_name not in self._budgets:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. "
                f"Available: {sorted(self._budgets)}"
            )
        return self._budgets[backbone_name]

    def log_budget(self, backbone_name: str) -> None:
        """Log the memory budget for a backbone."""
        b = self.get_budget(backbone_name)
        logger.info(
            f"  Memory budget for {b.backbone_name}:\n"
            f"    Tier:          {b.tier}\n"
            f"    Batch size:    {b.batch_size}\n"
            f"    Grad accum:    {b.grad_accum_steps}  "
            f"(effective BS={b.batch_size * b.grad_accum_steps})\n"
            f"    Grad ckpt:     {b.grad_checkpoint}\n"
            f"    Decoder ch:    {b.decoder_channels}\n"
            f"    Est. VRAM:     {b.estimated_vram_gb:.2f} GB / {VRAM_SAFE_LIMIT_GB:.1f} GB"
        )

    # ── Gradient checkpointing ──────────────────────────────────────────

    @staticmethod
    def apply_grad_checkpoint(model: nn.Module, backbone_name: str) -> bool:
        """Enable gradient checkpointing for HEAVY-tier backbones.

        IMPORTANT: Modules that have forward hooks (used by the dual-head
        feature extractor) must NOT be wrapped — gradient checkpoint
        re-runs forward during backprop, causing hooks to fire twice with
        stale/wrong tensors.  We collect hooked module paths and skip them
        plus all their ancestors.

        Returns True if checkpointing was enabled.
        """
        profile = BACKBONE_PROFILES.get(backbone_name, {})
        if not profile.get("grad_checkpoint", False):
            return False

        enabled = False
        backbone: nn.Module
        if hasattr(model, "backbone"):
            backbone = model.backbone  # type: ignore[assignment]
        else:
            backbone = model

        # ── Collect modules that must NOT be checkpointed ─────────────
        # Any module with a forward hook (registered by BackboneFeatureExtractor)
        # and its ancestor chain must be excluded.  Wrapping an ancestor would
        # re-run the child's forward during backprop, making the hook capture
        # stale tensors — the root cause of NaN collapse in Phase C.
        protected_names: set[str] = set()

        for name, module in backbone.named_modules():
            if getattr(module, "_forward_hooks", None):  # has registered forward hooks
                protected_names.add(name)
                # Also protect all ancestors of the hooked module
                parts = name.split(".")
                for i in range(1, len(parts)):
                    protected_names.add(".".join(parts[:i]))

        # Also protect the root backbone itself ("" path)
        protected_names.add("")

        if protected_names:
            logger.debug(
                f"  Grad-ckpt: protecting {len(protected_names)} modules with hooks "
                f"or ancestors: {sorted(protected_names)[:10]}..."
            )

        # ── Apply checkpointing to safe modules only ─────────────────
        wrapped_count = 0
        for name, module in backbone.named_modules():
            # Skip protected modules
            if name in protected_names:
                continue

            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()  # type: ignore[operator]
                enabled = True
            else:
                # Check BOTH the module path name AND the class name for
                # transformer/block patterns.  SwinTransformer's modules have
                # paths like "layers.0.0" but class names like
                # "SwinTransformerBlock", so checking only the path misses them.
                class_name = type(module).__name__.lower()
                path_name = name.lower()
                is_ckpt_candidate = (
                    "transformer" in path_name or "block" in path_name
                    or "transformer" in class_name or "block" in class_name
                )
                if is_ckpt_candidate:
                    # Wrap forward with checkpoint
                    if hasattr(module, "forward") and not getattr(module, "_ckpt_wrapped", False):
                        original_forward = module.forward

                        def make_ckpt_forward(orig_fn):
                            def ckpt_forward(*args, **kwargs):
                                return ckpt_util.checkpoint(
                                    orig_fn, *args, use_reentrant=False, **kwargs
                                )
                            return ckpt_forward

                        module.forward = make_ckpt_forward(original_forward)
                        module._ckpt_wrapped = True  # type: ignore[assignment]
                        wrapped_count += 1
                        enabled = True

        if enabled:
            logger.info(
                f"  Gradient checkpointing ENABLED for {backbone_name} "
                f"({wrapped_count} modules wrapped, "
                f"{len(protected_names)} hook-protected modules excluded)"
            )
        return enabled

    # ── VRAM monitoring ─────────────────────────────────────────────────

    @staticmethod
    def get_vram_usage() -> dict[str, float]:
        """Get current VRAM usage in GB.

        Returns
        -------
        dict with keys: allocated, reserved, free, total.
        """
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free = total - allocated

        return {
            "allocated": round(allocated, 3),
            "reserved": round(reserved, 3),
            "free": round(free, 3),
            "total": round(total, 3),
        }

    @staticmethod
    def log_vram(label: str = "") -> None:
        """Log current VRAM usage."""
        if not torch.cuda.is_available():
            return
        usage = MemoryManager.get_vram_usage()
        logger.info(
            f"  VRAM {label}: "
            f"{usage['allocated']:.2f}GB alloc / "
            f"{usage['reserved']:.2f}GB reserved / "
            f"{usage['free']:.2f}GB free / "
            f"{usage['total']:.2f}GB total"
        )

    @staticmethod
    def emergency_cleanup() -> None:
        """Emergency GPU memory cleanup after OOM."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.warning("  Emergency VRAM cleanup performed")

    # ── Gradient accumulation helper ────────────────────────────────────

    @staticmethod
    def should_step(batch_idx: int, accum_steps: int) -> bool:
        """Whether optimizer should step at this batch index.

        Parameters
        ----------
        batch_idx : int
            0-based index within the epoch.
        accum_steps : int
            Gradient accumulation steps.

        Returns
        -------
        True if this is an accumulation boundary.
        """
        return (batch_idx + 1) % accum_steps == 0

    @staticmethod
    def scale_loss(loss: torch.Tensor, accum_steps: int) -> torch.Tensor:
        """Scale loss for gradient accumulation.

        Parameters
        ----------
        loss : scalar tensor.
        accum_steps : int.

        Returns
        -------
        loss / accum_steps
        """
        if accum_steps > 1:
            return loss / accum_steps
        return loss
