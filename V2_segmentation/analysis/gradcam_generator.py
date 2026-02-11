"""
V2_segmentation/analysis/gradcam_generator.py
==============================================
GradCAM / GradCAM++ heatmap generation for all 15 V1 backbones.

Uses the hook-spec from ``backbone_adapter`` to automatically target the
correct layer for each architecture.  Generates per-class heatmaps that
reveal what spatial regions drive the V1 classifier — critical input for
pseudo-label quality assessment in Phase 1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from V2_segmentation.config import (
    DEVICE, IMG_SIZE,
)
from V2_segmentation.models.backbone_adapter import (
    BACKBONE_HOOK_SPEC, resolve_module,
)

logger = logging.getLogger(__name__)


class GradCAMGenerator:
    """Generate GradCAM heatmaps from V1 backbones.

    Parameters
    ----------
    model : nn.Module
        A V1 backbone (loaded with trained weights).
    backbone_name : str
        Key into BACKBONE_HOOK_SPEC.
    device : torch.device
        Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        backbone_name: str,
        device: torch.device = DEVICE,
    ) -> None:
        self.model = model.to(device).eval()
        self.backbone_name = backbone_name
        self.device = device

        spec = BACKBONE_HOOK_SPEC[backbone_name]
        self._target_module = resolve_module(model, spec.gradcam_layer)
        self._target_spec = spec.high_level  # same layer usually

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd_hook(module, inp, out):
            self._activations = out

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0]

        h1 = self._target_module.register_forward_hook(fwd_hook)
        h2 = self._target_module.register_full_backward_hook(bwd_hook)
        self._handles = [h1, h2]

    def cleanup(self) -> None:
        """Remove hooks and clear cached tensors to free GPU memory."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        # Clear cached tensors to free GPU memory
        self._activations = None
        self._gradients = None

    def _ensure_spatial(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape (B, N, C) → (B, C, H, W) if needed."""
        if tensor.dim() == 3:
            import math
            B, N, C = tensor.shape
            H = int(math.sqrt(N))
            return tensor.transpose(1, 2).reshape(B, C, H, H).contiguous()
        return tensor

    def generate(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate a GradCAM heatmap for a single image.

        Parameters
        ----------
        image : (1, 3, H, W) or (3, H, W)
            Preprocessed input image.
        target_class : int or None
            Class to explain.  If None, uses predicted class.

        Returns
        -------
        heatmap : np.ndarray of shape (H, W), values in [0, 1].
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device).requires_grad_(True)

        # Forward
        self.model.zero_grad()
        output = self.model(image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward from target class score
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=False)

        # Get activations and gradients
        assert self._activations is not None, "Forward hook did not fire"
        assert self._gradients is not None, "Backward hook did not fire"
        activations = self._ensure_spatial(self._activations.detach())
        gradients = self._ensure_spatial(self._gradients.detach())

        # GradCAM: global average pool gradients → weight activations
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze(0).squeeze(0)  # (H, W)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # Resize to input resolution
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        return cam.cpu().numpy()

    def generate_batch(
        self,
        images: torch.Tensor,
        target_classes: list[int] | None = None,
    ) -> list[np.ndarray]:
        """Generate GradCAM heatmaps for a batch of images.

        Parameters
        ----------
        images : (B, 3, H, W)
        target_classes : list of int, or None.

        Returns
        -------
        list of np.ndarray heatmaps.
        """
        heatmaps = []
        for i in range(images.size(0)):
            tc = target_classes[i] if target_classes else None
            heatmaps.append(self.generate(images[i], tc))
        return heatmaps

    def generate_class_average(
        self,
        dataloader: Any,
        class_idx: int,
        max_samples: int = 50,
    ) -> np.ndarray:
        """Average GradCAM heatmap for all samples of a given class.

        Parameters
        ----------
        dataloader : DataLoader yielding (images, labels).
        class_idx : int
            Target class index.
        max_samples : int
            Maximum samples to average over.

        Returns
        -------
        avg_heatmap : np.ndarray (H, W), values in [0, 1].
        """
        accum = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64)
        count = 0

        for images, labels in dataloader:
            for i in range(images.size(0)):
                if labels[i].item() == class_idx and count < max_samples:
                    hm = self.generate(images[i], target_class=class_idx)
                    accum += hm
                    count += 1
            if count >= max_samples:
                break

        if count > 0:
            accum /= count
        return accum.astype(np.float32)

    def save_heatmap(
        self,
        heatmap: np.ndarray,
        save_path: Path,
        original_image: np.ndarray | None = None,
    ) -> None:
        """Save a heatmap as a PNG image (with optional overlay).

        Parameters
        ----------
        heatmap : (H, W) array in [0, 1].
        save_path : Path to save.
        original_image : (H, W, 3) uint8 array for overlay.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(1, 2 if original_image is not None else 1,
                                     figsize=(10, 5))
            if original_image is not None:
                axes[0].imshow(original_image)
                axes[0].set_title("Original")
                axes[0].axis("off")
                axes[1].imshow(original_image)
                axes[1].imshow(heatmap, cmap="jet", alpha=0.5)
                axes[1].set_title("GradCAM Overlay")
                axes[1].axis("off")
            else:
                ax = axes if not hasattr(axes, "__len__") else axes[0]
                ax.imshow(heatmap, cmap="jet")
                ax.set_title("GradCAM")
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved GradCAM: {save_path}")
        except ImportError:
            logger.warning("  matplotlib not available — skipping heatmap save")
