"""
V2_segmentation/pseudo_labels/gradcam_mask_generator.py
========================================================
GradCAM-based pseudo-label generation from all 15 V1 backbones.

For each image, this module:
  1. Loads each V1 backbone (one at a time to stay within VRAM)
  2. Runs GradCAM on the backbone's GradCAM target layer
  3. Produces a per-backbone activation heatmap
  4. Ensembles all 15 heatmaps into a robust foreground mask

The ensemble GradCAM is the primary signal (weight 0.5) for mask fusion.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from PIL import Image
from torchvision.transforms import functional as TF

from V2_segmentation.config import (
    BACKBONES, DEVICE, IMG_SIZE, NUM_CLASSES,
    V1_CKPT_DIR,
)
from V2_segmentation.analysis.gradcam_generator import GradCAMGenerator
from V2_segmentation.models.model_factory import create_v1_backbone, load_v1_weights

logger = logging.getLogger(__name__)


def _preprocess_image(image_path: str | Path) -> torch.Tensor:
    """Load and preprocess a single image for inference."""
    img = Image.open(str(image_path)).convert("RGB")
    img_t = TF.resize(img, [IMG_SIZE, IMG_SIZE])  # type: ignore[arg-type]
    img_t = TF.to_tensor(img_t)  # type: ignore[arg-type]
    img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return img_t.unsqueeze(0)  # (1, 3, H, W)


def _find_v1_checkpoint(backbone_name: str) -> Path | None:
    """Find best available V1 checkpoint."""
    for suffix in ("_finetune_best.pth", "_final.pth", "_head_best.pth"):
        path = V1_CKPT_DIR / f"{backbone_name}{suffix}"
        if path.exists():
            return path
    return None


class GradCAMMaskGenerator:
    """Generate ensemble GradCAM masks from all V1 backbones.

    Workflow:
      1. For each backbone, load weights → GradCAM → heatmap
      2. Aggregate: pixel-wise mean across all backbone heatmaps
      3. Threshold to produce foreground mask
      4. Assign to disease channels based on class label

    Parameters
    ----------
    backbones : list[str] or None
        Which backbones to use. None = all 15.
    device : torch.device
        Computation device.
    heatmap_threshold : float
        Threshold for converting ensemble heatmap to binary mask.
    """

    def __init__(
        self,
        backbones: list[str] | None = None,
        device: torch.device = DEVICE,
        heatmap_threshold: float = 0.3,
    ) -> None:
        self.backbones = backbones or BACKBONES
        self.device = device
        self.heatmap_threshold = heatmap_threshold
        self._available_backbones: list[str] = []
        self._check_availability()

    def _check_availability(self) -> None:
        """Check which backbones have V1 checkpoints available."""
        self._available_backbones = []
        for name in self.backbones:
            ckpt = _find_v1_checkpoint(name)
            if ckpt is not None:
                self._available_backbones.append(name)
            else:
                logger.warning(f"No V1 checkpoint for {name} — skipping")
        logger.info(
            f"GradCAM generator: {len(self._available_backbones)}/{len(self.backbones)} "
            f"backbones available"
        )

    def generate_single_backbone(
        self,
        image_path: str | Path,
        backbone_name: str,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate GradCAM heatmap from a single V1 backbone.

        Returns (H, W) float32 heatmap in [0, 1].
        """
        ckpt_path = _find_v1_checkpoint(backbone_name)
        if ckpt_path is None:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        model = None
        generator = None
        img_tensor = None
        heatmap = None

        try:
            # Load model on CPU first, then move to GPU
            model = create_v1_backbone(backbone_name, NUM_CLASSES)
            load_v1_weights(model, ckpt_path, strict=False, map_location="cpu")
            model.to(self.device).eval()

            # Generate GradCAM
            generator = GradCAMGenerator(model, backbone_name, self.device)
            img_tensor = _preprocess_image(image_path).to(self.device)
            heatmap = generator.generate(img_tensor, target_class=target_class)

            # Ensure heatmap is on CPU as numpy before cleanup
            if isinstance(heatmap, torch.Tensor):
                heatmap = heatmap.cpu().numpy()

            return heatmap

        finally:
            # Aggressive cleanup - order matters!
            if generator is not None:
                generator.cleanup()
            if img_tensor is not None:
                del img_tensor
            if model is not None:
                # Move model to CPU before deleting to free GPU memory
                model.cpu()
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

    def generate_ensemble(
        self,
        image_path: str | Path,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate ensemble GradCAM heatmap from all available V1 backbones.

        Loads one backbone at a time to stay within VRAM budget.

        Returns (H, W) float32 ensemble heatmap in [0, 1].

        Raises
        ------
        RuntimeError
            If no backbones successfully generate a heatmap.
        """
        heatmaps: list[np.ndarray] = []

        for backbone_name in self._available_backbones:
            try:
                hm = self.generate_single_backbone(
                    image_path, backbone_name, target_class
                )
                # Only count non-zero heatmaps as success
                if hm.max() > 0:
                    heatmaps.append(hm)
            except Exception as e:
                logger.warning(
                    f"GradCAM failed for {backbone_name} on {image_path}: {e}"
                )
                continue

        if not heatmaps:
            # Raise instead of silently returning zeros
            raise RuntimeError(
                f"No GradCAM heatmaps generated for {image_path} "
                f"(tried {len(self._available_backbones)} backbones)"
            )

        # Pixel-wise mean across backbones
        ensemble = np.mean(heatmaps, axis=0)
        return ensemble.astype(np.float32)

    def generate_foreground_mask(
        self,
        image_path: str | Path,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate binary foreground mask from ensemble GradCAM.

        Returns (H, W) float32 binary mask.
        """
        ensemble = self.generate_ensemble(image_path, target_class)
        binary = (ensemble >= self.heatmap_threshold).astype(np.float32)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    def generate_for_split(
        self,
        split_dir: Path,
        output_dir: Path,
        max_backbones: int | None = None,
    ) -> dict[str, Any]:
        """Generate ensemble GradCAM masks for an entire split.

        Parameters
        ----------
        split_dir : Path
            e.g. split_dataset/train/
        output_dir : Path
            Where to save (e.g. segmentation_masks/gradcam/train/)
        max_backbones : int or None
            Limit number of backbones used (for speed). None = all.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if max_backbones is not None:
            original = self._available_backbones
            self._available_backbones = original[:max_backbones]

        stats: dict[str, Any] = {"total": 0, "success": 0, "failed": 0}

        # Count total images for progress bar
        all_images = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    all_images.append((class_dir.name, img_path))

        # Process with progress bar
        n_backbones = len(self._available_backbones)
        pbar = tqdm(all_images, desc=f"GradCAM ({n_backbones} backbones)", unit="img")
        cleanup_interval = 50  # Periodic GPU cleanup every N images
        
        for idx, (class_name, img_path) in enumerate(pbar):
            pbar.set_postfix({"class": class_name[:12], "file": img_path.stem[:15]})

            class_out = output_dir / class_name
            class_out.mkdir(parents=True, exist_ok=True)

            stats["total"] += 1
            try:
                heatmap = self.generate_ensemble(img_path)
                mask = (heatmap >= self.heatmap_threshold).astype(np.float32)
                np.save(str(class_out / f"{img_path.stem}_gradcam.npy"), heatmap)
                np.save(str(class_out / f"{img_path.stem}_gradcam_mask.npy"), mask)
                stats["success"] += 1
            except Exception as e:
                logger.error(f"Ensemble GradCAM failed for {img_path}: {e}")
                stats["failed"] += 1
            
            # Periodic aggressive GPU cleanup to prevent memory accumulation
            if (idx + 1) % cleanup_interval == 0 and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        if max_backbones is not None:
            pass  # _available_backbones already restored above

        logger.info(
            f"GradCAM mask generation: {stats['success']}/{stats['total']} succeeded"
        )
        return stats
