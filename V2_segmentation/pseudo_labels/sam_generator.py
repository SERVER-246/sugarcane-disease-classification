"""
V2_segmentation/pseudo_labels/sam_generator.py
===============================================
Optional SAM (Segment Anything Model) zero-shot mask generation.

If SAM weights are available, produces high-quality boundary masks.
If unavailable, returns None and the combiner falls back to 2-source fusion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np


from V2_segmentation.config import IMG_SIZE, DEVICE

logger = logging.getLogger(__name__)

# Try to import SAM — it's optional
_SAM_AVAILABLE = False
try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # type: ignore[import-unresolved]
    _SAM_AVAILABLE = True  # type: ignore[misc]
except ImportError:
    logger.info("SAM (segment_anything) not installed — SAM masks will be skipped")


class SAMGenerator:
    """Zero-shot segmentation using Segment Anything Model.

    Falls back gracefully if SAM is not installed or weights are missing.

    Parameters
    ----------
    model_type : str
        SAM model variant: ``"vit_h"``, ``"vit_l"``, or ``"vit_b"``.
    checkpoint_path : str or Path or None
        Path to SAM checkpoint. If None, auto-searches common locations.
    min_mask_area : int
        Minimum mask area in pixels to keep.
    """

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: str | Path | None = None,
        min_mask_area: int = 500,
    ) -> None:
        self.model_type = model_type
        self.min_mask_area = min_mask_area
        self.available = False
        self._generator = None

        if not _SAM_AVAILABLE:
            logger.warning("SAM not available — SAMGenerator disabled")
            return

        # Search for checkpoint
        ckpt = self._find_checkpoint(checkpoint_path)
        if ckpt is None:
            logger.warning("SAM checkpoint not found — SAMGenerator disabled")
            return

        try:
            if not _SAM_AVAILABLE:
                raise RuntimeError("segment_anything not installed")
            sam = sam_model_registry[model_type](checkpoint=str(ckpt))  # type: ignore[possibly-undefined]
            sam.to(DEVICE)
            self._generator = SamAutomaticMaskGenerator(  # type: ignore[possibly-undefined]
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=min_mask_area,
            )
            self.available = True
            logger.info(f"SAM initialized: {model_type} from {ckpt}")
        except Exception as e:
            logger.error(f"SAM initialization failed: {e}")

    def _find_checkpoint(self, path: str | Path | None) -> Path | None:
        """Search for SAM checkpoint in common locations."""
        if path is not None:
            p = Path(path)
            return p if p.exists() else None

        from V2_segmentation.config import BASE_DIR

        search_paths = [
            BASE_DIR / "pretrained_weights" / f"sam_{self.model_type}.pth",
            BASE_DIR / "pretrained_weights" / "sam_vit_b_01ec64.pth",
            BASE_DIR / "pretrained_weights" / "sam_vit_l_0b3195.pth",
            BASE_DIR / "pretrained_weights" / "sam_vit_h_4b8939.pth",
            Path.home() / ".cache" / "sam" / f"sam_{self.model_type}.pth",
        ]
        for p in search_paths:
            if p.exists():
                return p
        return None

    def generate(self, image_path: str | Path) -> np.ndarray | None:
        """Generate SAM masks for a single image.

        Returns
        -------
        mask : (H, W) float32 in [0, 1] or None if SAM unavailable.
            Foreground probability mask (union of top SAM segments).
        """
        if not self.available or self._generator is None:
            return None

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Cannot read image: {image_path}")
            return None

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            masks = self._generator.generate(img_rgb)
        except Exception as e:
            logger.warning(f"SAM generation failed for {image_path}: {e}")
            return None

        if not masks:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        # Sort by area (largest first), take top masks that cover most of image
        masks = sorted(masks, key=lambda m: m["area"], reverse=True)

        # Build foreground: union of large masks with high stability
        fg = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        total_area = IMG_SIZE * IMG_SIZE

        for m in masks:
            if m["area"] < self.min_mask_area:
                continue
            # Weight by stability score
            weight = m.get("stability_score", 0.9)
            fg = np.maximum(fg, m["segmentation"].astype(np.float32) * weight)

            # Stop if we've covered enough
            coverage = (fg > 0.5).sum() / total_area
            if coverage > 0.85:
                break

        return fg

    def generate_for_split(
        self,
        split_dir: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Generate SAM masks for an entire dataset split.

        Returns stats dict or None if SAM unavailable.
        """
        if not self.available:
            logger.info("SAM unavailable — skipping split generation")
            return {"status": "skipped", "reason": "SAM not available"}

        output_dir.mkdir(parents=True, exist_ok=True)
        stats: dict[str, Any] = {"total": 0, "success": 0, "failed": 0}

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_out = output_dir / class_dir.name
            class_out.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                stats["total"] += 1
                try:
                    mask = self.generate(img_path)
                    if mask is not None:
                        np.save(str(class_out / f"{img_path.stem}_sam.npy"), mask)
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
                except Exception as e:
                    logger.error(f"SAM failed for {img_path}: {e}")
                    stats["failed"] += 1

        logger.info(f"SAM generation: {stats['success']}/{stats['total']} succeeded")
        return stats
