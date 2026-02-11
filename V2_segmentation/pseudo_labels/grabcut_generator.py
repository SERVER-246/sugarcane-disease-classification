"""
V2_segmentation/pseudo_labels/grabcut_generator.py
===================================================
OpenCV GrabCut-based foreground extraction for pseudo-label generation.

Uses green-channel intensity to seed the initial foreground mask (sugarcane
leaves are predominantly green). The GrabCut algorithm refines the boundary
between plant tissue and background.

Output per image: a binary foreground mask (H, W) float32 in [0, 1].
This is one of three inputs to the mask combiner.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from V2_segmentation.config import IMG_SIZE

logger = logging.getLogger(__name__)


class GrabCutGenerator:
    """Generate foreground masks using OpenCV GrabCut.

    Workflow per image:
      1. Convert to LAB colour space → extract green-dominant regions
      2. Create initial trimap: definite FG / probable FG / probable BG / definite BG
      3. Run GrabCut (5 iterations) to refine
      4. Output binary foreground mask

    Parameters
    ----------
    n_iters : int
        Number of GrabCut iterations (default 5).
    green_threshold : float
        Fraction of image considered "green enough" for initial FG seed (0-1).
        Lower = more aggressive seeding. Default 0.35.
    border_margin : int
        Pixels around the image border forced to background.
    """

    def __init__(
        self,
        n_iters: int = 5,
        green_threshold: float = 0.35,
        border_margin: int = 10,
    ) -> None:
        self.n_iters = n_iters
        self.green_threshold = green_threshold
        self.border_margin = border_margin

    def generate(self, image_path: str | Path) -> np.ndarray:
        """Generate a foreground mask for a single image.

        Parameters
        ----------
        image_path : str or Path
            Path to the input image.

        Returns
        -------
        mask : (H, W) float32 in [0, 1]
            Foreground probability mask.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Cannot read image: {image_path}")
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # ── Step 1: Green channel analysis for initial seed ──────────
        trimap = self._create_trimap(img)

        # ── Step 2: Run GrabCut ──────────────────────────────────────
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                img, trimap, (0, 0, 0, 0),
                bgd_model, fgd_model,
                self.n_iters,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error as e:
            logger.warning(f"GrabCut failed for {image_path}: {e}")
            return self._fallback_green_mask(img)

        # ── Step 3: Convert GrabCut result to binary mask ────────────
        # GC_FGD=1, GC_PR_FGD=3 are foreground; GC_BGD=0, GC_PR_BGD=2 are background
        fg_mask = np.where(
            (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD), 1.0, 0.0
        ).astype(np.float32)

        # ── Step 4: Morphological cleanup ────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        return fg_mask

    def _create_trimap(self, img_bgr: np.ndarray) -> np.ndarray:
        """Create initial trimap from green channel dominance.

        Returns (H, W) uint8 with GrabCut label values.
        """
        H, W = img_bgr.shape[:2]

        # Convert to HSV for green detection
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Green hue range: ~35-85 in OpenCV's 0-179 scale
        green_mask = ((h >= 25) & (h <= 95) & (s >= 30) & (v >= 30)).astype(np.float32)

        # Also consider high-saturation regions (diseased tissue is often saturated)
        saturated = (s >= 50).astype(np.float32)

        # Combined: green OR saturated non-dark
        plant_prob = np.clip(green_mask + 0.5 * saturated, 0, 1)

        # Adaptive threshold: if very little green, lower the bar
        green_fraction = green_mask.mean()
        if green_fraction < 0.05:
            # Very little green — use broader threshold (edge case: red_rot, wilt)
            plant_prob = ((s >= 25) & (v >= 40)).astype(np.float32)

        # Create trimap
        trimap = np.full((H, W), cv2.GC_PR_BGD, dtype=np.uint8)

        # Probable foreground: where plant signal is strong
        trimap[plant_prob > self.green_threshold] = cv2.GC_PR_FGD

        # Definite foreground: where signal is very strong (core of leaves)
        trimap[plant_prob > 0.7] = cv2.GC_FGD

        # Definite background: image borders
        m = self.border_margin
        if m > 0:
            trimap[:m, :] = cv2.GC_BGD
            trimap[-m:, :] = cv2.GC_BGD
            trimap[:, :m] = cv2.GC_BGD
            trimap[:, -m:] = cv2.GC_BGD

        # Safety: need at least some FG and BG pixels
        if (trimap == cv2.GC_FGD).sum() == 0:
            # Force center as probable FG
            ch, cw = H // 2, W // 2
            r = min(H, W) // 4
            trimap[ch - r:ch + r, cw - r:cw + r] = cv2.GC_PR_FGD

        return trimap

    def _fallback_green_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Fallback: simple green-channel threshold if GrabCut fails."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        green_mask = ((h >= 25) & (h <= 95) & (s >= 30) & (v >= 30)).astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        return green_mask

    def generate_batch(
        self,
        image_paths: list[str | Path],
    ) -> list[np.ndarray]:
        """Generate foreground masks for a batch of images."""
        return [self.generate(p) for p in image_paths]

    def generate_for_split(
        self,
        split_dir: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Generate GrabCut masks for an entire dataset split.

        Parameters
        ----------
        split_dir : Path
            e.g. split_dataset/train/
        output_dir : Path
            Where to save masks (e.g. segmentation_masks/grabcut/train/)

        Returns
        -------
        dict with stats: total, success, failed, per_class counts.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        stats: dict[str, Any] = {"total": 0, "success": 0, "failed": 0, "per_class": {}}

        # Count total images for progress bar
        all_images = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    all_images.append((class_dir.name, img_path))

        # Process with progress bar
        pbar = tqdm(all_images, desc="GrabCut masks", unit="img")
        for class_name, img_path in pbar:
            pbar.set_postfix({"class": class_name[:12], "file": img_path.stem[:15]})

            class_out = output_dir / class_name
            class_out.mkdir(parents=True, exist_ok=True)

            stats["total"] += 1
            try:
                mask = self.generate(img_path)
                out_path = class_out / f"{img_path.stem}_grabcut.npy"
                np.save(str(out_path), mask)
                stats["success"] += 1
                stats["per_class"][class_name] = stats["per_class"].get(class_name, 0) + 1
            except Exception as e:
                logger.error(f"GrabCut failed for {img_path}: {e}")
                stats["failed"] += 1

        logger.info(
            f"GrabCut generation complete: "
            f"{stats['success']}/{stats['total']} succeeded, "
            f"{stats['failed']} failed"
        )
        return stats
