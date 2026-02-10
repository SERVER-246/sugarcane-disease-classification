"""
V2_segmentation/scripts/generate_draft_gold_masks.py
=====================================================
Phase 0.5: Generate draft gold masks using GrabCut → SAM pipeline.

For each gold-labeled image:
  1. GrabCut → initial foreground mask
  2. SAM → refine boundaries (if available)
  3. Auto-assign channels based on disease class
  4. Save as *_mask.npy (5-channel) and *_mask_preview.png (colored overlay)

The annotator then reviews/edits these drafts to create final gold labels.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from V2_segmentation.config import (
    CLASS_NAMES, DISEASE_CHANNEL_MAP, GOLD_LABELS_DIR, NUM_SEG_CHANNELS,
)

logger = logging.getLogger(__name__)


def generate_draft_masks(
    gold_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Generate draft masks for all gold-labeled images.

    Returns dict: class_name → number of masks generated.
    """
    gold_dir = gold_dir or GOLD_LABELS_DIR

    # Lazy import to avoid circular deps
    from V2_segmentation.pseudo_labels.grabcut_generator import GrabCutGenerator
    try:
        from V2_segmentation.pseudo_labels.sam_generator import SAMGenerator
        sam_available = True
    except ImportError:
        sam_available = False

    grabcut = GrabCutGenerator()
    sam = SAMGenerator() if sam_available else None  # type: ignore[possibly-undefined]

    counts: dict[str, int] = {}

    for class_name in sorted(CLASS_NAMES):
        class_dir = gold_dir / class_name
        if not class_dir.exists():
            logger.warning(f"No gold images for {class_name}")
            counts[class_name] = 0
            continue

        images = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            and "_mask" not in p.stem
        )

        count = 0
        for img_path in images:
            mask_npy = class_dir / f"{img_path.stem}_mask.npy"
            mask_preview = class_dir / f"{img_path.stem}_mask_preview.png"

            if mask_npy.exists() and not overwrite:
                count += 1
                continue

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Step 1: GrabCut (takes file path)
                gc_mask = grabcut.generate(img_path)

                # Step 2: SAM refinement (optional, takes file path)
                if sam is not None:
                    try:
                        sam_mask = sam.generate(img_path)
                        if sam_mask is not None:
                            # Blend: SAM refines GrabCut boundaries
                            combined = (0.6 * gc_mask + 0.4 * sam_mask)
                            fg_mask = (combined > 0.5).astype(np.float32)
                        else:
                            fg_mask = gc_mask.astype(np.float32)
                    except Exception:
                        fg_mask = gc_mask.astype(np.float32)
                else:
                    fg_mask = gc_mask.astype(np.float32)

                # Step 3: Auto-assign channels
                mask_5ch = _assign_channels(fg_mask, class_name)

                # Save .npy
                np.save(str(mask_npy), mask_5ch.astype(np.float32))

                # Save preview
                _save_preview(img, mask_5ch, mask_preview)

                count += 1

            except Exception as e:
                logger.warning(f"Failed on {img_path.name}: {e}")

        counts[class_name] = count
        logger.info(f"  {class_name}: {count}/{len(images)} draft masks generated")

    return counts


def _assign_channels(
    fg_mask: np.ndarray,
    class_name: str,
) -> np.ndarray:
    """Assign 5-channel segmentation labels based on disease class.

    Parameters
    ----------
    fg_mask : (H, W) float32, 1.0 = foreground
    class_name : disease class name

    Returns
    -------
    (H, W, 5) float32 mask
    """
    h, w = fg_mask.shape
    mask_5ch = np.zeros((h, w, NUM_SEG_CHANNELS), dtype=np.float32)

    # Channel 0 = background
    bg = 1.0 - fg_mask
    mask_5ch[:, :, 0] = bg

    # Get channel assignment from config
    mapping = DISEASE_CHANNEL_MAP.get(class_name, {})
    primary = mapping.get("primary", 1)  # Default: healthy
    secondary = mapping.get("secondary", None)

    if class_name.lower() == "healthy":
        # All foreground = healthy tissue
        mask_5ch[:, :, 1] = fg_mask
    elif secondary is not None:
        # Split: 70% primary, 30% secondary
        mask_5ch[:, :, primary] = fg_mask * 0.7
        mask_5ch[:, :, secondary] = fg_mask * 0.3
    else:
        mask_5ch[:, :, primary] = fg_mask

    return mask_5ch


def _save_preview(
    img: np.ndarray,
    mask_5ch: np.ndarray,
    save_path: Path,
) -> None:
    """Save colored overlay preview of the 5-channel mask."""
    from V2_segmentation.visualization.seg_overlay import CHANNEL_COLORS

    h, w = img.shape[:2]
    if mask_5ch.shape[:2] != (h, w):
        mask_5ch = cv2.resize(mask_5ch, (w, h))

    dominant = np.argmax(mask_5ch, axis=2)
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for ch, color in enumerate(CHANNEL_COLORS):
        colored[dominant == ch] = color

    overlay = cv2.addWeighted(img, 0.55, colored, 0.45, 0)
    cv2.imwrite(str(save_path), overlay)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("=== Phase 0.5: Generating Draft Gold Masks ===")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    counts = generate_draft_masks(overwrite=args.overwrite)

    print("\nSummary:")
    total = 0
    for cls, n in sorted(counts.items()):
        print(f"  {cls}: {n} masks")
        total += n
    print(f"  TOTAL: {total} draft masks")
    print("\nNext: Review and edit these masks in gold_labels/ to create final gold labels.")


if __name__ == "__main__":
    main()
