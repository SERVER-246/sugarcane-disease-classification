"""
V2_segmentation/scripts/sample_gold_set.py
============================================
Phase 0.5: Sample ~200 images for gold-label annotation.

Sampling strategy (from sprint plan):
  - ~15 images per class from train split
  - Oversample rare/difficult classes (Wilt, Grassy_shoot_disease, Smut) to 20
  - Uses stratified random sampling with fixed seed for reproducibility
  - Copies selected images to gold_labels/{class_name}/
"""

from __future__ import annotations

import logging
import random
import shutil
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from V2_segmentation.config import (
    CLASS_NAMES, GOLD_LABELS_DIR, SPLIT_DIR,
)

logger = logging.getLogger(__name__)

# Classes that need oversampling (rare or diagnostically challenging)
OVERSAMPLE_CLASSES = {
    "Wilt", "Grassy_shoot_disease", "Smut",
    "Leaf_flecking", "Leaf_scorching",
}

DEFAULT_PER_CLASS = 15
OVERSAMPLE_PER_CLASS = 20
SEED = 42


def sample_gold_set(
    split: str = "train",
    per_class: int = DEFAULT_PER_CLASS,
    oversample_per_class: int = OVERSAMPLE_PER_CLASS,
    oversample_classes: set[str] | None = None,
    output_dir: Path | None = None,
    seed: int = SEED,
    dry_run: bool = False,
) -> dict[str, list[Path]]:
    """Sample images for gold-label annotation.

    Parameters
    ----------
    split : str
        Which split to sample from (default: 'train').
    per_class : int
        Default number of images per class.
    oversample_per_class : int
        Number of images for oversampled classes.
    oversample_classes : set or None
        Classes to oversample. Defaults to OVERSAMPLE_CLASSES.
    output_dir : Path or None
        Where to copy images.
    dry_run : bool
        If True, only report what would be sampled.

    Returns
    -------
    dict : class_name → list of sampled image paths.
    """
    random.seed(seed)

    source_dir = SPLIT_DIR / split
    output_dir = output_dir or GOLD_LABELS_DIR
    oversample_classes = oversample_classes or OVERSAMPLE_CLASSES

    if not source_dir.exists():
        logger.error(f"Source split directory not found: {source_dir}")
        return {}

    selected: dict[str, list[Path]] = {}
    total = 0

    for class_name in sorted(CLASS_NAMES):
        class_dir = source_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            selected[class_name] = []
            continue

        # Collect all images
        images = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
        )

        if not images:
            logger.warning(f"No images in {class_dir}")
            selected[class_name] = []
            continue

        # Determine sample size
        n = oversample_per_class if class_name in oversample_classes else per_class
        n = min(n, len(images))  # Don't oversample beyond available

        sampled = random.sample(images, n)
        selected[class_name] = sampled
        total += len(sampled)

        # Copy to output
        if not dry_run:
            out_class = output_dir / class_name
            out_class.mkdir(parents=True, exist_ok=True)

            for img_path in sampled:
                dst = out_class / img_path.name
                if not dst.exists():
                    shutil.copy2(str(img_path), str(dst))

        logger.info(
            f"  {class_name}: sampled {len(sampled)}/{len(images)}"
            f" {'(oversampled)' if class_name in oversample_classes else ''}"
        )

    logger.info(f"Total gold-label samples: {total}")

    # Save manifest
    if not dry_run:
        manifest_path = output_dir / "gold_manifest.txt"
        with open(manifest_path, "w") as f:
            f.write(f"# Gold Label Sample Manifest (seed={seed})\n")
            f.write(f"# Total: {total} images\n\n")
            for cls, paths in sorted(selected.items()):
                f.write(f"\n## {cls} ({len(paths)} images)\n")
                for p in paths:
                    f.write(f"  {p.name}\n")
        logger.info(f"Manifest saved: {manifest_path}")

    return selected


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("=== Phase 0.5: Sampling Gold-Label Set ===")

    import argparse
    parser = argparse.ArgumentParser(description="Sample gold-label set")
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying")
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--oversample", type=int, default=OVERSAMPLE_PER_CLASS)
    args = parser.parse_args()

    selected = sample_gold_set(
        per_class=args.per_class,
        oversample_per_class=args.oversample,
        dry_run=args.dry_run,
    )

    print(f"\n{'DRY RUN - ' if args.dry_run else ''}Summary:")
    for cls, paths in sorted(selected.items()):
        print(f"  {cls}: {len(paths)} images")
    print(f"  TOTAL: {sum(len(v) for v in selected.values())}")


if __name__ == "__main__":
    main()
