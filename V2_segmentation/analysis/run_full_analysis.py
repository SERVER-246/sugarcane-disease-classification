"""
V2_segmentation/analysis/run_full_analysis.py
=============================================
Phase 0 orchestrator: runs GradCAM, feature analysis, error patterns,
and class attention maps for all 15 V1 backbones.

Usage
-----
    python -m V2_segmentation.analysis.run_full_analysis
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

from V2_segmentation.config import (
    ANALYSIS_DIR, BACKBONES,
    DEVICE, IMG_SIZE, NUM_CLASSES, V1_CKPT_DIR,
)
from V2_segmentation.models.model_factory import create_v1_backbone, load_v1_weights
from V2_segmentation.analysis.gradcam_generator import GradCAMGenerator
from V2_segmentation.analysis.feature_analyzer import FeatureAnalyzer
from V2_segmentation.analysis.error_pattern_analysis import ErrorPatternAnalyzer
from V2_segmentation.analysis.class_attention_maps import ClassAttentionMapper

logger = logging.getLogger(__name__)


def _find_best_checkpoint(backbone_name: str) -> Path | None:
    """Find the best V1 checkpoint for a backbone.

    Looks for (in priority order):
      1. {name}_finetune_best.pth
      2. {name}_final.pth
      3. {name}_head_best.pth
    """
    for suffix in ("_finetune_best.pth", "_final.pth", "_head_best.pth"):
        path = V1_CKPT_DIR / f"{backbone_name}{suffix}"
        if path.exists():
            return path
    return None


def _create_val_dataloader(batch_size: int = 32):
    """Create a validation dataloader from the split dataset.

    Uses the same transforms as V1 validation.
    """
    from V2_segmentation.config import SPLIT_DIR
    from torchvision import datasets, transforms

    val_dir = SPLIT_DIR / "val"
    if not val_dir.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}. "
            "Run the V1 pipeline first to create train/val/test splits."
        )

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_ds = datasets.ImageFolder(str(val_dir), transform=transform)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return loader


def run_analysis_for_backbone(
    backbone_name: str,
    dataloader: torch.utils.data.DataLoader,
    output_dir: Path | None = None,
    skip_gradcam: bool = False,
    skip_features: bool = False,
    skip_errors: bool = False,
    skip_attention: bool = False,
) -> dict:
    """Run full Phase 0 analysis for a single backbone.

    Returns
    -------
    dict with analysis results summary.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  Phase 0 Analysis: {backbone_name}")
    logger.info(f"{'='*60}")

    # 1. Load model
    ckpt_path = _find_best_checkpoint(backbone_name)
    if ckpt_path is None:
        logger.warning(f"  No checkpoint found for {backbone_name} — skipping")
        return {"backbone": backbone_name, "status": "no_checkpoint"}

    model = create_v1_backbone(backbone_name, NUM_CLASSES)
    load_v1_weights(model, ckpt_path, strict=False)
    model.to(DEVICE).eval()

    out_dir = Path(output_dir or ANALYSIS_DIR / backbone_name)
    results: dict[str, object] = {"backbone": backbone_name, "checkpoint": str(ckpt_path)}

    # 2. Error pattern analysis
    if not skip_errors:
        try:
            logger.info(f"  Running error pattern analysis...")
            analyzer = ErrorPatternAnalyzer(model, backbone_name)
            preds, labels, probs = analyzer.run_inference(dataloader)
            analyzer.save_analysis(preds, labels, probs, out_dir)
            metrics = analyzer.per_class_metrics(preds, labels)
            overall_acc = (preds == labels).mean()
            results["accuracy"] = float(overall_acc)
            results["per_class_f1"] = {
                k: v["f1"] for k, v in metrics.items()
            }
        except Exception as e:
            logger.error(f"  Error analysis failed: {e}")
            results["error_analysis"] = str(e)

    # 3. Feature analysis
    if not skip_features:
        try:
            logger.info(f"  Running feature analysis...")
            feat_analyzer = FeatureAnalyzer(model, backbone_name)
            features, feat_labels = feat_analyzer.extract_features(dataloader)
            feat_analyzer.save_analysis(features, feat_labels, out_dir)
            feat_analyzer.cleanup()
        except Exception as e:
            logger.error(f"  Feature analysis failed: {e}")
            results["feature_analysis"] = str(e)

    # 4. GradCAM (sample-based)
    if not skip_gradcam:
        try:
            logger.info(f"  Generating GradCAM samples...")
            gradcam = GradCAMGenerator(model, backbone_name)
            # Generate for first few samples
            for images, targets in dataloader:
                for i in range(min(5, images.size(0))):
                    hm = gradcam.generate(images[i], target_class=targets[i].item())
                    save_path = out_dir / "gradcam_samples" / f"sample_{i}_class{targets[i].item()}.png"
                    gradcam.save_heatmap(hm, save_path)
                break
            gradcam.cleanup()
        except Exception as e:
            logger.error(f"  GradCAM failed: {e}")
            results["gradcam"] = str(e)

    # 5. Class attention maps
    if not skip_attention:
        try:
            logger.info(f"  Generating class attention maps...")
            mapper = ClassAttentionMapper(model, backbone_name, max_samples_per_class=30)
            class_maps = mapper.generate_all_class_maps(dataloader)
            mapper.save_attention_maps(class_maps, out_dir / "attention_maps")
            mapper.cleanup()
        except Exception as e:
            logger.error(f"  Class attention maps failed: {e}")
            results["attention_maps"] = str(e)

    # Cleanup GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results["status"] = "complete"
    return results


def run_full_analysis(
    backbones: list[str] | None = None,
    skip_gradcam: bool = False,
    skip_features: bool = False,
    skip_errors: bool = False,
    skip_attention: bool = False,
) -> list[dict]:
    """Run Phase 0 analysis for all (or selected) backbones.

    Parameters
    ----------
    backbones : list of backbone names.  None = all 15.

    Returns
    -------
    list of per-backbone result dicts.
    """
    backbones = backbones or BACKBONES

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Phase 0 Analysis — {len(backbones)} backbones")
    logger.info(f"Output: {ANALYSIS_DIR}")

    # Create shared dataloader
    loader = _create_val_dataloader(batch_size=32)

    all_results = []
    for name in backbones:
        result = run_analysis_for_backbone(
            name, loader,
            skip_gradcam=skip_gradcam,
            skip_features=skip_features,
            skip_errors=skip_errors,
            skip_attention=skip_attention,
        )
        all_results.append(result)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("  Phase 0 Analysis — Summary")
    logger.info(f"{'='*60}")
    for r in all_results:
        acc = r.get("accuracy", "N/A")
        status = r.get("status", "unknown")
        logger.info(f"  {r['backbone']:30s}  acc={acc}  status={status}")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Parse optional backbone filter from CLI
    selected = sys.argv[1:] if len(sys.argv) > 1 else None
    run_full_analysis(backbones=selected)
