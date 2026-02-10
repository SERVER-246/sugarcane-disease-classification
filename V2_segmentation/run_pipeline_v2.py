"""
V2_segmentation/run_pipeline_v2.py
====================================
Main entry point for the entire V2 segmentation pipeline.

Orchestrates all phases end-to-end:
  Phase 0   : Analysis (GradCAM on V1 → identify key regions)
  Phase 0.5 : Gold labels (sample → draft masks → calibrate gate)
  Phase 1   : Pseudo-labels (GrabCut + GradCAM + SAM → fusion → quality → tiers)
  Phase 2   : Model setup (already done — backbone adapter, decoder, dual head)
  Phase 3   : Training (3-phase A/B/C per backbone, 15 backbones in waves)
  Phase 4   : Ensemble V2 (12-stage pipeline)
  Phase 5   : Evaluation & validation (leakage, overfit, audit)
  Phase 6   : Visualization (overlays, curves, comparisons)

Usage:
    python run_pipeline_v2.py                    # Full pipeline
    python run_pipeline_v2.py --phase 1          # Only pseudo-labels
    python run_pipeline_v2.py --phase 3          # Only training
    python run_pipeline_v2.py --phase 0.5 1 3    # Multiple phases
    python run_pipeline_v2.py --dry-run           # Preview without executing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from V2_segmentation.config import (
    BASE_DIR, CLASS_NAMES, GOLD_LABELS_DIR,
    PSEUDO_LABELS_DIR, CKPT_V2_DIR, SPLIT_DIR,
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class PipelineV2:
    """End-to-end V2 segmentation pipeline orchestrator."""

    PHASES = ["0", "0.5", "1", "2", "3", "4", "5", "6"]

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.results: dict[str, dict] = {}
        self.start_time = time.time()

    def run(self, phases: list[str] | None = None) -> dict:
        """Run selected phases (or all if None)."""
        phases = phases or self.PHASES

        logger.info("=" * 70)
        logger.info("  V2 SEGMENTATION PIPELINE")
        logger.info(f"  Phases: {phases}")
        logger.info(f"  Dry run: {self.dry_run}")
        logger.info("=" * 70)

        for phase in phases:
            logger.info(f"\n{'='*60}")
            logger.info(f"  PHASE {phase}")
            logger.info(f"{'='*60}")

            t0 = time.time()
            try:
                handler = self._get_phase_handler(phase)
                if handler is None:
                    logger.warning(f"Unknown phase: {phase}")
                    continue
                result = handler()
                elapsed = time.time() - t0
                self.results[f"phase_{phase}"] = {
                    "status": "success",
                    "elapsed_sec": round(elapsed, 1),
                    "result": result,
                }
                logger.info(f"Phase {phase} completed in {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - t0
                self.results[f"phase_{phase}"] = {
                    "status": "failed",
                    "elapsed_sec": round(elapsed, 1),
                    "error": str(e),
                }
                logger.error(f"Phase {phase} FAILED: {e}")
                logger.exception("Full traceback:")

        # Final report
        self._save_report()
        return self.results

    def _get_phase_handler(self, phase: str):
        handlers = {
            "0": self._phase_0_analysis,
            "0.5": self._phase_05_gold,
            "1": self._phase_1_pseudo_labels,
            "2": self._phase_2_models,
            "3": self._phase_3_training,
            "4": self._phase_4_ensemble,
            "5": self._phase_5_evaluation,
            "6": self._phase_6_visualization,
        }
        return handlers.get(phase)

    def _phase_0_analysis(self) -> dict:
        """Phase 0: GradCAM analysis on V1 backbones."""
        if self.dry_run:
            return {"dry_run": True, "would_run": "GradCAM analysis"}

        from V2_segmentation.analysis.gradcam_generator import GradCAMGenerator
        from V2_segmentation.config import BACKBONE_PROFILES, V1_CKPT_DIR
        from V2_segmentation.models.model_factory import create_v1_backbone
        import torch

        generated = []
        for bk_name in BACKBONE_PROFILES:
            try:
                model = create_v1_backbone(bk_name)
                # Try to load V1 checkpoint
                for suffix in ("_finetune_best.pth", "_final.pth", "_head_best.pth"):
                    ckpt = V1_CKPT_DIR / f"{bk_name}{suffix}"
                    if ckpt.exists():
                        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
                        if isinstance(state, dict) and "model_state_dict" in state:
                            model.load_state_dict(state["model_state_dict"], strict=False)
                        else:
                            model.load_state_dict(state, strict=False)
                        break
                gen = GradCAMGenerator(model, bk_name)
                generated.append(bk_name)
                gen.cleanup()
                del model
            except Exception as e:
                logger.warning(f"GradCAM failed for {bk_name}: {e}")

        return {"status": "GradCAM generation complete", "backbones": generated}

    def _phase_05_gold(self) -> dict:
        """Phase 0.5: Sample gold set + generate draft masks + calibrate."""
        if self.dry_run:
            return {"dry_run": True, "would_run": "gold sampling + draft masks + calibration"}

        # Step 1: Sample
        from V2_segmentation.scripts.sample_gold_set import sample_gold_set
        selected = sample_gold_set()
        total_sampled = sum(len(v) for v in selected.values())

        # Step 2: Generate draft masks
        from V2_segmentation.scripts.generate_draft_gold_masks import generate_draft_masks
        mask_counts = generate_draft_masks()

        # Step 3: Calibrate gate (only if gold masks exist)
        calibration = None
        has_masks = any(
            list((GOLD_LABELS_DIR / cls).glob("*_mask.*"))
            for cls in CLASS_NAMES
            if (GOLD_LABELS_DIR / cls).exists()
        )
        if has_masks:
            from V2_segmentation.validation.calibrate_gate import CalibrateGate
            calibrator = CalibrateGate()
            calibration = calibrator.calibrate()

        return {
            "sampled": total_sampled,
            "draft_masks": mask_counts,
            "calibration": calibration,
        }

    def _phase_1_pseudo_labels(self) -> dict:
        """Phase 1: Generate pseudo-labels for all training images."""
        if self.dry_run:
            return {"dry_run": True, "would_run": "pseudo-label generation pipeline"}

        from V2_segmentation.pseudo_labels import (
            GrabCutGenerator, GradCAMMaskGenerator,
            MaskCombiner, MaskQualityScorer, ClassSanityChecker,
        )

        train_dir = SPLIT_DIR / "train"
        pl_base = PSEUDO_LABELS_DIR

        # Step 1: GrabCut on all train images
        logger.info("Step 1/5: GrabCut masks...")
        grabcut = GrabCutGenerator()
        grabcut_out = pl_base / "grabcut"
        grabcut_out.mkdir(parents=True, exist_ok=True)
        grabcut_result = grabcut.generate_for_split(train_dir, grabcut_out)

        # Step 2: GradCAM ensemble
        logger.info("Step 2/5: GradCAM ensemble masks...")
        gradcam = GradCAMMaskGenerator()
        gradcam_out = pl_base / "gradcam"
        gradcam_out.mkdir(parents=True, exist_ok=True)
        gradcam_result = gradcam.generate_for_split(train_dir, gradcam_out)

        # Step 3: Combine
        logger.info("Step 3/5: Combining masks...")
        combiner = MaskCombiner()
        combined_out = pl_base / "combined"
        combined_out.mkdir(parents=True, exist_ok=True)
        combine_result = combiner.combine_for_split(
            "train", grabcut_out, gradcam_out, None, combined_out
        )

        # Step 4: Quality scoring
        logger.info("Step 4/5: Quality scoring...")
        scorer = MaskQualityScorer()
        scores = scorer.score_split(combined_out)

        # Step 5: Sanity check
        logger.info("Step 5/5: Class sanity check...")
        checker = ClassSanityChecker()
        tier_dir = pl_base / "tiers"
        tier_dir.mkdir(parents=True, exist_ok=True)
        _check_result = checker.check_split(combined_out, tier_dir)

        return {
            "grabcut_result": grabcut_result,
            "gradcam_result": gradcam_result,
            "combine_result": combine_result,
            "quality_scores": len(scores) if scores else 0,
        }

    def _phase_2_models(self) -> dict:
        """Phase 2: Model setup (verify, don't rebuild)."""
        if self.dry_run:
            return {"dry_run": True, "would_run": "model verification"}

        from V2_segmentation.models.model_factory import build_v2_model
        from V2_segmentation.config import BACKBONE_PROFILES

        results = {}
        for bk_name in BACKBONE_PROFILES:
            try:
                profile = BACKBONE_PROFILES[bk_name]
                model = build_v2_model(
                    bk_name,
                    decoder_channels=profile.get("decoder_channels", 256),
                    skip_channels=profile.get("skip_channels", 48),
                )
                n_params = sum(p.numel() for p in model.parameters())
                results[bk_name] = {
                    "status": "ok",
                    "params_M": round(n_params / 1e6, 2),
                }
                del model
            except Exception as e:
                results[bk_name] = {"status": "failed", "error": str(e)}

        return results

    def _phase_3_training(self) -> dict:
        """Phase 3: 3-phase training for all 15 backbones."""
        if self.dry_run:
            return {"dry_run": True, "would_run": "15 backbone training in waves"}

        from V2_segmentation.training.train_all_backbones import train_all_backbones
        report = train_all_backbones()
        return {"summary": report.summary()}

    def _phase_4_ensemble(self) -> dict:
        """Phase 4: 12-stage ensemble pipeline."""
        if self.dry_run:
            return {"dry_run": True, "would_run": "12-stage ensemble"}

        from V2_segmentation.ensemble_v2.ensemble_orchestrator import EnsembleOrchestratorV2
        orchestrator = EnsembleOrchestratorV2()
        results = orchestrator.run_all()
        return results

    def _phase_5_evaluation(self) -> dict:
        """Phase 5: Evaluation + validation gates."""
        if self.dry_run:
            return {"dry_run": True, "would_run": "leakage/overfit checks + audit"}

        from V2_segmentation.evaluation import (
            LeakageChecker, OverfitDetector, AuditReporter,
        )

        audit = AuditReporter()

        # Leakage check
        leakage = LeakageChecker()
        leakage_results = leakage.run_all_checks()
        audit.add_leakage_results(leakage_results)

        # Overfit check (if training history available)
        overfit = OverfitDetector()
        overfit_summary = overfit.get_summary()
        audit.add_overfit_results(overfit_summary)

        # Generate report
        report = audit.generate_report()
        audit.save()

        return report

    def _phase_6_visualization(self) -> dict:
        """Phase 6: Generate all visualization artifacts."""
        if self.dry_run:
            return {"dry_run": True, "would_run": "all visualization plots"}

        plots_generated = []

        # Tier distribution
        try:
            from V2_segmentation.visualization import TierDistribution
            tier_viz = TierDistribution()
            # Load tier data if available
            quality_file = PSEUDO_LABELS_DIR / "combined" / "quality_scores.csv"
            if quality_file.exists():
                import csv
                tiers: dict[str, int] = {"A": 0, "B": 0, "C": 0}
                with open(quality_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        t = row.get("tier", "C")
                        tiers[t] = tiers.get(t, 0) + 1
                path = tier_viz.plot_global_pie(tiers)
                plots_generated.append(str(path))
        except Exception as e:
            logger.warning(f"Tier visualization failed: {e}")

        # Training curves (from checkpoint metadata)
        try:
            from V2_segmentation.visualization import TrainingCurves
            curves = TrainingCurves()
            for json_file in sorted(CKPT_V2_DIR.glob("*_history.json")):
                path = curves.plot_from_json(json_file)
                plots_generated.append(str(path))
        except Exception as e:
            logger.warning(f"Training curves failed: {e}")

        # Per-backbone evaluation plots (confusion matrix, ROC, per-class)
        try:
            from V2_segmentation.visualization.backbone_plots import BackbonePlots
            bp = BackbonePlots()
            for npz_file in sorted(CKPT_V2_DIR.glob("*_eval.npz")):
                bk_name = npz_file.stem.replace("_eval", "")
                logger.info(f"  Generating eval plots for {bk_name}...")
                paths = bp.plot_from_saved_eval(bk_name, eval_dir=npz_file.parent)
                plots_generated.extend(str(p) for p in paths.values())
        except Exception as e:
            logger.warning(f"Backbone eval plots failed: {e}")

        # Ensemble stage plots (if ensemble stages saved predictions)
        try:
            from V2_segmentation.visualization.ensemble_stage_plots import EnsembleStagePlots
            esp = EnsembleStagePlots()
            ensemble_dir = CKPT_V2_DIR.parent / "ensemble_v2_results"
            if ensemble_dir.exists():
                for stage_npz in sorted(ensemble_dir.glob("stage_*_eval.npz")):
                    stage_name = stage_npz.stem.replace("_eval", "")
                    logger.info(f"  Generating ensemble plots for {stage_name}...")
                    paths = esp.plot_from_saved_eval(str(stage_npz), stage_name=stage_name)
                    plots_generated.extend(str(p) for p in paths.values())
        except Exception as e:
            logger.warning(f"Ensemble stage plots failed: {e}")

        return {
            "plots_generated": len(plots_generated),
            "paths": plots_generated,
        }

    def _save_report(self) -> None:
        elapsed = time.time() - self.start_time
        report = {
            "total_elapsed_sec": round(elapsed, 1),
            "phases": self.results,
        }

        output = BASE_DIR / "pipeline_v2_report.json"
        with open(output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Pipeline report saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2 Segmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline_v2.py                    # All phases
  python run_pipeline_v2.py --phase 1          # Pseudo-labels only
  python run_pipeline_v2.py --phase 3 4        # Training + ensemble
  python run_pipeline_v2.py --phase 0.5 1 3    # Gold + pseudo + training
  python run_pipeline_v2.py --dry-run           # Preview mode
        """,
    )
    parser.add_argument(
        "--phase", nargs="+", default=None,
        help="Phase(s) to run. Options: 0, 0.5, 1, 2, 3, 4, 5, 6",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would run without executing",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    pipeline = PipelineV2(dry_run=args.dry_run)
    results = pipeline.run(args.phase)

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    for phase_key, info in results.items():
        status = info.get("status", "unknown")
        elapsed = info.get("elapsed_sec", 0)
        emoji = "✓" if status == "success" else "✗"
        print(f"  {emoji} {phase_key}: {status} ({elapsed}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
