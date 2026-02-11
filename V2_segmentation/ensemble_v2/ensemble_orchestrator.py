"""
V2_segmentation/ensemble_v2/ensemble_orchestrator.py
=====================================================
Full 12-stage ensemble pipeline orchestrator.

Runs stages 1–12 in sequence with per-stage leakage checks,
metric tracking, and comprehensive reporting.
"""

from __future__ import annotations

import gc
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from V2_segmentation.config import ENSEMBLE_V2_DIR

logger = logging.getLogger(__name__)


class EnsembleOrchestratorV2:
    """Orchestrate the full 12-stage V2 ensemble pipeline.

    Manages dependencies between stages, runs leakage checks
    after each stage, and produces a comprehensive report.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        skip_stages: list[int] | None = None,
    ) -> None:
        self.output_dir = output_dir or ENSEMBLE_V2_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_stages = set(skip_stages or [])
        self.results: dict[str, Any] = {}
        self.start_time = datetime.now()

    def run_all(self) -> dict[str, Any]:
        """Run all 12 stages sequentially.

        Returns comprehensive results dict.
        """
        from V2_segmentation.evaluation.leakage_checker import LeakageChecker
        from V2_segmentation.evaluation.overfit_detector import OverfitDetector

        leakage = LeakageChecker()
        _overfit = OverfitDetector()

        # Pre-flight leakage check
        logger.info("=" * 60)
        logger.info("V2 ENSEMBLE PIPELINE — 12 STAGES")
        logger.info("=" * 60)
        pre_check = leakage.run_all_checks()
        self.results["pre_flight_leakage"] = pre_check
        if pre_check.get("overall_status") == "FAIL":
            logger.error("Pre-flight leakage check FAILED — aborting ensemble")
            return self.results

        # ── Stage 1: Individual V2 Predictions ────────────────────────
        if 1 not in self.skip_stages:
            self._run_stage(1, self._stage1)

        # ── Stage 2-7: V1 Stages Re-run ──────────────────────────────
        if not self.skip_stages.intersection({2, 3, 4, 5, 6, 7}):
            self._run_stage("2-7", self._stages2to7)

        # ── Stage 8: Segmentation-Informed ────────────────────────────
        if 8 not in self.skip_stages:
            self._run_stage(8, self._stage8)

        # ── Stage 9: Cascaded Sequential ──────────────────────────────
        if 9 not in self.skip_stages:
            self._run_stage(9, self._stage9)

        # ── Stage 10: Adversarial Boosting ────────────────────────────
        if 10 not in self.skip_stages:
            self._run_stage(10, self._stage10)

        # ── Stage 11: Cross-Architecture Referee ──────────────────────
        if 11 not in self.skip_stages:
            self._run_stage(11, self._stage11)

        # ── Stage 12: Multi-Teacher Distillation ──────────────────────
        if 12 not in self.skip_stages:
            self._run_stage(12, self._stage12)

        # Final report
        self.results["duration_seconds"] = (
            datetime.now() - self.start_time
        ).total_seconds()
        self._save_report()

        return self.results

    def _run_stage(self, stage_id: int | str, fn: Any) -> None:
        """Run a stage with error handling, timing, and eval artifact saving."""
        stage_name = f"stage_{stage_id}"
        logger.info(f"\n{'─'*40}\n  STAGE {stage_id}\n{'─'*40}")

        # GPU cleanup BEFORE stage to ensure max headroom
        self._gpu_cleanup()

        start = datetime.now()

        try:
            result = fn()
            result["duration_seconds"] = (datetime.now() - start).total_seconds()
            self.results[stage_name] = result
            logger.info(f"Stage {stage_id} completed in {result['duration_seconds']:.1f}s")

            # Save eval NPZ for downstream plot generation (Phase 6)
            self._save_stage_eval_artifact(stage_name, result)

        except Exception as e:
            logger.error(f"Stage {stage_id} FAILED: {e}")
            self.results[stage_name] = {
                "status": "FAIL",
                "error": str(e),
                "duration_seconds": (datetime.now() - start).total_seconds(),
            }
        finally:
            # GPU cleanup AFTER stage to free memory for next stage
            self._gpu_cleanup()

    def _gpu_cleanup(self) -> None:
        """Full GPU cleanup between stages to prevent OOM."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _save_stage_eval_artifact(
        self, stage_name: str, result: dict[str, Any],
    ) -> None:
        """Save stage predictions as NPZ for Phase 6 eval plots."""
        try:
            all_labels = result.get("all_labels")
            all_probs = result.get("all_probs")
            all_preds = result.get("all_preds")

            # Need at least labels + (probs or preds) to produce plots
            if all_labels is None:
                return

            eval_dir = self.output_dir / "ensemble_v2_results"
            eval_dir.mkdir(parents=True, exist_ok=True)

            save_dict: dict[str, Any] = {"all_labels": np.asarray(all_labels)}
            if all_probs is not None:
                save_dict["all_probs"] = np.asarray(all_probs)
            if all_preds is not None:
                save_dict["all_preds"] = np.asarray(all_preds)

            npz_path = eval_dir / f"{stage_name}_eval.npz"
            np.savez_compressed(str(npz_path), **save_dict)
            logger.info(f"  Saved stage eval: {npz_path.name}")

        except Exception as e:
            logger.warning(f"  Could not save eval for {stage_name}: {e}")

    def _stage1(self) -> dict[str, Any]:
        from .stage1_individual_v2 import Stage1IndividualV2
        return Stage1IndividualV2().run()

    def _stages2to7(self) -> dict[str, Any]:
        from .stage2_to_7_rerun import Stage2To7Rerun
        return Stage2To7Rerun().run_all()

    def _stage8(self) -> dict[str, Any]:
        from .stage8_seg_informed import Stage8SegInformed
        return Stage8SegInformed().run()

    def _stage9(self) -> dict[str, Any]:
        from .stage9_cascaded import Stage9Cascaded
        return Stage9Cascaded().run()

    def _stage10(self) -> dict[str, Any]:
        from .stage10_adversarial import Stage10Adversarial
        return Stage10Adversarial().run()

    def _stage11(self) -> dict[str, Any]:
        from .stage11_referee import Stage11Referee
        return Stage11Referee().run()

    def _stage12(self) -> dict[str, Any]:
        from .stage12_distillation_v2 import Stage12DistillationV2
        return Stage12DistillationV2().run()

    def _save_report(self) -> None:
        """Save ensemble pipeline report."""
        report_path = self.output_dir / "ensemble_v2_report.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Ensemble report saved to {report_path}")

        # Summary
        stages_run = sum(1 for k in self.results if k.startswith("stage_"))
        stages_pass = sum(
            1 for k, v in self.results.items()
            if k.startswith("stage_") and isinstance(v, dict) and v.get("status") != "FAIL"
        )
        logger.info(
            f"\n{'='*60}\n"
            f"ENSEMBLE PIPELINE COMPLETE: {stages_pass}/{stages_run} stages passed\n"
            f"Duration: {self.results.get('duration_seconds', 0):.1f}s\n"
            f"{'='*60}"
        )
