"""
V2_segmentation/evaluation/leakage_checker.py
==============================================
Automated data leakage detection across all pipeline stages.

Per Section 8.3:
  1. TRAIN-TEST OVERLAP CHECK — hash file paths, assert intersection = ∅
  2. TEST SET VIRGINITY CHECK — SHA-256 of test file list vs V1 hash
  3. INFORMATION FLOW AUDIT — verify OOF used for stacking, not direct preds
  4. PER-STAGE METRIC MONOTONICITY — warn if stage N+1 worse than N by >2%

Runs after each stage; results logged to evaluation/leakage_audit.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from V2_segmentation.config import (
    EVAL_DIR, SPLIT_DIR,
)

logger = logging.getLogger(__name__)


def _hash_file_list(directory: Path) -> str:
    """Compute SHA-256 hash of sorted file list under a directory."""
    files = sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file())
    content = "\n".join(files).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _get_image_stems(directory: Path) -> set[str]:
    """Get set of image stems (filename without extension) from a directory."""
    stems = set()
    for class_dir in directory.iterdir():
        if not class_dir.is_dir():
            continue
        for img in class_dir.iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                stems.add(f"{class_dir.name}/{img.stem}")
    return stems


class LeakageChecker:
    """Automated leakage detection engine.

    Tracks data flow across stages and verifies isolation guarantees.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or EVAL_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log: list[dict[str, Any]] = []
        self._stage_metrics: dict[str, float] = {}

    def check_train_test_overlap(self) -> dict[str, Any]:
        """Check that train and test sets have zero image overlap.

        Returns dict with status, overlap count, and overlapping files.
        """
        train_dir = SPLIT_DIR / "train"
        test_dir = SPLIT_DIR / "test"
        val_dir = SPLIT_DIR / "val"

        if not train_dir.exists() or not test_dir.exists():
            return {"status": "SKIP", "reason": "Split directories not found"}

        train_stems = _get_image_stems(train_dir)
        test_stems = _get_image_stems(test_dir)
        val_stems = _get_image_stems(val_dir) if val_dir.exists() else set()

        train_test_overlap = train_stems & test_stems
        train_val_overlap = train_stems & val_stems
        val_test_overlap = val_stems & test_stems

        total_overlap = len(train_test_overlap) + len(train_val_overlap) + len(val_test_overlap)

        result = {
            "status": "PASS" if total_overlap == 0 else "FAIL",
            "train_test_overlap": len(train_test_overlap),
            "train_val_overlap": len(train_val_overlap),
            "val_test_overlap": len(val_test_overlap),
            "overlapping_files": {
                "train_test": sorted(train_test_overlap)[:20],
                "train_val": sorted(train_val_overlap)[:20],
                "val_test": sorted(val_test_overlap)[:20],
            },
            "set_sizes": {
                "train": len(train_stems),
                "val": len(val_stems),
                "test": len(test_stems),
            },
        }

        self._log_check("train_test_overlap", result)
        return result

    def check_test_set_virginity(
        self,
        v1_hash: str | None = None,
    ) -> dict[str, Any]:
        """Verify test set hasn't changed since V1.

        Parameters
        ----------
        v1_hash : str or None
            SHA-256 hash from V1. If None, computes and stores for future use.
        """
        test_dir = SPLIT_DIR / "test"
        if not test_dir.exists():
            return {"status": "SKIP", "reason": "Test directory not found"}

        current_hash = _hash_file_list(test_dir)

        hash_file = self.output_dir / "test_set_hash.json"

        if v1_hash is None:
            # Try to load stored V1 hash
            if hash_file.exists():
                with open(hash_file) as f:
                    stored = json.load(f)
                v1_hash = stored.get("v1_hash")

        if v1_hash is None:
            # First run: store current hash as V1 baseline
            with open(hash_file, "w") as f:
                json.dump({"v1_hash": current_hash, "note": "auto-created"}, f, indent=2)
            result = {
                "status": "BASELINE_SET",
                "hash": current_hash,
                "message": "V1 baseline hash stored. Future checks will compare against it.",
            }
        else:
            match = current_hash == v1_hash
            result = {
                "status": "PASS" if match else "FAIL",
                "v1_hash": v1_hash,
                "current_hash": current_hash,
                "match": match,
            }
            if not match:
                logger.error("TEST SET CHANGED SINCE V1! Possible data leakage.")

        self._log_check("test_set_virginity", result)
        return result

    def check_oof_integrity(
        self,
        oof_predictions_path: Path,
        train_size: int,
        n_folds: int = 5,
    ) -> dict[str, Any]:
        """Verify OOF predictions have correct structure.

        Checks:
          - OOF array has exactly train_size rows (each sample predicted once)
          - No NaN/Inf values
          - Predictions are valid probabilities
        """
        import numpy as np

        if not oof_predictions_path.exists():
            return {"status": "SKIP", "reason": f"OOF file not found: {oof_predictions_path}"}

        oof = np.load(str(oof_predictions_path))

        checks: list[dict[str, Any]] = []

        # Check shape
        if oof.shape[0] != train_size:
            checks.append({
                "check": "row_count",
                "status": "FAIL",
                "expected": train_size,
                "actual": oof.shape[0],
            })
        else:
            checks.append({"check": "row_count", "status": "PASS"})

        # Check NaN/Inf
        has_nan = bool(np.isnan(oof).any())
        has_inf = bool(np.isinf(oof).any())
        checks.append({
            "check": "nan_inf",
            "status": "FAIL" if (has_nan or has_inf) else "PASS",
            "has_nan": has_nan,
            "has_inf": has_inf,
        })

        # Check probability range
        in_range = bool((oof >= 0).all() and (oof <= 1).all())
        checks.append({
            "check": "probability_range",
            "status": "PASS" if in_range else "FAIL",
            "min": float(oof.min()),
            "max": float(oof.max()),
        })

        all_pass = all(c["status"] == "PASS" for c in checks)
        result = {"status": "PASS" if all_pass else "FAIL", "checks": checks}
        self._log_check("oof_integrity", result)
        return result

    def check_metric_monotonicity(
        self,
        stage_name: str,
        val_accuracy: float,
        tolerance: float = 0.02,
    ) -> dict[str, Any]:
        """Check that new stage doesn't degrade val accuracy beyond tolerance.

        Parameters
        ----------
        stage_name : str
            e.g. "stage_8_seg_informed"
        val_accuracy : float
            Validation accuracy for this stage.
        tolerance : float
            Maximum acceptable accuracy drop (default 2%).
        """
        self._stage_metrics[stage_name] = val_accuracy

        # Find previous stage metric
        stage_keys = list(self._stage_metrics.keys())
        if len(stage_keys) < 2:
            result = {"status": "PASS", "message": "First stage, no comparison"}
            self._log_check("metric_monotonicity", result)
            return result

        prev_stage = stage_keys[-2]
        prev_acc = self._stage_metrics[prev_stage]
        drop = prev_acc - val_accuracy

        if drop > tolerance:
            status = "FAIL" if drop > 2 * tolerance else "WARNING"
        else:
            status = "PASS"

        result = {
            "status": status,
            "stage": stage_name,
            "val_accuracy": val_accuracy,
            "prev_stage": prev_stage,
            "prev_accuracy": prev_acc,
            "drop": round(drop, 4),
            "tolerance": tolerance,
        }

        if status == "WARNING":
            logger.warning(
                f"Stage {stage_name} val_acc={val_accuracy:.4f} is "
                f"{drop:.4f} below {prev_stage} ({prev_acc:.4f})"
            )
        elif status == "FAIL":
            logger.error(
                f"HALT: Stage {stage_name} degraded by {drop:.4f} "
                f"(>{2*tolerance:.4f}) from {prev_stage}"
            )

        self._log_check("metric_monotonicity", result)
        return result

    def run_all_checks(self) -> dict[str, Any]:
        """Run all leakage checks and return combined report."""
        results = {
            "train_test_overlap": self.check_train_test_overlap(),
            "test_set_virginity": self.check_test_set_virginity(),
        }

        overall = all(
            r.get("status") in ("PASS", "SKIP", "BASELINE_SET")
            for r in results.values()
        )
        results["overall_status"] = {"value": "PASS" if overall else "FAIL"}

        # Save full audit
        self._save_audit()
        return results

    def _log_check(self, name: str, result: dict[str, Any]) -> None:
        """Add check result to audit log."""
        import datetime
        self.audit_log.append({
            "check": name,
            "timestamp": datetime.datetime.now().isoformat(),
            **result,
        })

    def _save_audit(self) -> None:
        """Save audit log to JSON."""
        audit_path = self.output_dir / "leakage_audit.json"
        with open(audit_path, "w") as f:
            json.dump(self.audit_log, f, indent=2)
        logger.info(f"Leakage audit saved to {audit_path}")
