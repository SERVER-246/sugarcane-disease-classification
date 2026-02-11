"""
V2_segmentation/evaluation/audit_reporter.py
=============================================
Comprehensive audit report aggregating all evaluation checks.

Combines results from:
  - LeakageChecker (train-test overlap, test virginity, OOF integrity)
  - OverfitDetector (train-val gaps, per-class, fold variance)
  - Per-stage metric tracking

Produces a single JSON report + human-readable summary.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from V2_segmentation.config import EVAL_DIR

logger = logging.getLogger(__name__)


class AuditReporter:
    """Aggregate and report all evaluation audit results.

    Collects results from multiple checkers and produces a unified
    report with clear PASS/FAIL verdicts.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or EVAL_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sections: dict[str, Any] = {}
        self.start_time = datetime.now()

    def add_section(self, name: str, results: dict[str, Any]) -> None:
        """Add an audit section."""
        self.sections[name] = {
            "timestamp": datetime.now().isoformat(),
            **results,
        }

    def add_leakage_results(self, results: dict[str, Any]) -> None:
        """Add leakage checker results."""
        self.add_section("leakage_detection", results)

    def add_overfit_results(self, results: dict[str, Any]) -> None:
        """Add overfitting detection results."""
        self.add_section("overfit_detection", results)

    def add_stage_results(
        self,
        stage_name: str,
        metrics: dict[str, float],
        checks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add per-stage evaluation results."""
        if "stages" not in self.sections:
            self.sections["stages"] = {}
        self.sections["stages"][stage_name] = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "checks": checks or [],
        }

    def generate_report(self) -> dict[str, Any]:
        """Generate the full audit report."""
        # Determine overall status
        all_statuses = []
        for section in self.sections.values():
            if isinstance(section, dict):
                s = section.get("status") or section.get("overall_status")
                if s:
                    all_statuses.append(s)

        if "HALT" in all_statuses or "FAIL" in all_statuses:
            overall = "FAIL"
        elif "WARNING" in all_statuses:
            overall = "WARNING"
        else:
            overall = "PASS"

        report = {
            "report_version": "2.0",
            "generated_at": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "overall_status": overall,
            "sections": self.sections,
            "summary": self._generate_summary(overall),
        }

        return report

    def _generate_summary(self, overall: str) -> dict[str, Any]:
        """Generate human-readable summary."""
        issues: list[str] = []
        passes: list[str] = []

        for name, section in self.sections.items():
            if isinstance(section, dict):
                status = section.get("status") or section.get("overall_status", "?")
                if status in ("FAIL", "HALT"):
                    issues.append(f"❌ {name}: {status}")
                elif status == "WARNING":
                    issues.append(f"⚠️ {name}: {status}")
                elif status == "PASS":
                    passes.append(f"✅ {name}")

        return {
            "overall": overall,
            "issues": issues,
            "passes": passes,
            "n_sections": len(self.sections),
        }

    def save(self, filename: str = "audit_report.json") -> Path:
        """Save the full audit report."""
        report = self.generate_report()
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Audit report saved to {path}")

        # Also save human-readable text version
        txt_path = self.output_dir / filename.replace(".json", ".txt")
        self._save_text(report, txt_path)

        return path

    def _save_text(self, report: dict[str, Any], path: Path) -> None:
        """Save a human-readable text summary."""
        lines = [
            "=" * 60,
            "V2 SEGMENTATION PIPELINE — EVALUATION AUDIT REPORT",
            "=" * 60,
            f"Generated: {report['generated_at']}",
            f"Duration: {report['duration_seconds']:.1f}s",
            f"Overall Status: {report['overall_status']}",
            "",
        ]

        summary = report.get("summary", {})
        if summary.get("passes"):
            lines.append("PASSED CHECKS:")
            for p in summary["passes"]:
                lines.append(f"  {p}")
            lines.append("")

        if summary.get("issues"):
            lines.append("ISSUES:")
            for i in summary["issues"]:
                lines.append(f"  {i}")
            lines.append("")

        # Per-stage metrics
        stages = self.sections.get("stages", {})
        if stages:
            lines.append("-" * 40)
            lines.append("PER-STAGE METRICS:")
            for stage_name, data in stages.items():
                metrics = data.get("metrics", {})
                lines.append(f"  {stage_name}:")
                for k, v in metrics.items():
                    lines.append(f"    {k}: {v}")
            lines.append("")

        lines.append("=" * 60)
        # Use UTF-8 encoding to support Unicode characters like ✅ ✗
        path.write_text("\n".join(lines), encoding="utf-8")
