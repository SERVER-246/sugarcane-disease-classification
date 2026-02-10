"""
V2 Evaluation Isolation Module
==============================
Automated leakage detection, overfit monitoring, OOF generation,
and per-stage audit reporting.

Per Section 8 of the sprint plan: with 12 ensemble stages, 3 training phases,
pseudo-label refinement, cascaded training, and adversarial boosting, there
are multiple vectors for data leakage and overfitting.
"""

from .leakage_checker import LeakageChecker
from .overfit_detector import OverfitDetector
from .oof_generator import OOFGenerator
from .audit_reporter import AuditReporter
