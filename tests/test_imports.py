# ============================================================================
# DISEASE Classification Pipeline - Import Validation Tests
# ============================================================================
# Created: Sprint 2 - CI/CD Without Behavior Change
# Purpose: Verify all critical modules import successfully
# ============================================================================

import pytest
import sys
from pathlib import Path


class TestCoreImports:
    """Test that core modules import without errors."""

    def test_import_torch(self):
        """Test PyTorch import."""
        import torch
        assert hasattr(torch, "__version__")

    def test_import_torchvision(self):
        """Test TorchVision import."""
        import torchvision
        assert hasattr(torchvision, "__version__")

    def test_import_numpy(self):
        """Test NumPy import."""
        import numpy as np
        assert hasattr(np, "__version__")

    def test_import_pandas(self):
        """Test Pandas import."""
        import pandas as pd
        assert hasattr(pd, "__version__")

    def test_import_sklearn(self):
        """Test scikit-learn import."""
        import sklearn
        assert hasattr(sklearn, "__version__")

    def test_import_pillow(self):
        """Test Pillow import."""
        from PIL import Image
        assert hasattr(Image, "open")


class TestProjectImports:
    """Test that project modules import without errors."""

    def test_import_base_backbones(self):
        """Test Base_backbones module import."""
        from Base_backbones import (
            create_custom_backbone,
            BACKBONES,
            NUM_CLASSES,
            IMG_SIZE,
        )
        assert len(BACKBONES) == 15
        assert callable(create_custom_backbone)
        # NUM_CLASSES may be None (set dynamically from dataset)
        assert NUM_CLASSES is None or isinstance(NUM_CLASSES, int)
        assert isinstance(IMG_SIZE, int)  # IMG_SIZE is always set

    def test_import_backbones_list(self):
        """Test that BACKBONES list is correct."""
        from Base_backbones import BACKBONES
        
        expected_backbones = [
            "CustomConvNeXt",
            "CustomEfficientNetV4",
            "CustomGhostNetV2",
            "CustomResNetMish",
            "CustomCSPDarkNet",
            "CustomInceptionV4",
            "CustomViTHybrid",
            "CustomSwinTransformer",
            "CustomCoAtNet",
            "CustomRegNet",
            "CustomDenseNetHybrid",
            "CustomDeiTStyle",
            "CustomMaxViT",
            "CustomMobileOne",
            "CustomDynamicConvNet",
        ]
        
        assert BACKBONES == expected_backbones

    def test_import_image_validator(self):
        """Test image_validator module import."""
        import image_validator
        assert hasattr(image_validator, "ImageValidator")

    def test_import_disease_classifier_gui(self):
        """Test disease_classifier_gui module import (skip if no display)."""
        try:
            import disease_classifier_gui
            # Verify module has expected attributes
            assert disease_classifier_gui is not None
        except Exception as e:
            # GUI imports may fail in headless environments
            if "display" in str(e).lower() or "tk" in str(e).lower():
                pytest.skip("GUI not available in headless environment")
            raise


class TestExportImports:
    """Test that export-related modules import."""

    def test_import_onnx(self):
        """Test ONNX import."""
        import onnx
        assert hasattr(onnx, "__version__")

    def test_import_onnxruntime(self):
        """Test ONNX Runtime import."""
        import onnxruntime as ort
        assert hasattr(ort, "__version__")


class TestOptionalImports:
    """Test optional dependencies (may skip if not installed)."""

    def test_import_timm(self):
        """Test timm (PyTorch Image Models) import."""
        try:
            import timm
            assert hasattr(timm, "__version__")
        except ImportError:
            pytest.skip("timm not installed")

    def test_import_xgboost(self):
        """Test XGBoost import."""
        try:
            import xgboost
            assert hasattr(xgboost, "__version__")
        except ImportError:
            pytest.skip("xgboost not installed")


class TestBaseBackModuleImports:
    """Test BASE-BACK module imports if available."""

    def test_baseback_structure_exists(self, project_root):
        """Test that BASE-BACK directory structure exists."""
        baseback_path = project_root / "BASE-BACK" / "src"
        assert baseback_path.exists(), "BASE-BACK/src directory should exist"

    def test_import_baseback_config(self):
        """Test BASE-BACK config import."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "BASE-BACK" / "src"))
            from config.settings import BACKBONES, NUM_CLASSES
            assert len(BACKBONES) == 15
            assert isinstance(NUM_CLASSES, int)
        except ImportError:
            pytest.skip("BASE-BACK config not importable")

    def test_import_baseback_utils(self):
        """Test BASE-BACK utils import."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "BASE-BACK" / "src"))
            from utils import logger
            assert hasattr(logger, "info")
        except ImportError:
            pytest.skip("BASE-BACK utils not importable")
