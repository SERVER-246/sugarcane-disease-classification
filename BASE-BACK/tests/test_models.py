"""Unit tests for models module."""

import sys
import unittest
from pathlib import Path

import torch


# Add both root and src to path for proper imports
root_dir = Path(__file__).parent.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(src_dir))

# These imports must come after path modification - noqa required
from src.config.settings import BACKBONES  # noqa: E402
from src.models.architectures import BACKBONE_MAP  # noqa: E402
from src.models.architectures import (  # noqa: E402
    create_custom_backbone as create_custom_backbone_safe,
)


# Create settings object for compatibility
class settings:
    """Settings container for test compatibility."""

    BACKBONES = BACKBONES


class TestModelCreation(unittest.TestCase):
    """Test backbone model creation."""

    def test_all_backbones_creatable(self):
        """Test that all 15 backbones can be created."""
        for backbone_name in settings.BACKBONES:
            with self.subTest(backbone=backbone_name):
                model = create_custom_backbone_safe(backbone_name, num_classes=13)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, torch.nn.Module)

    def test_backbone_forward_pass(self):
        """Test forward pass on all backbones."""
        dummy_input = torch.randn(2, 3, 224, 224)

        for backbone_name in settings.BACKBONES[:3]:  # Test subset for speed
            with self.subTest(backbone=backbone_name):
                model = create_custom_backbone_safe(backbone_name, num_classes=13)
                model.eval()

                with torch.no_grad():
                    output = model(dummy_input)

                self.assertEqual(output.shape, (2, 13))
                self.assertFalse(torch.isnan(output).any())

    def test_backbone_parameter_count(self):
        """Test that backbones have reasonable parameter counts."""
        model = create_custom_backbone_safe(settings.BACKBONES[0], num_classes=13)
        param_count = sum(p.numel() for p in model.parameters())

        # Should have at least 1M parameters
        self.assertGreater(param_count, 1_000_000)

    def test_backbone_registry(self):
        """Test backbone registry."""
        self.assertEqual(len(BACKBONE_MAP), 15)

        for backbone_name in settings.BACKBONES:
            self.assertIn(backbone_name, BACKBONE_MAP)

    def test_invalid_backbone_name(self):
        """Test error handling for invalid backbone names."""
        model = create_custom_backbone_safe("InvalidBackbone", num_classes=13)
        self.assertIsNone(model)  # Should return None for invalid names


class TestModelGradients(unittest.TestCase):
    """Test gradient flow in models."""

    def test_backward_pass(self):
        """Test that gradients flow correctly."""
        model = create_custom_backbone_safe("CustomConvNeXt", num_classes=13)
        model.train()

        dummy_input = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")


if __name__ == "__main__":
    unittest.main()
