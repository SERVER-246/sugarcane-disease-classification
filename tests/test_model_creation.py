# ============================================================================
# DISEASE Classification Pipeline - Model Creation Tests
# ============================================================================
# Created: Sprint 2 - CI/CD Without Behavior Change
# Purpose: Verify all 15 backbone models can be instantiated
# ============================================================================

import pytest
import torch


class TestModelCreation:
    """Test that all backbone models can be created."""

    def test_create_custom_convnext(self, num_classes, device):
        """Test CustomConvNeXt creation."""
        from Base_backbones import create_custom_backbone

        model = create_custom_backbone("CustomConvNeXt", num_classes)
        assert model is not None
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_create_custom_efficientnetv4(self, num_classes, device):
        """Test CustomEfficientNetV4 creation."""
        from Base_backbones import create_custom_backbone

        model = create_custom_backbone("CustomEfficientNetV4", num_classes)
        assert model is not None
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_create_custom_resnetmish(self, num_classes, device):
        """Test CustomResNetMish creation."""
        from Base_backbones import create_custom_backbone

        model = create_custom_backbone("CustomResNetMish", num_classes)
        assert model is not None
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_create_custom_coatnet(self, num_classes, device):
        """Test CustomCoAtNet creation."""
        from Base_backbones import create_custom_backbone

        model = create_custom_backbone("CustomCoAtNet", num_classes)
        assert model is not None
        assert sum(p.numel() for p in model.parameters()) > 0

    @pytest.mark.slow
    @pytest.mark.parametrize("backbone_name", [
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
    ])
    def test_create_all_backbones(self, backbone_name, num_classes):
        """Test creation of all 15 backbone architectures."""
        from Base_backbones import create_custom_backbone

        model = create_custom_backbone(backbone_name, num_classes)

        assert model is not None, f"Failed to create {backbone_name}"

        # Check model has parameters
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0, f"{backbone_name} has no parameters"

        # Verify it's a PyTorch module
        assert isinstance(model, torch.nn.Module)


class TestModelProperties:
    """Test model properties and configuration."""

    def test_model_has_trainable_params(self, sample_backbone_names, num_classes):
        """Test that models have trainable parameters."""
        from Base_backbones import create_custom_backbone

        for name in sample_backbone_names:
            model = create_custom_backbone(name, num_classes)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert trainable > 0, f"{name} has no trainable parameters"

    def test_model_can_move_to_device(self, sample_backbone_names, num_classes, device):
        """Test that models can be moved to device."""
        from Base_backbones import create_custom_backbone

        for name in sample_backbone_names:
            model = create_custom_backbone(name, num_classes)
            model = model.to(device)

            # Verify at least one parameter is on the correct device
            param = next(model.parameters())
            assert param.device.type == device.type

    def test_model_train_eval_modes(self, sample_backbone_names, num_classes):
        """Test that models can switch between train and eval modes."""
        from Base_backbones import create_custom_backbone

        for name in sample_backbone_names:
            model = create_custom_backbone(name, num_classes)

            # Test train mode
            model.train()
            assert model.training is True

            # Test eval mode
            model.eval()
            assert model.training is False


class TestBackbonesList:
    """Test BACKBONES list integrity."""

    def test_backbones_count(self, backbone_names):
        """Test that exactly 15 backbones are defined."""
        assert len(backbone_names) == 15

    def test_backbones_unique(self, backbone_names):
        """Test that all backbone names are unique."""
        assert len(backbone_names) == len(set(backbone_names))

    def test_backbones_naming_convention(self, backbone_names):
        """Test that all backbone names start with 'Custom'."""
        for name in backbone_names:
            assert name.startswith("Custom"), f"{name} doesn't start with 'Custom'"
