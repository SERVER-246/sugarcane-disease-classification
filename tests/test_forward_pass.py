# ============================================================================
# DISEASE Classification Pipeline - Forward Pass Tests
# ============================================================================
# Created: Sprint 2 - CI/CD Without Behavior Change
# Purpose: Verify forward pass works correctly for all backbones
# ============================================================================

import pytest
import torch


class TestForwardPass:
    """Test forward pass for backbone models."""

    def test_forward_pass_convnext(self, num_classes, img_size, batch_size, device):
        """Test CustomConvNeXt forward pass."""
        from Base_backbones import create_custom_backbone

        model = create_custom_backbone("CustomConvNeXt", num_classes)
        model = model.to(device)
        model.eval()

        x = torch.randn(batch_size, 3, img_size, img_size, device=device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_pass_resnetmish(self, num_classes, img_size, batch_size, device):
        """Test CustomResNetMish forward pass."""
        from Base_backbones import create_custom_backbone

        model = create_custom_backbone("CustomResNetMish", num_classes)
        model = model.to(device)
        model.eval()

        x = torch.randn(batch_size, 3, img_size, img_size, device=device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any()

    def test_forward_pass_coatnet(self, num_classes, img_size, batch_size, device):
        """Test CustomCoAtNet forward pass."""
        from Base_backbones import create_custom_backbone

        model = create_custom_backbone("CustomCoAtNet", num_classes)
        model = model.to(device)
        model.eval()

        x = torch.randn(batch_size, 3, img_size, img_size, device=device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any()

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
    def test_forward_pass_all_backbones(
        self, backbone_name, num_classes, img_size, batch_size, device
    ):
        """Test forward pass for all 15 backbone architectures."""
        from Base_backbones import create_custom_backbone

        model = create_custom_backbone(backbone_name, num_classes)
        model = model.to(device)
        model.eval()

        x = torch.randn(batch_size, 3, img_size, img_size, device=device)

        with torch.no_grad():
            output = model(x)

        # Verify output shape
        assert output.shape == (batch_size, num_classes), \
            f"{backbone_name} output shape mismatch: {output.shape}"

        # Verify no NaN/Inf values
        assert not torch.isnan(output).any(), \
            f"{backbone_name} produced NaN values"
        assert not torch.isinf(output).any(), \
            f"{backbone_name} produced Inf values"


class TestBackwardPass:
    """Test backward pass (gradient computation) for models."""

    def test_backward_pass_convnext(self, num_classes, img_size, batch_size, device):
        """Test CustomConvNeXt backward pass."""
        from Base_backbones import create_custom_backbone

        model = create_custom_backbone("CustomConvNeXt", num_classes)
        model = model.to(device)
        model.train()

        x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        target = torch.randint(0, num_classes, (batch_size,), device=device)

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Verify gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                assert not torch.isnan(param.grad).any()
                break

        assert has_grad, "No gradients computed"

    def test_backward_pass_sample_backbones(
        self, sample_backbone_names, num_classes, img_size, batch_size, device
    ):
        """Test backward pass for sample backbones."""
        from Base_backbones import create_custom_backbone

        for name in sample_backbone_names:
            model = create_custom_backbone(name, num_classes)
            model = model.to(device)
            model.train()

            x = torch.randn(batch_size, 3, img_size, img_size, device=device)
            target = torch.randint(0, num_classes, (batch_size,), device=device)

            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()

            # Verify at least one parameter has gradient
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters()
            )
            assert has_grad, f"{name} failed to compute gradients"


class TestOutputProperties:
    """Test output properties and consistency."""

    def test_output_is_logits(self, sample_backbone_names, num_classes, img_size, device):
        """Test that output is logits (not probabilities)."""
        from Base_backbones import create_custom_backbone

        for name in sample_backbone_names:
            model = create_custom_backbone(name, num_classes)
            model = model.to(device)
            model.eval()

            x = torch.randn(1, 3, img_size, img_size, device=device)

            with torch.no_grad():
                output = model(x)

            # Logits can be any value, probabilities sum to 1
            # Check that values are not all between 0 and 1 (likely logits)
            assert output.min() < 0 or output.max() > 1, \
                f"{name} output looks like probabilities, expected logits"

    def test_output_deterministic_in_eval(
        self, sample_backbone_names, num_classes, img_size, device
    ):
        """Test that eval mode produces deterministic outputs."""
        from Base_backbones import create_custom_backbone

        for name in sample_backbone_names:
            model = create_custom_backbone(name, num_classes)
            model = model.to(device)
            model.eval()

            x = torch.randn(1, 3, img_size, img_size, device=device)

            with torch.no_grad():
                output1 = model(x)
                output2 = model(x)

            assert torch.allclose(output1, output2), \
                f"{name} produces non-deterministic outputs in eval mode"
