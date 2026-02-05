"""
Unit test suite for backbone models and pipeline components.
Extracted from original Base_backbones.py unit testing functionality.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn


# Add parent directory to path for proper imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import from src package
try:
    from config.settings import BACKBONES, DEVICE, IMG_SIZE
    from models.architectures import create_custom_backbone
except ImportError:
    # Fallback for different execution contexts
    import sys
    # Try adding src to path if not already there
    src_path = str(Path(__file__).parent.parent)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from config.settings import BACKBONES
    from models.architectures import create_custom_backbone
    IMG_SIZE = 224
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_model_creation() -> bool:
    """Test that all backbones can be instantiated."""
    try:
        for backbone_name in BACKBONES:
            model = create_custom_backbone(backbone_name, num_classes=13)
            assert model is not None, f"{backbone_name} returned None"
            assert isinstance(model, nn.Module), f"{backbone_name} not nn.Module"
        return True
    except Exception as e:
        print(f"x Model creation test failed: {e}")
        return False


def test_forward_pass() -> bool:
    """Test forward pass with dummy input for all backbones."""
    try:
        dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

        for backbone_name in BACKBONES:
            model = create_custom_backbone(backbone_name, num_classes=13).to(DEVICE)
            model.eval()

            with torch.no_grad():
                output = model(dummy_input)

            assert output.shape == (2, 13), f"{backbone_name} wrong output shape: {output.shape}"
            assert not torch.isnan(output).any(), f"{backbone_name} produced NaN"
            assert not torch.isinf(output).any(), f"{backbone_name} produced Inf"

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return True
    except Exception as e:
        print(f"x Forward pass test failed: {e}")
        return False


def test_backward_pass() -> bool:
    """Test backward pass and gradient flow for all backbones."""
    try:
        dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        dummy_target = torch.randint(0, 13, (2,)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()

        for backbone_name in BACKBONES:
            model = create_custom_backbone(backbone_name, num_classes=13).to(DEVICE)
            model.train()

            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()

            # Check gradients exist and are not NaN
            has_gradients = False
            for param in model.parameters():
                if param.grad is not None:
                    has_gradients = True
                    assert not torch.isnan(param.grad).any(), f"{backbone_name} has NaN gradients"
                    assert not torch.isinf(param.grad).any(), f"{backbone_name} has Inf gradients"

            assert has_gradients, f"{backbone_name} has no gradients"

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return True
    except Exception as e:
        print(f"x Backward pass test failed: {e}")
        return False


def test_output_shape() -> bool:
    """Test output shapes are correct for different batch sizes."""
    try:
        batch_sizes = [1, 4, 8]
        num_classes = 13

        for backbone_name in BACKBONES[:3]:  # Test subset for speed
            model = create_custom_backbone(backbone_name, num_classes).to(DEVICE)
            model.eval()

            for bs in batch_sizes:
                dummy_input = torch.randn(bs, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

                with torch.no_grad():
                    output = model(dummy_input)

                expected_shape = (bs, num_classes)
                assert output.shape == expected_shape, \
                    f"{backbone_name} wrong shape for batch {bs}: {output.shape} vs {expected_shape}"

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return True
    except Exception as e:
        print(f"x Output shape test failed: {e}")
        return False


def test_parameter_count() -> bool:
    """Test that models have reasonable parameter counts."""
    try:
        for backbone_name in BACKBONES:
            model = create_custom_backbone(backbone_name, num_classes=13)
            param_count = sum(p.numel() for p in model.parameters())

            # Sanity check: all models should have > 1000 params
            assert param_count > 1000, \
                f"{backbone_name} has suspiciously low param count: {param_count}"

            # Check: no model should exceed 500M params (memory constraint)
            assert param_count < 500_000_000, \
                f"{backbone_name} has excessive param count: {param_count}"

            del model

        return True
    except Exception as e:
        print(f"x Parameter count test failed: {e}")
        return False


def test_device_compatibility() -> bool:
    """Test models work on configured device."""
    try:
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

        for backbone_name in BACKBONES[:2]:  # Test subset
            model = create_custom_backbone(backbone_name, num_classes=13).to(DEVICE)
            model.eval()

            with torch.no_grad():
                output = model(dummy_input)

            assert output.device.type == DEVICE, \
                f"{backbone_name} output on wrong device: {output.device}"

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return True
    except Exception as e:
        print(f"x Device compatibility test failed: {e}")
        return False


def run_all_unit_tests() -> tuple[int, int]:
    """
    Run all unit tests and return (passed, failed) counts.

    Returns:
        Tuple[int, int]: (number of passed tests, number of failed tests)
    """
    print("\n" + "="*80)
    print("RUNNING UNIT TEST SUITE")
    print("="*80)

    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Backward Pass", test_backward_pass),
        ("Output Shape", test_output_shape),
        ("Parameter Count", test_parameter_count),
        ("Device Compatibility", test_device_compatibility),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}...", end=" ")
        try:
            result = test_func()
            if result:
                print("[PASS]")
                passed += 1
            else:
                print("[FAIL]")
                failed += 1
        except Exception as e:
            print(f"[FAIL]: {e}")
            print(f"x FAILED: {e}")
            failed += 1

    print("\n" + "="*80)
    print(f"UNIT TEST RESULTS: {passed} passed, {failed} failed")
    print("="*80)

    return passed, failed


if __name__ == '__main__':
    # Run tests if executed directly
    passed, failed = run_all_unit_tests()
    sys.exit(0 if failed == 0 else 1)
