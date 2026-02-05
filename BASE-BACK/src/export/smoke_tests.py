"""Smoke tests for exported models - verify correctness after export"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


# Handle both package imports and direct sys.path imports
try:
    from ..utils import logger
except ImportError:
    from utils import logger


class SmokeTestValidator:
    """Validates exported models produce correct outputs"""

    def __init__(self, num_classes: int = 13, image_size: int = 224):
        """
        Initialize validator.

        Args:
            num_classes: Number of output classes
            image_size: Image input size
        """
        self.num_classes = num_classes
        self.image_size = image_size
        self.dummy_input = torch.randn(1, 3, image_size, image_size)

    def test_state_dict(self, model: nn.Module, checkpoint_path: Path) -> Dict[str, Any]:
        """Test state_dict export and loading."""
        try:
            # Load state dict
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)

            # Test forward pass
            with torch.no_grad():
                model.eval()
                output = model(self.dummy_input)

            assert output.shape == (1, self.num_classes), f"Wrong output shape: {output.shape}"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"

            return {'status': 'pass', 'output_shape': str(output.shape)}
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}

    def test_torchscript(self, model_path: Path) -> Dict[str, Any]:
        """Test TorchScript export and inference."""
        try:
            # Load and test TorchScript model
            scripted_model = torch.jit.load(str(model_path), map_location='cpu')

            with torch.no_grad():
                scripted_model.eval()
                output = scripted_model(self.dummy_input)

            if isinstance(output, tuple):
                output = output[0]  # Handle multi-output models

            assert output.shape == (1, self.num_classes), f"Wrong output shape: {output.shape}"
            assert not torch.isnan(output).any(), "Output contains NaN"

            return {'status': 'pass', 'output_shape': str(output.shape)}
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}

    def test_onnx(self, model_path: Path) -> Dict[str, Any]:
        """Test ONNX export and inference."""
        try:
            import onnx
            import onnxruntime as ort

            # Load ONNX model
            onnx_model = onnx.load(str(model_path))
            onnx.checker.check_model(onnx_model)

            # Create ONNX Runtime session
            session = ort.InferenceSession(str(model_path))

            # Run inference
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            input_data = self.dummy_input.numpy()
            output = session.run([output_name], {input_name: input_data})[0]

            assert output.shape == (1, self.num_classes), f"Wrong output shape: {output.shape}"
            assert not np.isnan(output).any(), "Output contains NaN"

            return {'status': 'pass', 'output_shape': str(output.shape)}
        except ImportError:
            return {'status': 'skip', 'reason': 'ONNX packages not installed'}
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}

    def test_tflite(self, model_path: Path) -> Dict[str, Any]:
        """Test TFLite export and inference."""
        try:
            import tensorflow as tf

            # Load TFLite model
            interpreter = tf.lite.Interpreter(str(model_path))
            interpreter.allocate_tensors()

            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Prepare input
            input_data = self.dummy_input.numpy().astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            assert output.shape[0] == 1, f"Wrong batch size: {output.shape[0]}"
            assert not np.isnan(output).any(), "Output contains NaN"

            return {'status': 'pass', 'output_shape': str(output.shape)}
        except ImportError:
            return {'status': 'skip', 'reason': 'TensorFlow not installed'}
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}


def run_export_smoke_tests(
    model: nn.Module,
    export_results: Dict[str, Any],
    model_dir: Path,
    num_classes: int = 13,
    image_size: int = 224
) -> Dict[str, Dict[str, Any]]:
    """
    Run smoke tests on all exported formats.

    Args:
        model: Original PyTorch model
        export_results: Results from export
        model_dir: Directory containing exported models
        num_classes: Number of output classes
        image_size: Image size

    Returns:
        Dictionary of test results for each format
    """
    validator = SmokeTestValidator(num_classes, image_size)
    test_results = {}

    logger.info("Running export smoke tests...")

    for export_format in export_results.keys():
        logger.info(f"  Testing {export_format}...")

        try:
            if export_format == 'state_dict':
                model_path = model_dir / 'model.pth'
                result = validator.test_state_dict(model, model_path)

            elif export_format == 'torchscript':
                model_path = model_dir / 'model.pt'
                result = validator.test_torchscript(model_path)

            elif export_format == 'onnx':
                model_path = model_dir / 'model.onnx'
                result = validator.test_onnx(model_path)

            elif export_format == 'tflite':
                model_path = model_dir / 'model.tflite'
                result = validator.test_tflite(model_path)

            else:
                result = {'status': 'unknown', 'format': export_format}

            test_results[export_format] = result
            logger.info(f"    [OK] {export_format}: {result['status']}")

        except Exception as e:
            test_results[export_format] = {'status': 'error', 'error': str(e)}
            logger.error(f"    [FAIL] {export_format}: {e}")

    return test_results
