# ============================================================================
# DISEASE Classification Pipeline - Export Format Tests
# ============================================================================
# Created: Sprint 2 - CI/CD Without Behavior Change
# Purpose: Verify ONNX and TorchScript export functionality
# ============================================================================

import pytest
import torch
import tempfile
from pathlib import Path


class TestTorchScriptExport:
    """Test TorchScript export functionality."""

    def test_torchscript_trace_convnext(self, num_classes, img_size, device):
        """Test TorchScript tracing for CustomConvNeXt."""
        from Base_backbones import create_custom_backbone
        
        model = create_custom_backbone("CustomConvNeXt", num_classes)
        model = model.to(device)
        model.eval()
        
        example_input = torch.randn(1, 3, img_size, img_size, device=device)
        
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)  # type: ignore[arg-type]
        
        assert traced_model is not None
        
        # Verify traced model produces same output
        with torch.no_grad():
            original_output = model(example_input)
            traced_output = traced_model(example_input)  # type: ignore[operator]
        
        assert torch.allclose(original_output, traced_output, atol=1e-5)

    def test_torchscript_save_load(self, num_classes, img_size, device):
        """Test TorchScript save and load."""
        from Base_backbones import create_custom_backbone
        
        model = create_custom_backbone("CustomResNetMish", num_classes)
        model = model.to(device)
        model.eval()
        
        example_input = torch.randn(1, 3, img_size, img_size, device=device)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            
            # Trace and save
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)  # type: ignore[arg-type]
            traced_model.save(str(save_path))  # type: ignore[union-attr]
            
            assert save_path.exists()
            
            # Load and verify
            loaded_model = torch.jit.load(str(save_path), map_location=device)
            
            with torch.no_grad():
                original_output = model(example_input)
                loaded_output = loaded_model(example_input)
            
            assert torch.allclose(original_output, loaded_output, atol=1e-5)

    @pytest.mark.parametrize("backbone_name", [
        "CustomConvNeXt",
        "CustomResNetMish",
        "CustomCoAtNet",
    ])
    def test_torchscript_multiple_backbones(
        self, backbone_name, num_classes, img_size, device
    ):
        """Test TorchScript export for multiple backbones."""
        from Base_backbones import create_custom_backbone
        
        model = create_custom_backbone(backbone_name, num_classes)
        model = model.to(device)
        model.eval()
        
        example_input = torch.randn(1, 3, img_size, img_size, device=device)
        
        try:
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)  # type: ignore[arg-type]
            
            # Verify output
            with torch.no_grad():
                traced_output = traced_model(example_input)  # type: ignore[operator]
            
            assert traced_output.shape == (1, num_classes)
        except Exception as e:
            pytest.fail(f"TorchScript export failed for {backbone_name}: {e}")


class TestONNXExport:
    """Test ONNX export functionality."""

    def test_onnx_export_convnext(self, num_classes, img_size):
        """Test ONNX export for CustomConvNeXt."""
        import onnx
        from Base_backbones import create_custom_backbone
        
        model = create_custom_backbone("CustomConvNeXt", num_classes)
        model = model.cpu()  # ONNX export requires CPU
        model.eval()
        
        example_input = torch.randn(1, 3, img_size, img_size)  # CPU tensor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.onnx"
            
            torch.onnx.export(
                model,
                (example_input,),
                str(save_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
                opset_version=17,
            )
            
            assert save_path.exists()
            
            # Verify ONNX model is valid
            onnx_model = onnx.load(str(save_path))
            onnx.checker.check_model(onnx_model)

    def test_onnx_inference(self, num_classes, img_size):
        """Test ONNX model inference with ONNX Runtime."""
        import onnxruntime as ort
        import numpy as np
        from Base_backbones import create_custom_backbone
        
        model = create_custom_backbone("CustomResNetMish", num_classes)
        model = model.cpu()  # ONNX export requires CPU
        model.eval()
        
        example_input = torch.randn(1, 3, img_size, img_size)  # CPU tensor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.onnx"
            
            torch.onnx.export(
                model,
                (example_input,),
                str(save_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
            )
            
            # Create ONNX Runtime session
            session = ort.InferenceSession(
                str(save_path),
                providers=["CPUExecutionProvider"],
            )
            
            # Run inference
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            onnx_result = session.run(
                [output_name],
                {input_name: example_input.numpy()},
            )
            onnx_output = np.asarray(onnx_result[0])
            
            # Verify output shape
            assert onnx_output.shape == (1, num_classes)
            
            # Verify output matches PyTorch
            with torch.no_grad():
                pytorch_output = model(example_input).numpy()
            
            np.testing.assert_allclose(
                pytorch_output, onnx_output, rtol=1e-3, atol=1e-5
            )

    @pytest.mark.parametrize("backbone_name", [
        "CustomConvNeXt",
        "CustomResNetMish",
    ])
    def test_onnx_multiple_backbones(self, backbone_name, num_classes, img_size):
        """Test ONNX export for multiple backbones."""
        import onnx
        from Base_backbones import create_custom_backbone
        
        model = create_custom_backbone(backbone_name, num_classes)
        model = model.cpu()  # ONNX export requires CPU
        model.eval()
        
        example_input = torch.randn(1, 3, img_size, img_size)  # CPU tensor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.onnx"
            
            try:
                torch.onnx.export(
                    model,
                    (example_input,),
                    str(save_path),
                    input_names=["input"],
                    output_names=["output"],
                    opset_version=17,
                )
                
                # Verify model is valid
                onnx_model = onnx.load(str(save_path))
                onnx.checker.check_model(onnx_model)
            except Exception as e:
                pytest.fail(f"ONNX export failed for {backbone_name}: {e}")


class TestExportIntegrity:
    """Test export file integrity and metadata."""

    def test_onnx_file_size_reasonable(self, num_classes, img_size):
        """Test that ONNX file size is reasonable."""
        from Base_backbones import create_custom_backbone
        
        model = create_custom_backbone("CustomResNetMish", num_classes)
        model = model.cpu()  # ONNX export requires CPU
        model.eval()
        
        example_input = torch.randn(1, 3, img_size, img_size)  # CPU tensor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.onnx"
            
            torch.onnx.export(
                model,
                (example_input,),
                str(save_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
            )
            
            file_size_mb = save_path.stat().st_size / (1024 * 1024)
            
            # ONNX file should be at least 1MB (has real weights)
            # and less than 500MB (reasonable model size)
            assert 1 < file_size_mb < 500, \
                f"ONNX file size {file_size_mb:.1f}MB seems unreasonable"

    def test_torchscript_file_size_reasonable(self, num_classes, img_size, device):
        """Test that TorchScript file size is reasonable."""
        from Base_backbones import create_custom_backbone
        
        model = create_custom_backbone("CustomResNetMish", num_classes)
        model = model.to(device)
        model.eval()
        
        example_input = torch.randn(1, 3, img_size, img_size, device=device)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)  # type: ignore[arg-type]
            traced_model.save(str(save_path))  # type: ignore[union-attr]
            
            file_size_mb = save_path.stat().st_size / (1024 * 1024)
            
            # TorchScript file should be at least 1MB and less than 500MB
            assert 1 < file_size_mb < 500, \
                f"TorchScript file size {file_size_mb:.1f}MB seems unreasonable"
