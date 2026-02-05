"""Core export engine for multi-format model export"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# Handle both package imports and direct sys.path imports
try:
    from ..config.settings import DEPLOY_DIR, IMG_SIZE
    from ..utils import logger
except ImportError:
    from config.settings import DEPLOY_DIR, IMG_SIZE
    from utils import logger


@dataclass
class ExportConfig:
    """Configuration for model export"""
    model_name: str
    backbone_name: str
    num_classes: int
    image_size: int = IMG_SIZE
    export_formats: List[str] = None
    output_dir: Optional[Path] = None
    include_metadata: bool = True
    device: str = 'cpu'

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ['state_dict', 'torchscript']
        if self.output_dir is None:
            self.output_dir = Path(DEPLOY_DIR)


class ExportEngine:
    """Main export engine coordinating all export formats"""

    def __init__(self, config: ExportConfig):
        """
        Initialize export engine.

        Args:
            config: Export configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory for this model
        self.model_dir = self.output_dir / f"{config.backbone_name}_{config.model_name}"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.export_results = {}

    def export(self, model: nn.Module, dummy_input: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Export model to all configured formats.

        Args:
            model: PyTorch model to export
            dummy_input: Dummy input for tracing (B, 3, H, W)

        Returns:
            Dictionary of export results
        """
        model = model.to(self.config.device)
        model.eval()

        # Create dummy input if not provided
        if dummy_input is None:
            dummy_input = torch.randn(
                1, 3, self.config.image_size, self.config.image_size,
                device=self.config.device
            )

        logger.info(f"Exporting {self.config.backbone_name} to formats: {self.config.export_formats}")

        for export_format in self.config.export_formats:
            try:
                logger.info(f"  Exporting to {export_format}...")

                if export_format == 'state_dict':
                    self._export_state_dict(model)
                elif export_format == 'torchscript':
                    self._export_torchscript(model, dummy_input)
                elif export_format == 'onnx':
                    self._export_onnx(model, dummy_input)
                elif export_format == 'tflite':
                    self._export_tflite(model, dummy_input)
                else:
                    logger.warning(f"Unknown export format: {export_format}")

                self.export_results[export_format] = {
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                }
                logger.info(f"  [OK] {export_format} export successful")

            except Exception as e:
                logger.error(f"  [FAIL] {export_format} export failed: {e}")
                self.export_results[export_format] = {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

        # Save metadata
        if self.config.include_metadata:
            self._save_metadata(model)

        return self.export_results

    def _export_state_dict(self, model: nn.Module) -> str:
        """Export PyTorch state dict."""
        output_path = self.model_dir / 'model.pth'
        torch.save(model.state_dict(), output_path)
        logger.info(f"    Saved to {output_path}")
        return str(output_path)

    def _export_torchscript(self, model: nn.Module, dummy_input: torch.Tensor) -> str:
        """Export TorchScript format."""
        output_path = self.model_dir / 'model.pt'

        try:
            # Try scripting first
            scripted = torch.jit.script(model)
        except Exception:
            # Fall back to tracing
            scripted = torch.jit.trace(model, dummy_input)

        scripted.save(str(output_path))
        logger.info(f"    Saved to {output_path}")
        return str(output_path)

    def _export_onnx(self, model: nn.Module, dummy_input: torch.Tensor) -> str:
        """Export ONNX format."""
        try:
            import onnx
            output_path = self.model_dir / 'model.onnx'

            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                verbose=False
            )

            logger.info(f"    Saved to {output_path}")
            return str(output_path)
        except ImportError:
            raise ImportError("ONNX export requires onnx and onnxruntime packages")

    def _export_tflite(self, model: nn.Module, dummy_input: torch.Tensor) -> str:
        """Export TensorFlow Lite format (via ONNX conversion)."""
        try:
            import onnx
            import tensorflow as tf
            from onnx_tf.backend import prepare

            # First export to ONNX
            onnx_path = self.model_dir / 'model_temp.onnx'
            torch.onnx.export(
                model, dummy_input, str(onnx_path),
                export_params=True, opset_version=12
            )

            # Convert ONNX to TF
            onnx_model = onnx.load(str(onnx_path))
            tf_rep = prepare(onnx_model)

            # Convert TF to TFLite
            output_path = self.model_dir / 'model.tflite'
            converter = tf.lite.TFLiteConverter.from_session(
                tf_rep.export_graph.session,
                [tf_rep.export_graph.inputs[0]],
                [tf_rep.export_graph.outputs[0]]
            )
            tflite_model = converter.convert()

            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            # Clean up temp ONNX file
            onnx_path.unlink()
            logger.info(f"    Saved to {output_path}")
            return str(output_path)
        except ImportError:
            logger.warning("    TFLite export requires onnx_tf and tensorflow packages")
            return ""

    def _save_metadata(self, model: nn.Module) -> None:
        """Save model metadata and configuration."""
        metadata = {
            'backbone_name': self.config.backbone_name,
            'model_name': self.config.model_name,
            'num_classes': self.config.num_classes,
            'image_size': self.config.image_size,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'export_formats': self.config.export_formats,
            'export_results': self.export_results,
            'timestamp': datetime.now().isoformat(),
            'device': self.config.device
        }

        metadata_path = self.model_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  Metadata saved to {metadata_path}")


def export_model(
    model: nn.Module,
    model_name: str,
    output_dir: Path,
    input_shape: tuple = (1, 3, 224, 224),
    formats: List[str] = None,
    class_names: List[str] = None,
    training_metadata: Dict = None
) -> Dict[str, Any]:
    """
    Comprehensive model export with metadata generation.

    Args:
        model: PyTorch model
        model_name: Name of the model
        output_dir: Output directory
        input_shape: Input tensor shape (batch, channels, height, width)
        formats: List of formats to export ['pytorch', 'onnx', 'torchscript']
        class_names: List of class names
        training_metadata: Training metrics and metadata

    Returns:
        Dictionary of export paths by format
    """
    if formats is None:
        formats = ['pytorch', 'onnx', 'torchscript']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_results = {}

    try:
        model.eval()
        model = model.cpu()  # Move to CPU for export

        # Create dummy input
        dummy_input = torch.randn(*input_shape)

        # 1. PyTorch State Dict Export
        if 'pytorch' in formats or 'state_dict' in formats:
            pytorch_path = output_dir / f"{model_name}_state_dict.pth"
            torch.save(model.state_dict(), pytorch_path)
            export_results['pytorch'] = str(pytorch_path)
            logger.info(f"[OK] PyTorch state dict: {pytorch_path}")

        # 2. ONNX Export
        if 'onnx' in formats:
            onnx_path = output_dir / f"{model_name}.onnx"
            try:
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                export_results['onnx'] = str(onnx_path)
                logger.info(f"[OK] ONNX: {onnx_path}")
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")

        # 3. TorchScript Export
        if 'torchscript' in formats:
            torchscript_path = output_dir / f"{model_name}_torchscript.pt"
            try:
                # Try script first, fallback to trace
                try:
                    scripted_model = torch.jit.script(model)
                except:
                    scripted_model = torch.jit.trace(model, dummy_input)
                scripted_model.save(str(torchscript_path))
                export_results['torchscript'] = str(torchscript_path)
                logger.info(f"[OK] TorchScript: {torchscript_path}")
            except Exception as e:
                logger.warning(f"TorchScript export failed: {e}")

        # Generate metadata files
        _generate_export_metadata(
            output_dir=output_dir,
            model_name=model_name,
            input_shape=input_shape,
            class_names=class_names,
            training_metadata=training_metadata,
            export_results=export_results
        )

        logger.info(f"[OK] Export completed: {len(export_results)} formats")

    except Exception as e:
        logger.error(f"x Export failed for {model_name}: {e}")
        logger.exception("Full traceback:")

    return export_results


def _generate_export_metadata(
    output_dir: Path,
    model_name: str,
    input_shape: tuple,
    class_names: List[str] = None,
    training_metadata: Dict = None,
    export_results: Dict = None
):
    """Generate comprehensive metadata files for exported models"""
    import json
    import time

    try:
        # 1. metadata.json - Core model information
        metadata = {
            'model_name': model_name,
            'input_shape': list(input_shape),
            'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'exported_formats': list(export_results.keys()) if export_results else [],
        }

        if training_metadata:
            metadata['training'] = training_metadata

        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"[OK] Created: metadata.json")

        # 2. class_mapping.json - Class names and indices
        if class_names:
            class_mapping = {str(i): name for i, name in enumerate(class_names)}
            class_map_path = output_dir / 'class_mapping.json'
            with open(class_map_path, 'w') as f:
                json.dump(class_mapping, f, indent=2, default=str)
            logger.info(f"[OK] Created: class_mapping.json")

        # 3. export_info.json - Export details
        export_info = {
            'model_name': model_name,
            'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'exported_files': export_results,
            'input_shape': list(input_shape),
            'num_classes': len(class_names) if class_names else None
        }

        export_info_path = output_dir / 'export_info.json'
        with open(export_info_path, 'w') as f:
            json.dump(export_info, f, indent=2, default=str)
        logger.info(f"[OK] Created: export_info.json")

        # 4. README.txt - Usage instructions
        readme_content = f"""
# {model_name} - Exported Model

## Model Information
- Model Name: {model_name}
- Export Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Input Shape: {input_shape}
- Number of Classes: {len(class_names) if class_names else 'Unknown'}

## Exported Formats
{chr(10).join(f'- {fmt}: {path}' for fmt, path in (export_results or {}).items())}

## Usage

### PyTorch
```python
import torch
model = YourModelClass()
model.load_state_dict(torch.load('{model_name}_state_dict.pth'))
model.eval()
```

### ONNX
```python
import onnxruntime as ort
session = ort.InferenceSession('{model_name}.onnx')
outputs = session.run(None, {{'input': input_data}})
```

### TorchScript
```python
import torch
model = torch.jit.load('{model_name}_torchscript.pt')
output = model(input_tensor)
```

## Metadata Files
- metadata.json: Core model and training information
- class_mapping.json: Class name to index mapping
- export_info.json: Export details and file paths

## Notes
- All models expect input shape: {input_shape}
- Ensure input data is properly normalized before inference
- Refer to training metadata for normalization statistics
"""

        readme_path = output_dir / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        logger.info(f"[OK] Created: README.txt")

    except Exception as e:
        logger.error(f"x Failed to generate metadata files: {e}")


def export_model_legacy(
    model: nn.Module,
    backbone_name: str,
    num_classes: int,
    export_formats: List[str] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Export model to multiple formats.

    Args:
        model: PyTorch model
        backbone_name: Name of backbone architecture
        num_classes: Number of output classes
        export_formats: List of formats to export (default: ['state_dict', 'torchscript'])
        output_dir: Output directory (default: DEPLOY_DIR)

    Returns:
        Dictionary of export results
    """
    if export_formats is None:
        export_formats = ['state_dict', 'torchscript']

    config = ExportConfig(
        model_name='exported_model',
        backbone_name=backbone_name,
        num_classes=num_classes,
        export_formats=export_formats,
        output_dir=output_dir
    )

    engine = ExportEngine(config)

    # Create dummy input
    dummy_input = torch.randn(1, 3, config.image_size, config.image_size)

    return engine.export(model, dummy_input)


def export_to_format(
    model: nn.Module,
    export_format: str,
    output_path: Path,
    dummy_input: Optional[torch.Tensor] = None
) -> bool:
    """
    Export model to a single format.

    Args:
        model: PyTorch model
        export_format: Format name ('state_dict', 'torchscript', 'onnx', 'tflite')
        output_path: Output file path
        dummy_input: Dummy input for tracing

    Returns:
        True if successful, False otherwise
    """
    try:
        model.eval()

        if export_format == 'state_dict':
            torch.save(model.state_dict(), output_path)

        elif export_format == 'torchscript':
            if dummy_input is None:
                dummy_input = torch.randn(1, 3, 224, 224)
            try:
                scripted = torch.jit.script(model)
            except:
                scripted = torch.jit.trace(model, dummy_input)
            scripted.save(str(output_path))

        elif export_format == 'onnx':
            if dummy_input is None:
                dummy_input = torch.randn(1, 3, 224, 224)
            torch.onnx.export(model, dummy_input, str(output_path), opset_version=12)

        else:
            logger.error(f"Unknown format: {export_format}")
            return False

        logger.info(f"[OK] Exported to {export_format}: {output_path}")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Export failed: {e}")
        return False
