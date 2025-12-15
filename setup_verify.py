#!/usr/bin/env python3
"""
Setup and Verification Script for Sugarcane Disease Classification Pipeline

This script:
1. Checks Python version
2. Verifies all required packages
3. Tests GPU/CUDA availability  
4. Creates necessary directories
5. Validates the pipeline is ready to run
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.10+"""
    print("\n" + "="*80)
    print("PYTHON VERSION CHECK")
    print("="*80)
    
    version = sys.version_info
    print(f"Current Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def check_packages():
    """Check if all required packages are installed"""
    print("\n" + "="*80)
    print("DEPENDENCY CHECK")
    print("="*80)
    
    required = {
        'torch': '2.6.0',
        'torchvision': '0.21.0',
        'numpy': '1.26.4',
        'pandas': '2.3.2',
        'sklearn': '1.4.2',
        'xgboost': '3.1.1',
        'matplotlib': '3.8.4',
        'seaborn': '0.13.2',
        'tqdm': '4.66.6',
        'PIL': '10.4.0',
        'onnx': '1.16.2',
        'onnxruntime': '1.17.3'
    }
    
    all_installed = True
    
    for pkg, expected_version in required.items():
        try:
            if pkg == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif pkg == 'PIL':
                from PIL import Image
                import PIL
                version = PIL.__version__
            else:
                mod = __import__(pkg)
                version = mod.__version__
            
            print(f"  ✅ {pkg:<20} {version}")
        except ImportError:
            print(f"  ❌ {pkg:<20} NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_gpu():
    """Check GPU/CUDA availability"""
    print("\n" + "="*80)
    print("GPU/CUDA CHECK")
    print("="*80)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        print(f"  CUDA Available: {cuda_available}")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {gpu_name}")
            print(f"  VRAM: {vram_gb:.1f} GB")
            print(f"  ✅ GPU acceleration available")
            return True
        else:
            print(f"  ⚠️  CPU mode (no GPU detected)")
            return False
    except Exception as e:
        print(f"  ❌ Error checking GPU: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n" + "="*80)
    print("DIRECTORY SETUP")
    print("="*80)
    
    dirs = [
        'split_dataset/train',
        'split_dataset/val',
        'split_dataset/test',
        'checkpoints',
        'ensembles',
        'metrics_output',
        'plots_metrics',
        'deployment_models',
        'debug_logs'
    ]
    
    base = Path(__file__).parent
    
    for dir_path in dirs:
        full_path = base / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")
    
    print("  ✅ All directories created/verified")
    return True

def test_imports():
    """Test critical imports"""
    print("\n" + "="*80)
    print("MODULE IMPORTS TEST")
    print("="*80)
    
    tests = []
    
    # Test image validator
    try:
        from image_validator import ImageValidator
        print("  ✅ image_validator")
        tests.append(True)
    except Exception as e:
        print(f"  ❌ image_validator: {e}")
        tests.append(False)
    
    # Test disease GUI
    try:
        import disease_classifier_gui
        print("  ✅ disease_classifier_gui")
        tests.append(True)
    except Exception as e:
        print(f"  ❌ disease_classifier_gui: {e}")
        tests.append(False)
    
    # Test BASE-BACK config
    try:
        sys.path.insert(0, 'BASE-BACK/src')
        from config.settings import BACKBONES, NUM_CLASSES
        print(f"  ✅ BASE-BACK config ({len(BACKBONES)} backbones)")
        tests.append(True)
    except Exception as e:
        print(f"  ❌ BASE-BACK config: {e}")
        tests.append(False)
    
    return all(tests)

def main():
    """Run all checks"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "SUGARCANE DISEASE CLASSIFICATION SETUP VERIFICATION" + " "*12 + "║")
    print("╚" + "="*78 + "╝")
    
    results = []
    
    # Run checks
    results.append(("Python Version", check_python_version()))
    results.append(("Packages", check_packages()))
    results.append(("GPU/CUDA", check_gpu()))
    results.append(("Directories", create_directories()))
    results.append(("Imports", test_imports()))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "⚠️  PARTIAL"
        print(f"  {name:<30} {status}")
    
    all_passed = all(p for _, p in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ SETUP COMPLETE - Pipeline is ready to use!")
        print("\nNext steps:")
        print("  1. Run the GUI: python disease_classifier_gui.py")
        print("  2. Run full pipeline: python reproduce_pipeline.py --mode interactive")
        print("  3. Run tests: python test_dependencies.py")
    else:
        print("⚠️  SETUP INCOMPLETE - Fix issues above before proceeding")
        print("\nCommon fixes:")
        print("  - Install PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print("  - Install requirements: pip install -r requirements.txt")
        print("  - Check Python version: python --version")
    
    print("="*80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
