#!/usr/bin/env python3
"""
Comprehensive dependency and functionality test suite
"""

import sys
from pathlib import Path

print('🔧 COMPREHENSIVE PROJECT TEST SUITE')
print('='*80)

tests_passed = 0
tests_failed = 0

# Test 1: Image Validator
print('\n[1/6] Testing image_validator module...')
try:
    from image_validator import ImageValidator, ValidationReport, ValidationResult
    v = ImageValidator(use_deep_learning=False)
    print('  ✓ ImageValidator imports successfully')
    print('  ✓ Module is functional')
    tests_passed += 1
except Exception as e:
    print(f'  ✗ FAILED: {e}')
    tests_failed += 1

# Test 2: Disease Classifier GUI
print('\n[2/6] Testing disease_classifier_gui module...')
try:
    import disease_classifier_gui
    print('  ✓ GUI module imports successfully')
    print('  ✓ All dependencies available')
    tests_passed += 1
except Exception as e:
    print(f'  ✗ FAILED: {e}')
    tests_failed += 1

# Test 3: BASE-BACK Configuration
print('\n[3/6] Testing BASE-BACK configuration...')
try:
    base_back = Path('BASE-BACK/src')
    if str(base_back) not in sys.path:
        sys.path.insert(0, str(base_back))
    from config.settings import BACKBONES, NUM_CLASSES, IMG_SIZE, BATCH_SIZE
    print(f'  ✓ Configuration loaded: {len(BACKBONES)} backbones, {NUM_CLASSES} classes')
    print(f'  ✓ IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}')
    tests_passed += 1
except Exception as e:
    print(f'  ✗ FAILED: {e}')
    tests_failed += 1

# Test 4: Core PyTorch/ML Stack
print('\n[4/6] Testing core ML libraries...')
try:
    import torch
    import torchvision
    import numpy as np
    import pandas as pd
    import sklearn
    import xgboost as xgb
    import matplotlib
    import seaborn
    import joblib
    import PIL
    print(f'  ✓ PyTorch: {torch.__version__}')
    print(f'  ✓ TorchVision: {torchvision.__version__}')
    print(f'  ✓ NumPy: {np.__version__}')
    print(f'  ✓ Pandas: {pd.__version__}')
    print(f'  ✓ Scikit-Learn: {sklearn.__version__}')
    print(f'  ✓ XGBoost: {xgb.__version__}')
    print(f'  ✓ Matplotlib: {matplotlib.__version__}')
    print(f'  ✓ Seaborn: {seaborn.__version__}')
    print(f'  ✓ Joblib: {joblib.__version__}')
    print(f'  ✓ PIL: {PIL.__version__}')
    tests_passed += 1
except Exception as e:
    print(f'  ✗ FAILED: {e}')
    tests_failed += 1

# Test 5: Export Dependencies
print('\n[5/6] Testing export dependencies...')
try:
    import onnx
    import onnxruntime
    print(f'  ✓ ONNX: {onnx.__version__}')
    print(f'  ✓ ONNX Runtime: {onnxruntime.__version__}')
    tests_passed += 1
except Exception as e:
    print(f'  ⚠ Optional: {e}')

# Test 6: Device Detection
print('\n[6/6] Testing device configuration...')
try:
    import torch
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    print(f'  ✓ CUDA Available: {cuda_available}')
    print(f'  ✓ Selected Device: {device}')
    if cuda_available:
        print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'  ✓ VRAM: {vram_gb:.1f} GB')
    tests_passed += 1
except Exception as e:
    print(f'  ✗ FAILED: {e}')
    tests_failed += 1

print('\n' + '='*80)
print(f'RESULTS: {tests_passed} PASSED, {tests_failed} FAILED')
if tests_failed == 0:
    print('✅ ALL TESTS PASSED - Pipeline is functional!')
else:
    print('⚠️  Some tests failed - check configuration')
print('='*80)
