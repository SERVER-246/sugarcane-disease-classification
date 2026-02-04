# ============================================================================
# DEPENDENCY MANIFEST - DISEASE Classification Pipeline
# ============================================================================
# Created: Sprint 1 - Repository Integrity & Safety Baseline
# Purpose: Document all dependencies, versions, and compatibility constraints
# Last Updated: 2026-02-04
# ============================================================================

## Overview

This document tracks all dependencies for the DISEASE Classification Pipeline,
including version constraints, compatibility notes, and security considerations.

---

## Python Version Support

| Version | Status | CI Matrix | Notes |
|---------|--------|-----------|-------|
| 3.9     | ✅ Supported | Yes | Minimum supported version |
| 3.10    | ✅ Primary | Yes | Development version |
| 3.11    | ✅ Supported | Yes | Full compatibility |
| 3.12    | ✅ Supported | Yes | Latest stable |
| 3.13    | ❌ Excluded | No | Not yet stable for ML workloads |

---

## Core Dependencies

### Deep Learning Framework

| Package | Required Version | Installed Version | Notes |
|---------|-----------------|-------------------|-------|
| torch | 2.6.0 | 2.6.0+cu124 | CUDA 12.4 build |
| torchvision | 0.21.0 | 0.21.0+cu124 | Matches torch version |
| torchaudio | 2.6.0 | 2.6.0+cu124 | Matches torch version |

### Data Processing

| Package | Required Version | Installed Version | Notes |
|---------|-----------------|-------------------|-------|
| numpy | 1.26.4 | 2.2.6 | ⚠️ Installed newer than required |
| pandas | 2.3.2 | 2.3.2 | ✅ Match |
| scipy | 1.15.3 | - | Scientific computing |
| scikit-learn | 1.4.2 | 1.4.2 | ✅ Match |

### Image Processing

| Package | Required Version | Installed Version | Notes |
|---------|-----------------|-------------------|-------|
| Pillow | 10.4.0 | 10.4.0 | ✅ Match |
| opencv-python | 4.9.0.80 | 4.9.0.80 | ✅ Match |

### Visualization

| Package | Required Version | Installed Version | Notes |
|---------|-----------------|-------------------|-------|
| matplotlib | 3.8.4 | 3.8.4 | ✅ Match |
| seaborn | 0.13.2 | 0.13.2 | ✅ Match |

### Model Export

| Package | Required Version | Installed Version | Notes |
|---------|-----------------|-------------------|-------|
| onnx | 1.16.2 | 1.16.2 | ✅ Match |
| onnxruntime | 1.17.3 | 1.23.2 | ⚠️ Newer installed |
| tensorrt | 10.13.3.9 | 10.13.3.9 | NVIDIA optimization |
| coremltools | 8.3.0 | 8.3.0 | Apple export |

### Machine Learning

| Package | Required Version | Installed Version | Notes |
|---------|-----------------|-------------------|-------|
| xgboost | 3.1.1 | 3.1.1 | ✅ Match |
| timm | 0.9.16 | 0.9.16 | PyTorch Image Models |

---

## Development Dependencies

| Package | Required Version | Purpose |
|---------|-----------------|---------|
| ruff | >=0.4.0 | Fast linter & formatter |
| mypy | >=1.10.0 | Static type checking |
| pyright | >=1.1.350 | Type checking (VS Code) |
| vulture | >=2.11 | Dead code detection |
| pytest | >=8.0.0 | Testing framework |
| pytest-cov | >=5.0.0 | Coverage reporting |
| pre-commit | >=3.7.0 | Git hooks |
| bandit | >=1.7.8 | Security scanning |

---

## Version Constraints

### Hard Constraints (Breaking Changes)

1. **PyTorch 2.6.x** - Required for AMP autocast API
2. **Python >=3.9** - f-string improvements, typing features
3. **CUDA 11.8/12.x** - GPU support (12.4 recommended)
4. **NumPy <2.0** - Some dependencies not yet compatible with NumPy 2.0

### Soft Constraints (Recommended)

1. **onnxruntime 1.17+** - Performance improvements
2. **timm 0.9+** - Latest pretrained weights
3. **Pillow 10+** - Security patches

---

## Security Considerations

### Known Vulnerabilities

Run `safety check -r requirements.txt` regularly to scan for vulnerabilities.

### Pinned vs Flexible Versions

| Category | Strategy | Rationale |
|----------|----------|-----------|
| Core ML (torch, numpy) | Pinned | Reproducibility critical |
| Export (onnx, tensorrt) | Pinned | Model compatibility |
| Utilities (tqdm, PyYAML) | Flexible | Minor updates safe |
| Dev tools (ruff, pytest) | Flexible | Always use latest |

---

## Compatibility Matrix

### GPU/CUDA Compatibility

| CUDA Version | PyTorch | TensorRT | Status |
|--------------|---------|----------|--------|
| 11.8 | 2.6.0+cu118 | 8.6.x | ✅ Tested |
| 12.1 | 2.6.0+cu121 | 10.x | ✅ Tested |
| 12.4 | 2.6.0+cu124 | 10.13.x | ✅ Primary |
| CPU-only | 2.6.0+cpu | N/A | ✅ Fallback |

### OS Compatibility

| OS | Status | Notes |
|----|--------|-------|
| Windows 10/11 | ✅ Primary | Development platform |
| Ubuntu 20.04+ | ✅ Supported | CI/deployment |
| macOS (ARM) | ⚠️ Limited | CoreML export only |

---

## Installation Commands

### Production Installation

```bash
# GPU (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

---

## Dependency Updates

### Update Procedure

1. Create branch: `git checkout -b deps/update-YYYY-MM-DD`
2. Update single package: `pip install --upgrade <package>`
3. Run full test suite: `pytest`
4. Run smoke tests: `python -c "from Base_backbones import create_custom_backbone; m = create_custom_backbone('CustomCoAtNet', 13)"`
5. Update this manifest
6. Create PR with changelog

### Frozen Dependencies

To generate exact versions for reproducibility:

```bash
pip freeze > requirements-frozen.txt
```

---

## Sprint 1 Baseline Analysis

### Static Analysis Results (2026-02-04)

| Tool | Issues Found | Report |
|------|-------------|--------|
| Ruff | 445 issues (231 auto-fixable) | sprint1_ruff_report.txt |
| Pyright | 363 type errors | sprint1_pyright_report.json |
| Vulture | 32 dead code findings | sprint1_dead_code_report.txt |
| Compile Check | 0 errors | All files compile |
| Import Validation | 7/7 pass | UTF-8 encoding required |

### Key Observations

1. **Type Errors**: Most are in `ensemble_system/` - missing `config.settings` and `utils` modules
2. **Dead Code**: Unused imports can be auto-fixed by ruff
3. **sklearn API**: `zero_division=0` should be `zero_division="warn"` for newer sklearn
4. **Encoding**: Windows console requires `PYTHONIOENCODING=utf-8` for emoji output

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-02-04 | Initial manifest created (Sprint 1) | Copilot |

