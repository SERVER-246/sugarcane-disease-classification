# .github Directory - Documentation & Backups

## Contents

### 📋 Documentation Files

#### 1. **ARCHITECTURE_AND_WORKFLOW.md** (48.7 KB)
Comprehensive technical documentation covering:
- **System Architecture Overview** - Visual pipeline diagram
- **Project Structure** - Directory organization
- **Core Components** - Logger, seed management, device setup
- **15 Custom Backbones** - Detailed specs with performance targets
- **Training Pipeline** - Head training + fine-tuning stages
- **Data Processing** - Dataset preparation, augmentation, validation
- **Export System** - 6-format export with smoke tests
- **Configuration** - Learning rates, schedulers per backbone
- **Error Handling** - 50+ try-except patterns
- **Debug System** - 13 debug modes for troubleshooting
- **Performance Optimizations** - Mixed precision, gradient clipping
- **Troubleshooting Guide** - Solutions for common issues

**Use**: Reference for architecture understanding and deployment

#### 2. **copilot-instructions.md** (11 KB)
AI agent guidance for code modifications:
- Architecture patterns and conventions
- Best practices for feature additions
- Common gotchas and solutions
- Integration guidelines

**Use**: Context for code reviews and future modifications

### 🔒 Backup Files

#### 3. **Base_backbones_backup.py** (315 KB)
Complete backup of the main training script:
- Timestamp: November 17, 2025
- Status: Production-ready
- Contains all 15 backbones and training logic
- Original file location: `f:\DBT-Base-DIr\Base_backbones.py`

**Use**: Recovery if main file is accidentally modified

---

## Quick Start Guide

### Reading the Documentation

1. **First Time?** → Start with `ARCHITECTURE_AND_WORKFLOW.md` Table of Contents
2. **Need specific info?** → Use Ctrl+F to search by keyword
3. **Implementing features?** → Check `copilot-instructions.md`
4. **Troubleshooting?** → Jump to Troubleshooting Guide in main docs

### Backup Restoration

```bash
# If main script is corrupted or lost:
Copy-Item -Path ".github/Base_backbones_backup.py" `
          -Destination "Base_backbones.py" -Force
```

---

## File Statistics

| File | Size | Lines | Purpose |
|------|------|-------|---------|
| ARCHITECTURE_AND_WORKFLOW.md | 48.7 KB | 1,000+ | Full technical reference |
| Base_backbones_backup.py | 315 KB | 7,903 | Complete backup |
| copilot-instructions.md | 11 KB | 250+ | AI agent guidance |

**Total Documentation**: 64.7 KB of actionable reference material

---

## Key Information Quick Reference

### Architecture Summary
- **Total Backbones**: 15 custom-built (CNN, Transformer, Hybrid)
- **Training Stages**: 2 (Head training 40 epochs + Fine-tuning 25 epochs)
- **K-fold CV**: 5-fold stratified cross-validation
- **Export Formats**: 6 (PyTorch, ONNX, TorchScript, TensorRT, CoreML, TFLite)
- **Enhanced Models**: CustomCoAtNet (12 transformer blocks), CustomMaxViT (10 transformer blocks)

### Performance Targets
- **CNN-based**: 82-87% accuracy
- **Hybrid models**: 87-90% accuracy
- **Transformer-based**: 85-89% accuracy
- **Enhanced models**: **88-90%** accuracy

### Configuration
- **Input Size**: 224×224 pixels
- **Batch Size**: 32
- **Device**: Auto-detect CUDA/CPU
- **Mixed Precision**: Enabled on CUDA
- **Gradient Clipping**: 1.0 norm

### Windows Optimization
- Multiprocessing method: `spawn` (line 26)
- Path handling: Windows-compatible via `pathlib.Path`
- DataLoader: Custom `WindowsCompatibleImageFolder` class

---

## Troubleshooting Quick Links

**Problem** → **Solution Section**
- CUDA OOM → "CUDA Out of Memory" in Troubleshooting
- Dataset not found → "Dataset Not Found"
- Model won't converge → "NaN Loss During Training"
- Export fails → "Export Format Fails"
- Slow speed → "Slow Training Speed"

---

## Version Information

- **Script Version**: Final Production Ready
- **Date**: November 17, 2025
- **Total Lines**: 7,903
- **Status**: ✅ APPROVED FOR PRODUCTION
- **Audit Status**: COMPLETE

---

## Maintenance Notes

### When to Update These Files

1. **Script Major Changes** → Update backup
2. **New Features** → Document in ARCHITECTURE_AND_WORKFLOW.md
3. **Architecture Updates** → Update copilot-instructions.md
4. **Performance Improvements** → Log in respective docs

### Regular Maintenance

- Review logs: `training_run_*.log`
- Check metrics: `metrics_output/` directory
- Archive results: Keep checkpoint files for reproducibility

---

## Contact & Support

For issues or questions:
1. Check Troubleshooting Guide in ARCHITECTURE_AND_WORKFLOW.md
2. Review debug modes (13 available)
3. Enable debug logging with `DBT_DEBUG_MODE=true`

---

**Last Updated**: November 17, 2025  
**Created By**: Comprehensive Audit & Documentation Process  
**Status**: ✅ Ready for Production Deployment
