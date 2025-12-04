# EXECUTION GUIDE - Disease Classification Pipeline

## Directory Structure

```
f:\DBT-Base-DIr\
├── BASE-BACK/          # Main codebase
│   ├── src/           # Source code
│   │   ├── config/    # Configuration
│   │   ├── models/    # Model architectures
│   │   ├── training/  # Training logic
│   │   ├── utils/     # Utilities
│   │   ├── export/    # Model export
│   │   ├── tests/     # Test suite
│   │   └── main.py    # Main pipeline
│   └── tests/         # Unit tests
├── Data/              # Raw dataset
├── split_dataset/     # Train/val/test splits
├── checkpoints/       # Model checkpoints
├── deployment_models/ # Exported models
├── kfold_results/     # K-fold CV results
└── run_pipeline.py    # Execution script
```

## How to Execute

### Option 1: Using the execution script (RECOMMENDED)
```powershell
cd f:\DBT-Base-DIr
python run_pipeline.py
```

### Option 2: Direct execution
```powershell
cd f:\DBT-Base-DIr
python -c "import sys; sys.path.insert(0, 'BASE-BACK'); from src.main import run_full_pipeline; run_full_pipeline()"
```

### Option 3: Module execution
```powershell
cd f:\DBT-Base-DIr\BASE-BACK
python -m src.main
```

## What Will Happen

The pipeline will execute these stages:

1. **Stage 0.1:** Run unit tests (6 tests)
2. **Stage 0.2:** Prepare datasets (train/val/test split)
3. **Stage 0.3:** Verify all 15 backbone models
4. **For each backbone:**
   - Stage 2.1: K-fold Cross Validation (5 folds)
   - Stage 2.2: Final model training (head + finetune)
   - Stage 2.3: Test set evaluation
   - Stage 2.4: Model export (PyTorch, ONNX, TorchScript)

## Expected Runtime

- **Per backbone:** 30-60 minutes
- **Total (15 backbones):** 7-15 hours

## Output Files

After execution, you'll have:

- `checkpoints/{backbone_name}/` - Training checkpoints
- `deployment_models/{backbone_name}/` - Exported models
- `kfold_results/{backbone_name}/` - K-fold CV results
- `plots_metrics/` - Training curves
- `metrics_output/pipeline_summary.json` - Complete results

## Progress Tracking

The pipeline will display:
- Current backbone being trained (X/15)
- Progress percentage
- Estimated time remaining
- Per-stage completion status

## Checkpoint Recovery

If interrupted:
- The pipeline automatically saves progress
- Re-run the same command to resume
- Already completed backbones will be skipped

## Ready to Execute!

Everything is set up and ready. Just run:

```powershell
python run_pipeline.py
```
