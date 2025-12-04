"""
Pipeline Validation Script
Quick checks before running the full pipeline
"""

import sys
from pathlib import Path

# Add BASE-BACK to path
base_back_dir = Path(__file__).parent.parent / 'BASE-BACK' / 'src'
sys.path.insert(0, str(base_back_dir))

print("="*80)
print("15-COIN ENSEMBLE PIPELINE VALIDATION")
print("="*80)

# Check 1: Imports
print("\n[1/5] Checking imports...")
try:
    from run_15coin_pipeline import run_complete_15coin_pipeline
    from stage1_individual import extract_all_predictions_and_embeddings
    from stage2_score_ensembles import train_all_score_ensembles
    from stage3_stacking import train_all_stacking_models
    from stage4_feature_fusion import train_all_fusion_models
    from stage5_mixture_experts import train_mixture_of_experts
    from stage6_meta_ensemble import train_meta_ensemble_controller
    from stage7_distillation import train_distilled_student
    print("  ✅ All imports successful")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

# Check 2: Required directories
print("\n[2/5] Checking dataset directories...")
from config.settings import TRAIN_DIR, VAL_DIR, TEST_DIR, SPLIT_DIR

for dir_path, name in [(TRAIN_DIR, "Train"), (VAL_DIR, "Val"), (TEST_DIR, "Test")]:
    if Path(dir_path).exists():
        num_classes = len(list(Path(dir_path).iterdir()))
        print(f"  ✅ {name} directory found ({num_classes} classes)")
    else:
        print(f"  ❌ {name} directory not found: {dir_path}")
        sys.exit(1)

# Check 3: Trained backbone checkpoints
print("\n[3/5] Checking trained backbone checkpoints...")
from config.settings import BACKBONES, CKPT_DIR

found_backbones = 0
missing_backbones = []

for backbone_name in BACKBONES:
    # Check multiple checkpoint patterns (matching stage1_individual.py)
    checkpoint_candidates = [
        CKPT_DIR / f'{backbone_name}_final.pth',
        CKPT_DIR / f'{backbone_name}_finetune_best.pth',
        CKPT_DIR / f'{backbone_name}_head_best.pth'
    ]
    
    if any(ckpt.exists() for ckpt in checkpoint_candidates):
        found_backbones += 1
    else:
        missing_backbones.append(backbone_name)

print(f"  Found {found_backbones}/{len(BACKBONES)} backbone checkpoints")
if missing_backbones:
    print(f"  ⚠️  Missing: {', '.join(missing_backbones)}")
    print(f"  ⚠️  Stage 1 will skip missing backbones")
else:
    print(f"  ✅ All backbone checkpoints found")

# Check 4: Output directories
print("\n[4/5] Checking output directories...")
from run_15coin_pipeline import ENSEMBLE_DIR, STAGE1_DIR, STAGE2_DIR, STAGE3_DIR, STAGE4_DIR, STAGE5_DIR, STAGE6_DIR, STAGE7_DIR

for dir_path in [ENSEMBLE_DIR, STAGE1_DIR, STAGE2_DIR, STAGE3_DIR, STAGE4_DIR, STAGE5_DIR, STAGE6_DIR, STAGE7_DIR]:
    if dir_path.exists():
        print(f"  ✅ {dir_path.name} exists")
    else:
        print(f"  ℹ️  {dir_path.name} will be created")

# Check 5: Previous stage results (for recovery)
print("\n[5/5] Checking for existing stage results (recovery)...")
from ensemble_checkpoint_manager import EnsembleCheckpointManager

checkpoint_mgr = EnsembleCheckpointManager()
recovery_status = checkpoint_mgr.get_recovery_status()

if recovery_status['recovery_available']:
    completed = recovery_status.get('completed_stages', [])
    print(f"  ℹ️  Recovery available - {len(completed)} stages completed:")
    for stage in completed:
        print(f"     - {stage}")
    print(f"  ℹ️  Pipeline will resume from stage {len(completed) + 1}")
else:
    print(f"  ℹ️  No previous results - will start from Stage 1")

# Check 6: CUDA availability
print("\n[6/6] Checking CUDA availability...")
import torch
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  ✅ CUDA available: {device_name} ({memory_gb:.1f} GB)")
else:
    print(f"  ⚠️  CUDA not available - will use CPU (slower)")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)

if found_backbones > 0:
    print("\n✅ Pipeline is ready to run!")
    print("\nRun the complete pipeline with:")
    print("  python test_pipeline.py")
    print("\nOr run directly:")
    print("  python run_15coin_pipeline.py")
    
    if found_backbones < len(BACKBONES):
        print(f"\n⚠️  Note: Only {found_backbones}/{len(BACKBONES)} backbones available")
        print("   Pipeline will work with available backbones")
else:
    print("\n❌ No trained backbones found!")
    print("   Please train backbones first using Base_backbones.py")

print("="*80)
