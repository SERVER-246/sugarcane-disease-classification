# 🔄 Checkpoint Recovery System - Complete Guide

## Overview

The checkpoint recovery system protects against system shutdowns, power losses, or unexpected interruptions during the 15-backbone training pipeline. If your system crashes, the training can **resume from where it left off** instead of restarting from scratch.

---

## How It Works

### Key Features

1. **Automatic Checkpointing**
   - Before training each backbone, a checkpoint is saved
   - After each epoch, interim checkpoints are saved
   - All progress is tracked in recovery files

2. **Smart Recovery**
   - Detects which backbones have completed
   - Automatically skips completed backbones
   - Resumes interrupted backbones from last epoch

3. **Zero Manual Intervention**
   - Just run `python src/main.py` again after restart
   - System automatically detects and continues

---

## Technical Details

### Checkpoint Files

**Location:** `checkpoints/`

#### Files Created:
```
checkpoints/
├── .recovery_state.json           # Pipeline recovery state
├── .kfold_progress.json           # K-fold progress
├── {backbone_name}_head_epoch*.pth       # Head training checkpoints
├── {backbone_name}_finetune_epoch*.pth   # Fine-tuning checkpoints
└── recovery_status.txt            # Human-readable status
```

#### Recovery State File (`.recovery_state.json`)
```json
{
  "completed_backbones": ["CustomConvNeXt", "CustomEfficientNetV4"],
  "current_backbone": "CustomGhostNetV2",
  "current_stage": "finetune",
  "current_epoch": 15,
  "results": {
    "CustomConvNeXt": {
      "final_accuracy": 0.95,
      "final_loss": 0.12
    }
  },
  "timestamp": "2025-11-19T14:30:45.123456"
}
```

---

## Usage Guide

### Automatic Recovery

**Scenario: System crashes during training of backbone #5**

```bash
# Training was running...
# SYSTEM CRASHES!

# After restart:
python src/main.py

# Output:
# ✓ Previous training detected!
# ✓ Loaded recovery state
# ✓ Completed: 4 backbones
# ✓ Resuming: CustomGhostNetV2 (epoch 15/25)
# ✓ Remaining: 10 backbones
```

The system will:
1. Load the recovery state
2. Skip the 4 already-completed backbones
3. Resume CustomGhostNetV2 from epoch 15
4. Continue with the remaining 10 backbones

### Manual Recovery Status Check

```python
from src.utils.checkpoint_manager import CheckpointManager

manager = CheckpointManager()
status = manager.get_recovery_status()

print(f"Completed: {status['completed']} / {status['total_backbones']}")
print(f"Next backbone: {status['next_backbone']}")
```

### Export Recovery Report

```python
manager = CheckpointManager()
report_path = manager.export_recovery_summary()
# File will contain human-readable status
```

---

## Code Integration

### In main.py

The checkpoint manager is automatically integrated:

```python
from utils.checkpoint_manager import CheckpointManager

# Initialize checkpoint manager
checkpoint_mgr = CheckpointManager()

# Check if recovery is needed
recovery_status = checkpoint_mgr.get_recovery_status()
if recovery_status['recovery_available']:
    print(f"✓ Resuming from last checkpoint")
    print(f"  Completed: {recovery_status['completed']}")
    print(f"  Next: {recovery_status['next_backbone']}")

# Get next backbone to train
next_backbone = checkpoint_mgr.get_next_backbone_to_train()
for backbone_name in BACKBONES:
    if next_backbone and backbone_name != next_backbone:
        continue  # Skip already completed
    
    # Train backbone
    model, acc, history, metrics = train_backbone_with_metrics(...)
    
    # Save checkpoint after completion
    checkpoint_mgr.mark_backbone_complete(backbone_name, {
        'accuracy': acc,
        'history': history,
        'metrics': metrics
    })
    
    # Cleanup old checkpoints
    checkpoint_mgr.cleanup_old_checkpoints(backbone_name, keep_latest=3)
```

---

## Checkpoint Data Structure

### Backbone Checkpoint Format

```python
{
    'backbone_name': 'CustomSwinTransformer',
    'model_state_dict': {...},  # Model weights
    'epoch': 23,                # Current epoch
    'stage': 'finetune',        # 'head' or 'finetune'
    'metrics': {
        'accuracy': 0.945,
        'loss': 0.125,
        'f1': 0.94
    },
    'history': {
        'train_loss': [...],
        'train_acc': [...],
        'val_loss': [...],
        'val_acc': [...]
    },
    'optimizer_state': {...},   # For resuming optimization
    'scheduler_state': {...},   # For resuming learning rate
    'timestamp': '2025-11-19T14:30:45.123456',
    'device': 'cuda'
}
```

---

## Recovery Scenarios

### Scenario 1: System Crash During Backbone Training

**Before Crash:**
```
Training: CustomGhostNetV2
Stage: finetune
Epoch: 15/25
```

**After Restart:**
```
✓ Recovery detected
✓ Loading checkpoint from epoch 15
✓ Resuming optimizer and scheduler states
✓ Continue training from epoch 16
```

### Scenario 2: System Crash During K-Fold CV

**Before Crash:**
```
K-Fold: CustomCoAtNet
Fold: 3/5
```

**After Restart:**
```
✓ K-fold progress detected
✓ Load fold 1-2 results
✓ Resume fold 3
```

### Scenario 3: Complete Training Run Interrupted

**Before Crash:**
```
Completed: 7/15 backbones
Current: CustomRegNet
```

**After Restart:**
```
✓ 7 backbones already trained
✓ Skip first 7
✓ Continue with CustomRegNet
✓ Train remaining 7 backbones
```

---

## Best Practices

### 1. Regular Status Checks

```bash
# Check progress anytime
python -c "
import sys
sys.path.insert(0, 'src')
from utils.checkpoint_manager import CheckpointManager
m = CheckpointManager()
print(m.export_recovery_summary())
"
```

### 2. Backup Important Results

```python
# After each backbone completes
import shutil
backup_dir = Path('backups') / backbone_name
backup_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(checkpoint_path, backup_dir)
```

### 3. Monitor Checkpoint Size

```python
# Checkpoints are ~500MB-1GB per backbone
# Keep latest 3, delete older ones
checkpoint_mgr.cleanup_old_checkpoints(backbone_name, keep_latest=3)
```

### 4. Verify Recovery on Restart

```bash
# Always verify recovery before continuing
python src/main.py --verify-recovery

# Or in code:
if checkpoint_mgr.get_recovery_status()['recovery_available']:
    print("✓ Ready to resume")
```

---

## Checkpoint Files Reference

### Function: `save_backbone_checkpoint()`
Saves checkpoint after each epoch.

```python
checkpoint_mgr.save_backbone_checkpoint(
    backbone_name='CustomSwinTransformer',
    model_state_dict=model.state_dict(),
    epoch=20,
    metrics={'accuracy': 0.94, 'loss': 0.15},
    history={...},
    optimizer_state=optimizer.state_dict(),
    scheduler_state=scheduler.state_dict(),
    stage='finetune'  # or 'head'
)
```

### Function: `save_pipeline_state()`
Saves overall pipeline state.

```python
checkpoint_mgr.save_pipeline_state({
    'completed_backbones': ['Backbone1', 'Backbone2'],
    'current_backbone': 'Backbone3',
    'current_stage': 'finetune',
    'current_epoch': 15,
    'results': {...}
})
```

### Function: `load_pipeline_state()`
Loads previous state for recovery.

```python
state = checkpoint_mgr.load_pipeline_state()
if state:
    print(f"Resume from: {state['current_backbone']}")
else:
    print("No previous state found - starting fresh")
```

### Function: `get_next_backbone_to_train()`
Returns next backbone to train.

```python
next_backbone = checkpoint_mgr.get_next_backbone_to_train()
if next_backbone:
    print(f"Training: {next_backbone}")
else:
    print("All backbones completed!")
```

---

## Troubleshooting

### Issue: Recovery not detected

**Solution:**
```python
# Check if recovery files exist
from pathlib import Path
checkpoint_dir = Path('checkpoints')
recovery_file = checkpoint_dir / '.recovery_state.json'

if recovery_file.exists():
    print("Recovery file found")
else:
    print("No recovery file - starting fresh")
```

### Issue: Checkpoint files are too large

**Solution:**
```python
# Cleanup older checkpoints
checkpoint_mgr.cleanup_old_checkpoints(
    backbone_name='CustomConvNeXt',
    keep_latest=2  # Keep only 2 latest
)
```

### Issue: Want to reset recovery and start over

**Solution:**
```python
# Clear all recovery state
checkpoint_mgr.reset_recovery()

# Now run training - will start fresh
python src/main.py
```

### Issue: Verify integrity of checkpoint

**Solution:**
```python
# Load and verify checkpoint
import torch
checkpoint = torch.load('checkpoints/CustomConvNeXt_finetune_epoch20.pth')

# Check contents
print(f"Backbone: {checkpoint['backbone_name']}")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Accuracy: {checkpoint['metrics']['accuracy']}")
```

---

## Recovery Workflow Diagram

```
Training Starts
     ↓
Load Recovery State
     ↓
Any previous runs?
     ├─ YES → Get completed backbones
     │         Skip completed
     │         Get next backbone
     │         Load checkpoint
     │         Resume from last epoch
     │
     └─ NO → Start from beginning
             Backbone 1
             ↓
        Save checkpoint after each epoch
             ↓
        Training complete for backbone
             ↓
        Mark as completed
        Save recovery state
             ↓
        Move to next backbone
             ↓
        (repeat for all 15)
             ↓
        All complete?
        ├─ NO → Continue
        └─ YES → Training finished!
```

---

## Performance Impact

- **Checkpoint save time:** ~2-5 seconds per epoch
- **Checkpoint load time:** ~1-2 seconds on resume
- **Storage:** ~500MB-1GB per backbone
- **Memory overhead:** Minimal (<1%)

---

## Files to Keep/Clean

### Keep:
- ✅ All `.pth` checkpoint files in `checkpoints/`
- ✅ `.recovery_state.json` (recovery tracking)
- ✅ `recovery_status.txt` (human-readable status)

### Clean (optional, to free space):
- ✅ Old `.pth` files (after new checkpoint is saved)
- ✅ Temp ONNX files (if exporting)

---

## Example: Complete Recovery Workflow

```python
"""Complete example showing checkpoint recovery in action"""

import sys
sys.path.insert(0, 'src')

from utils.checkpoint_manager import CheckpointManager
from main import train_backbone_with_metrics

# Initialize checkpoint manager
checkpoint_mgr = CheckpointManager()

# Check recovery status
status = checkpoint_mgr.get_recovery_status()
print(f"Progress: {status['completed']} / {status['total_backbones']} completed")

# Get list of backbones to train
backbones_to_train = []
if status['next_backbone']:
    # Find index of next backbone
    from config.settings import BACKBONES
    idx = BACKBONES.index(status['next_backbone'])
    backbones_to_train = BACKBONES[idx:]
else:
    print("All backbones already completed!")
    backbones_to_train = []

# Train each remaining backbone
for backbone_name in backbones_to_train:
    print(f"\nTraining {backbone_name}...")
    
    try:
        # Train backbone
        model, acc, history, metrics = train_backbone_with_metrics(
            backbone_name,
            model,
            train_ds,
            val_ds,
            epochs_head=40,
            epochs_finetune=25
        )
        
        # Mark as completed
        checkpoint_mgr.mark_backbone_complete(backbone_name, {
            'accuracy': acc,
            'history': history,
            'metrics': metrics
        })
        
        print(f"✓ {backbone_name} completed (Accuracy: {acc:.4f})")
        
    except KeyboardInterrupt:
        print(f"Training interrupted - saving checkpoint...")
        checkpoint_mgr.save_pipeline_state({
            'completed_backbones': checkpoint_mgr.get_completed_backbones(),
            'current_backbone': backbone_name,
            'current_stage': 'interrupted'
        })
        break
    
    except Exception as e:
        print(f"✗ Error training {backbone_name}: {e}")
        continue

print("\n" + "="*50)
print("Training Summary:")
final_status = checkpoint_mgr.get_recovery_status()
print(f"Completed: {final_status['completed']} / {final_status['total_backbones']}")
print(f"Remaining: {final_status['remaining']}")
print("="*50)
```

---

## Summary

✅ **Automatic Protection** - Checkpoints saved automatically  
✅ **Smart Recovery** - Resumes from last checkpoint  
✅ **Zero Intervention** - Just run the script again  
✅ **Complete Tracking** - All progress logged  
✅ **Fast Recovery** - Minimal overhead  

**With this system, even with frequent system shutdowns, you can train all 15 backbones to completion!**

---

**Last Updated:** November 19, 2025  
**For Issues:** Check `recovery_status.txt` in checkpoints/ directory
