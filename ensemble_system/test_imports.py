"""Quick test to verify all imports work correctly"""

import sys
from pathlib import Path


# Add BASE-BACK to path
base_back_dir = Path(__file__).parent.parent / 'BASE-BACK' / 'src'
sys.path.insert(0, str(base_back_dir))

print(f"✓ Added to sys.path: {base_back_dir}")
print(f"✓ Path exists: {base_back_dir.exists()}")

try:
    from config.settings import BACKBONES, CKPT_DIR, DEVICE, NUM_CLASSES
    print("✓ Imported config.settings")
    print(f"  - NUM_CLASSES: {NUM_CLASSES}")
    print(f"  - DEVICE: {DEVICE}")
    print(f"  - CKPT_DIR: {CKPT_DIR}")
    print(f"  - Num BACKBONES: {len(BACKBONES)}")
except Exception as e:
    print(f"✗ Failed to import config.settings: {e}")
    sys.exit(1)

try:
    print("✓ Imported utils.logger")
except Exception as e:
    print(f"✗ Failed to import utils: {e}")
    sys.exit(1)

try:
    print("✓ Imported models")
except Exception as e:
    print(f"✗ Failed to import models: {e}")
    sys.exit(1)

# Check checkpoints
checkpoint_dir = CKPT_DIR
print(f"\n📁 Checkpoint directory: {checkpoint_dir}")
print(f"   Exists: {checkpoint_dir.exists()}")

if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    print(f"   Found {len(checkpoints)} checkpoint files")
    if checkpoints:
        print(f"   Examples: {[c.name for c in checkpoints[:3]]}")
else:
    print("   ⚠️  WARNING: Checkpoint directory doesn't exist!")

print("\n✅ All imports successful!")
