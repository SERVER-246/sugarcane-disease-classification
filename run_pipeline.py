"""
Execute Disease Classification Backbone Training Pipeline
Run this script to start the complete training workflow
"""

import sys
from pathlib import Path
import os

# Add BASE-BACK to Python path for imports
base_back_dir = Path(__file__).parent / 'BASE-BACK'
sys.path.insert(0, str(base_back_dir))

# Change to BASE-BACK directory for relative imports to work
os.chdir(str(base_back_dir))

# Import and run the pipeline
try:
    from src.main import run_full_pipeline  # type: ignore
except ImportError as e:
    # Fallback: try direct import after path adjustment
    sys.path.insert(0, str(base_back_dir / 'src'))
    from main import run_full_pipeline  # type: ignore

if __name__ == '__main__':
    print("\n" + "="*80)
    print("STARTING DISEASE CLASSIFICATION TRAINING PIPELINE")
    print("="*80)
    print(f"Working directory: {Path.cwd()}")
    print(f"Python path includes: {base_back_dir}")
    print("="*80 + "\n")
    
    # Execute the full pipeline
    results = run_full_pipeline()
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED")
    print("="*80)
    print(f"Results: {len(results)} backbones processed")
    print("="*80 + "\n")
