"""
Quick Pipeline Test - Run with output logging
"""

import sys
from pathlib import Path


# Add BASE-BACK to path
base_back_dir = Path(__file__).parent.parent / 'BASE-BACK' / 'src'
sys.path.insert(0, str(base_back_dir))

if __name__ == '__main__':
    # Required for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()

    print("="*80)
    print("15-COIN ENSEMBLE PIPELINE TEST")
    print("="*80)

    # Import and run
    from run_15coin_pipeline import run_complete_15coin_pipeline

    try:
        results = run_complete_15coin_pipeline()
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nFinal Results:")
        for stage, result in results.items():
            print(f"  {stage}: {result}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
