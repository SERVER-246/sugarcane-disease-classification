"""
Quick verification script to check plotting integration
Verifies imports and function availability without running full pipeline
"""

import sys
from pathlib import Path

# Add paths
BASE_BACK_PATH = Path(__file__).parent.parent / 'BASE-BACK' / 'src'
if str(BASE_BACK_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_BACK_PATH))

def verify_plotting_imports():
    """Verify all plotting imports work correctly"""
    
    print("=" * 80)
    print("ENSEMBLE PLOTTING VERIFICATION")
    print("=" * 80)
    
    errors = []
    
    # Check ensemble_plots module
    print("\n[1/7] Checking ensemble_plots.py...")
    try:
        from ensemble_plots import (
            plot_confusion_matrix,
            plot_roc_curves,
            plot_per_class_metrics,
            plot_training_history,
            plot_ensemble_comparison,
            create_all_plots
        )
        print("  ✓ ensemble_plots.py imports successful")
        print("    - plot_confusion_matrix")
        print("    - plot_roc_curves")
        print("    - plot_per_class_metrics")
        print("    - plot_training_history")
        print("    - plot_ensemble_comparison")
        print("    - create_all_plots")
    except ImportError as e:
        errors.append(f"ensemble_plots.py: {e}")
        print(f"  ✗ ERROR: {e}")
    
    # Check Stage 2
    print("\n[2/7] Checking stage2_score_ensembles.py...")
    try:
        import stage2_score_ensembles
        print("  ✓ Stage 2 imports successful")
    except ImportError as e:
        errors.append(f"stage2_score_ensembles.py: {e}")
        print(f"  ✗ ERROR: {e}")
    
    # Check Stage 3
    print("\n[3/7] Checking stage3_stacking.py...")
    try:
        import stage3_stacking
        print("  ✓ Stage 3 imports successful")
    except ImportError as e:
        errors.append(f"stage3_stacking.py: {e}")
        print(f"  ✗ ERROR: {e}")
    
    # Check Stage 4
    print("\n[4/7] Checking stage4_feature_fusion.py...")
    try:
        import stage4_feature_fusion
        print("  ✓ Stage 4 imports successful")
    except ImportError as e:
        errors.append(f"stage4_feature_fusion.py: {e}")
        print(f"  ✗ ERROR: {e}")
    
    # Check Stage 5
    print("\n[5/7] Checking stage5_mixture_experts.py...")
    try:
        import stage5_mixture_experts
        print("  ✓ Stage 5 imports successful")
    except ImportError as e:
        errors.append(f"stage5_mixture_experts.py: {e}")
        print(f"  ✗ ERROR: {e}")
    
    # Check Stage 6
    print("\n[6/7] Checking stage6_meta_ensemble.py...")
    try:
        import stage6_meta_ensemble
        print("  ✓ Stage 6 imports successful")
    except ImportError as e:
        errors.append(f"stage6_meta_ensemble.py: {e}")
        print(f"  ✗ ERROR: {e}")
    
    # Check Stage 7
    print("\n[7/7] Checking stage7_distillation.py...")
    try:
        import stage7_distillation
        print("  ✓ Stage 7 imports successful")
    except ImportError as e:
        errors.append(f"stage7_distillation.py: {e}")
        print(f"  ✗ ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    if not errors:
        print("✓ ALL IMPORTS SUCCESSFUL - Plotting system ready!")
        print("\nNext steps:")
        print("  1. Run full pipeline: python test_pipeline.py")
        print("  2. Check plots in ensemble_results/ directory")
        print("  3. Verify plot quality (DPI 1200, proper formatting)")
    else:
        print("✗ ERRORS DETECTED:")
        for error in errors:
            print(f"  - {error}")
        return False
    print("=" * 80)
    
    return True

if __name__ == '__main__':
    success = verify_plotting_imports()
    sys.exit(0 if success else 1)
