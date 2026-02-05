"""
Complete 15-Coin Ensemble Pipeline
Implements all 7 stages of the ensemble roadmap
Mirrors run_pipeline.py structure with full recovery support
"""

import json
import sys
import time
from pathlib import Path
from typing import Any


# Add BASE-BACK to path
base_back_dir = Path(__file__).parent.parent / 'BASE-BACK'
sys.path.insert(0, str(base_back_dir / 'src'))

from config.settings import (
    BACKBONES,
    BASE_DIR,
    BATCH_SIZE,
    CKPT_DIR,
    DEPLOY_DIR,
    IMG_SIZE,
    METRICS_DIR,
    PLOTS_DIR,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)
from ensemble_checkpoint_manager import EnsembleCheckpointManager

# Import stages
from stage1_individual import extract_all_predictions_and_embeddings
from stage2_score_ensembles import train_all_score_ensembles
from stage3_stacking import train_all_stacking_models
from stage4_feature_fusion import train_all_fusion_models
from stage5_mixture_experts import train_mixture_of_experts
from stage6_meta_ensemble import train_meta_ensemble_controller
from stage7_distillation import train_distilled_student
from utils import get_device_info, logger, set_seed
from utils.datasets import WindowsCompatibleImageFolder, create_optimized_dataloader, create_optimized_transforms


# Ensemble directories
ENSEMBLE_DIR = BASE_DIR / 'ensembles'
ENSEMBLE_METRICS_DIR = METRICS_DIR / 'ensembles'
ENSEMBLE_PLOTS_DIR = PLOTS_DIR / 'ensembles'
ENSEMBLE_CKPT_DIR = CKPT_DIR / 'ensembles'
ENSEMBLE_DEPLOY_DIR = DEPLOY_DIR / 'ensembles'

# Stage-specific directories
STAGE1_DIR = ENSEMBLE_DIR / 'stage1_individual'
STAGE2_DIR = ENSEMBLE_DIR / 'stage2_score_ensembles'
STAGE3_DIR = ENSEMBLE_DIR / 'stage3_stacking'
STAGE4_DIR = ENSEMBLE_DIR / 'stage4_fusion'
STAGE5_DIR = ENSEMBLE_DIR / 'stage5_moe'
STAGE6_DIR = ENSEMBLE_DIR / 'stage6_meta'
STAGE7_DIR = ENSEMBLE_DIR / 'stage7_distillation'

for d in [ENSEMBLE_DIR, ENSEMBLE_METRICS_DIR, ENSEMBLE_PLOTS_DIR,
          ENSEMBLE_CKPT_DIR, ENSEMBLE_DEPLOY_DIR, STAGE1_DIR, STAGE2_DIR,
          STAGE3_DIR, STAGE4_DIR, STAGE5_DIR, STAGE6_DIR, STAGE7_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def run_complete_15coin_pipeline() -> dict[str, Any]:
    """
    Complete 15-Coin Ensemble Pipeline
    
    Stage 1: Individual Models (Already trained - extract predictions/embeddings)
    Stage 2: Score-Level Ensembles (soft, hard, weighted, logit voting)
    Stage 3: Stacking (LR, XGBoost, MLP meta-learners with OOF)
    Stage 4: Feature Fusion (concat+MLP, attention, bilinear)
    Stage 5: Mixture of Experts (gating network)
    Stage 6: Meta-Ensemble (combine all ensemble types)
    Stage 7: Knowledge Distillation (compress to single model)
    """

    pipeline_start = time.time()

    logger.log_system_info_once()
    logger.info(f"Device info: {json.dumps(get_device_info(), indent=2, default=str)}")

    logger.info("="*80)
    logger.info("15-COIN ENSEMBLE PIPELINE - COMPLETE 7-STAGE SYSTEM")
    logger.info("="*80)
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"All 15 backbones: {BACKBONES}")

    set_seed(SEED)

    # Initialize checkpoint manager
    checkpoint_mgr = EnsembleCheckpointManager()

    # Check for recovery
    recovery_status = checkpoint_mgr.get_recovery_status()
    if recovery_status['recovery_available']:
        logger.info("\n" + "="*80)
        logger.info("RECOVERY DETECTED - RESUMING FROM CHECKPOINT")
        logger.info("="*80)
        logger.info(f"Completed stages: {recovery_status.get('completed_stages', [])}")

    results = {}
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # Prepare datasets (used across all stages)
    logger.info("\n" + "="*80)
    logger.info("PREPARING DATASETS")
    logger.info("="*80)

    train_ds = WindowsCompatibleImageFolder(
        str(TRAIN_DIR),
        transform=create_optimized_transforms(IMG_SIZE, is_training=False)
    )
    val_ds = WindowsCompatibleImageFolder(
        str(VAL_DIR),
        transform=create_optimized_transforms(IMG_SIZE, is_training=False)
    )
    test_ds = WindowsCompatibleImageFolder(
        str(TEST_DIR),
        transform=create_optimized_transforms(IMG_SIZE, is_training=False)
    )

    # Use num_workers=0 for Windows multiprocessing compatibility
    train_loader = create_optimized_dataloader(train_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = create_optimized_dataloader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = create_optimized_dataloader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    logger.info(f"Train: {len(train_ds)} samples")
    logger.info(f"Val: {len(val_ds)} samples")
    logger.info(f"Test: {len(test_ds)} samples")
    logger.info(f"Classes: {train_ds.classes}")

    # =============================================================================
    # STAGE 1: INDIVIDUAL MODELS (Extract predictions & embeddings)
    # =============================================================================

    stage1_complete = checkpoint_mgr.is_stage_complete('stage1')

    if not stage1_complete:
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: INDIVIDUAL MODELS - EXTRACT PREDICTIONS & EMBEDDINGS")
        logger.info("="*80)

        try:
            stage1_results = extract_all_predictions_and_embeddings(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                output_dir=STAGE1_DIR
            )

            checkpoint_mgr.mark_stage_complete('stage1', stage1_results)
            results['stage1'] = stage1_results

            logger.info(f"[OK] Stage 1 completed: {stage1_results['num_backbones']} backbones")

        except Exception as e:
            logger.error(f"x Stage 1 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': f'Stage 1 failed: {e}'}
    else:
        logger.info("\n[SKIP] Stage 1 already completed")
        results['stage1'] = checkpoint_mgr.get_stage_results('stage1')

    # =============================================================================
    # STAGE 2: SCORE-LEVEL ENSEMBLES
    # =============================================================================

    stage2_complete = checkpoint_mgr.is_stage_complete('stage2')

    if not stage2_complete:
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: SCORE-LEVEL ENSEMBLES")
        logger.info("="*80)
        logger.info("Training: Soft Voting, Hard Voting, Weighted Voting, Logit Averaging")

        try:
            stage2_results = train_all_score_ensembles(
                stage1_dir=STAGE1_DIR,
                val_loader=val_loader,
                test_loader=test_loader,
                output_dir=STAGE2_DIR,
                class_names=train_ds.classes
            )

            checkpoint_mgr.mark_stage_complete('stage2', stage2_results)
            results['stage2'] = stage2_results

            logger.info(f"[OK] Stage 2 completed: {len(stage2_results['ensembles'])} ensembles")

        except Exception as e:
            logger.error(f"x Stage 2 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': f'Stage 2 failed: {e}'}
    else:
        logger.info("\n[SKIP] Stage 2 already completed")
        results['stage2'] = checkpoint_mgr.get_stage_results('stage2')

    # =============================================================================
    # STAGE 3: STACKING META-LEARNERS
    # =============================================================================

    stage3_complete = checkpoint_mgr.is_stage_complete('stage3')

    if not stage3_complete:
        logger.info("\n" + "="*80)
        logger.info("STAGE 3: STACKING META-LEARNERS")
        logger.info("="*80)
        logger.info("Training: Logistic Regression, XGBoost, MLP (with OOF)")

        try:
            stage3_results = train_all_stacking_models(
                stage1_dir=STAGE1_DIR,
                train_ds=train_ds,
                val_loader=val_loader,
                test_loader=test_loader,
                output_dir=STAGE3_DIR,
                class_names=train_ds.classes
            )

            checkpoint_mgr.mark_stage_complete('stage3', stage3_results)
            results['stage3'] = stage3_results

            logger.info(f"[OK] Stage 3 completed: {len(stage3_results['stackers'])} stackers")

        except Exception as e:
            logger.error(f"x Stage 3 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': f'Stage 3 failed: {e}'}
    else:
        logger.info("\n[SKIP] Stage 3 already completed")
        results['stage3'] = checkpoint_mgr.get_stage_results('stage3')

    # =============================================================================
    # STAGE 4: FEATURE-LEVEL FUSION
    # =============================================================================

    stage4_complete = checkpoint_mgr.is_stage_complete('stage4')

    if not stage4_complete:
        logger.info("\n" + "="*80)
        logger.info("STAGE 4: FEATURE-LEVEL FUSION")
        logger.info("="*80)
        logger.info("Training: Concat+MLP, Attention Fusion, Bilinear Pooling")

        try:
            stage4_results = train_all_fusion_models(
                stage1_dir=STAGE1_DIR,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                output_dir=STAGE4_DIR,
                class_names=train_ds.classes
            )

            checkpoint_mgr.mark_stage_complete('stage4', stage4_results)
            results['stage4'] = stage4_results

            logger.info(f"[OK] Stage 4 completed: {len(stage4_results['fusion_models'])} fusion models")

        except Exception as e:
            logger.error(f"x Stage 4 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': f'Stage 4 failed: {e}'}
    else:
        logger.info("\n[SKIP] Stage 4 already completed")
        results['stage4'] = checkpoint_mgr.get_stage_results('stage4')

    # =============================================================================
    # STAGE 5: MIXTURE OF EXPERTS
    # =============================================================================

    stage5_complete = checkpoint_mgr.is_stage_complete('stage5')

    if not stage5_complete:
        logger.info("\n" + "="*80)
        logger.info("STAGE 5: MIXTURE OF EXPERTS")
        logger.info("="*80)
        logger.info("Training: Gating Network with Top-K Routing")

        try:
            stage5_results = train_mixture_of_experts(
                stage1_dir=STAGE1_DIR,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                output_dir=STAGE5_DIR,
                class_names=train_ds.classes
            )

            checkpoint_mgr.mark_stage_complete('stage5', stage5_results)
            results['stage5'] = stage5_results

            logger.info(f"[OK] Stage 5 completed: MoE accuracy={stage5_results['test_accuracy']:.4f}")

        except Exception as e:
            logger.error(f"x Stage 5 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': f'Stage 5 failed: {e}'}
    else:
        logger.info("\n[SKIP] Stage 5 already completed")
        results['stage5'] = checkpoint_mgr.get_stage_results('stage5')

    # =============================================================================
    # STAGE 6: META-ENSEMBLE (Ensemble-of-Ensembles)
    # =============================================================================

    stage6_complete = checkpoint_mgr.is_stage_complete('stage6')

    if not stage6_complete:
        logger.info("\n" + "="*80)
        logger.info("STAGE 6: META-ENSEMBLE CONTROLLER")
        logger.info("="*80)
        logger.info("Combining: Score ensembles + Stackers + Fusion + MoE")

        try:
            stage6_results = train_meta_ensemble_controller(
                stage2_dir=STAGE2_DIR,
                stage3_dir=STAGE3_DIR,
                stage4_dir=STAGE4_DIR,
                stage5_dir=STAGE5_DIR,
                val_loader=val_loader,
                test_loader=test_loader,
                output_dir=STAGE6_DIR,
                class_names=train_ds.classes
            )

            checkpoint_mgr.mark_stage_complete('stage6', stage6_results)
            results['stage6'] = stage6_results

            logger.info(f"[OK] Stage 6 completed: Meta accuracy={stage6_results['test_accuracy']:.4f}")

        except Exception as e:
            logger.error(f"x Stage 6 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': f'Stage 6 failed: {e}'}
    else:
        logger.info("\n[SKIP] Stage 6 already completed")
        results['stage6'] = checkpoint_mgr.get_stage_results('stage6')

    # =============================================================================
    # STAGE 7: KNOWLEDGE DISTILLATION
    # =============================================================================

    stage7_complete = checkpoint_mgr.is_stage_complete('stage7')

    if not stage7_complete:
        logger.info("\n" + "="*80)
        logger.info("STAGE 7: KNOWLEDGE DISTILLATION")
        logger.info("="*80)
        logger.info("Distilling: Stage 6 teacher -> Compact student")

        # Create NEW DataLoaders with PROPER training settings for Stage 7
        # (Previous stages needed shuffle=False for consistent predictions)
        logger.info("Creating training DataLoaders with augmentation and shuffle...")

        train_ds_aug = WindowsCompatibleImageFolder(
            str(TRAIN_DIR),
            transform=create_optimized_transforms(IMG_SIZE, is_training=True)  # WITH augmentation
        )
        val_ds_clean = WindowsCompatibleImageFolder(
            str(VAL_DIR),
            transform=create_optimized_transforms(IMG_SIZE, is_training=False)
        )
        test_ds_clean = WindowsCompatibleImageFolder(
            str(TEST_DIR),
            transform=create_optimized_transforms(IMG_SIZE, is_training=False)
        )

        # Create loaders with shuffle=True for training
        train_loader_s7 = create_optimized_dataloader(train_ds_aug, BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader_s7 = create_optimized_dataloader(val_ds_clean, BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader_s7 = create_optimized_dataloader(test_ds_clean, BATCH_SIZE, shuffle=False, num_workers=0)

        logger.info(f"Stage 7 DataLoaders: Train={len(train_ds_aug)} (shuffle=True, augmented)")

        try:
            stage7_results = train_distilled_student(
                teacher_stage6_dir=STAGE6_DIR,
                train_loader=train_loader_s7,
                val_loader=val_loader_s7,
                test_loader=test_loader_s7,
                output_dir=STAGE7_DIR,
                class_names=train_ds.classes
            )

            checkpoint_mgr.mark_stage_complete('stage7', stage7_results)
            results['stage7'] = stage7_results

            logger.info(f"[OK] Stage 7 completed: Student accuracy={stage7_results['test_accuracy']:.4f}")

        except Exception as e:
            logger.error(f"x Stage 7 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': f'Stage 7 failed: {e}'}
    else:
        logger.info("\n[SKIP] Stage 7 already completed")
        results['stage7'] = checkpoint_mgr.get_stage_results('stage7')

    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================

    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start

    logger.info("\n" + "="*80)
    logger.info("15-COIN PIPELINE COMPLETE - FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total execution time: {total_time/3600:.2f} hours ({total_time/60:.1f} min)")

    logger.info("\nStage Results:")
    logger.info(f"  Stage 1 (Individual): {results['stage1']['num_backbones']} backbones")
    logger.info(f"  Stage 2 (Score): {len(results['stage2']['ensembles'])} ensembles, best={max([e['test_accuracy'] for e in results['stage2']['ensembles'].values()]):.4f}")
    logger.info(f"  Stage 3 (Stacking): {len(results['stage3']['stackers'])} stackers, best={max([s['test_accuracy'] for s in results['stage3']['stackers'].values()]):.4f}")
    logger.info(f"  Stage 4 (Fusion): {len(results['stage4']['fusion_models'])} models, best={max([f['test_accuracy'] for f in results['stage4']['fusion_models'].values()]):.4f}")
    logger.info(f"  Stage 5 (MoE): {results['stage5']['test_accuracy']:.4f}")
    logger.info(f"  Stage 6 (Meta): {results['stage6']['test_accuracy']:.4f}")
    logger.info(f"  Stage 7 (Distilled): {results['stage7']['test_accuracy']:.4f}")

    # Save final results
    final_results = {
        'timestamp': timestamp,
        'total_execution_time': total_time,
        'stage1_individual': results['stage1'],
        'stage2_score_ensembles': results['stage2'],
        'stage3_stacking': results['stage3'],
        'stage4_fusion': results['stage4'],
        'stage5_moe': results['stage5'],
        'stage6_meta': results['stage6'],
        'stage7_distillation': results['stage7']
    }

    results_file = ENSEMBLE_METRICS_DIR / f'15coin_complete_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    logger.info(f"\n[DONE] Complete results saved to: {results_file}")
    logger.info("="*80)

    return final_results


if __name__ == '__main__':
    results = run_complete_15coin_pipeline()

    if 'error' not in results:
        print("\n✅ 15-COIN PIPELINE COMPLETE!")
        print(f"Final distilled student accuracy: {results['stage7_distillation']['test_accuracy']:.4f}")
    else:
        print(f"\n❌ Pipeline failed: {results['error']}")
