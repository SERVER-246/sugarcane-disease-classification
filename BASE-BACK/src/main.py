"""
Main entry point for Disease Classification Framework
Phase-1: Modular architecture with streamlined training pipeline
"""

import sys
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import json
import time

import numpy as np
import torch
import torch.nn as nn

# Debug configuration (conditional imports)
from config import settings

# Configuration
from config.settings import (
    BACKBONE_LR,
    BACKBONES,
    BATCH_SIZE,
    CKPT_DIR,
    DEPLOY_DIR,
    ENABLE_EXPORT,
    ENABLE_KFOLD_CV,
    EPOCHS_FINETUNE,
    EPOCHS_HEAD,
    HEAD_LR,
    IMG_SIZE,
    K_FOLDS,
    KFOLD_DIR,
    METRICS_DIR,
    NUM_CLASSES,
    PATIENCE_FT,
    PATIENCE_HEAD,
    RAW_DIR,
    SEED,
    SPLIT_DIR,
)
from sklearn.model_selection import StratifiedKFold


DEBUG_HEAD_EPOCHS = getattr(settings, 'DEBUG_EPOCHS_HEAD', EPOCHS_HEAD)
DEBUG_FT_EPOCHS = getattr(settings, 'DEBUG_EPOCHS_FINETUNE', EPOCHS_FINETUNE)
DEBUG_BATCH_SIZE = getattr(settings, 'DEBUG_BATCH_SIZE', BATCH_SIZE)

# Utils
# Models
from models import create_custom_backbone_safe  # noqa: E402

# Training
from training import (  # noqa: E402
    create_improved_scheduler,
    create_optimized_optimizer,
    get_loss_function_for_backbone,
    save_checkpoint,
    train_epoch_optimized,
    validate_epoch_optimized,
)
from utils import DEVICE, get_device_info, logger, set_seed  # noqa: E402
from utils.checkpoint_manager import CheckpointManager  # noqa: E402
from utils.datasets import (  # noqa: E402
    OptimizedTempDataset,
    WindowsCompatibleImageFolder,
    create_optimized_dataloader,
    create_optimized_transforms,
    prepare_datasets_for_backbone,
    prepare_optimized_datasets,
    verify_dataset_split,
)


# =============================================================================
# CORE TRAINING ORCHESTRATOR
# =============================================================================

def train_backbone_with_metrics(backbone_name, model, train_ds, val_ds,
                                epochs_head=EPOCHS_HEAD, epochs_finetune=EPOCHS_FINETUNE):
    """
    Complete training pipeline for a backbone with detailed logging
    Includes head training, fine-tuning, and comprehensive metrics tracking
    """

    logger.info(f"Starting training for {backbone_name}")

    if len(train_ds) == 0:
        raise ValueError(f"Training dataset is empty for {backbone_name}")
    if len(val_ds) == 0:
        raise ValueError(f"Validation dataset is empty for {backbone_name}")

    model.to(DEVICE)

    train_loader = create_optimized_dataloader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = create_optimized_dataloader(val_ds, BATCH_SIZE, shuffle=False)

    if hasattr(train_ds, 'classes'):
        class_names = train_ds.classes
    elif hasattr(train_ds, 'dataset') and hasattr(train_ds.dataset, 'classes'):
        class_names = train_ds.dataset.classes
    else:
        class_names = [f'Class_{i}' for i in range(NUM_CLASSES)]

    criterion = get_loss_function_for_backbone(backbone_name, NUM_CLASSES)
    best_acc = 0.0
    history = {'head': [], 'finetune': []}
    best_model_state = None
    patience_counter = 0

    # Stage 1: HEAD TRAINING
    logger.info(f"Stage 1: Head training for {backbone_name} (Epochs: {epochs_head})")

    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = False
        if hasattr(model.backbone, 'eval'):
            model.backbone.eval()

    optimizer = create_optimized_optimizer(model, lr=HEAD_LR, backbone_name=backbone_name)
    scheduler = create_improved_scheduler(optimizer, epochs_head, len(train_loader), backbone_name)

    for epoch in range(epochs_head):
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc, train_prec, train_rec, train_f1, _, _, _ = train_epoch_optimized(
            model, train_loader, optimizer, criterion
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = validate_epoch_optimized(
            model, val_loader, criterion
        )

        scheduler.step()

        # Check for NaN/Inf in metrics
        if np.isnan(train_loss) or np.isinf(train_loss) or np.isnan(val_loss) or np.isinf(val_loss):
            logger.error(f"NaN/Inf detected in epoch {epoch+1} metrics. Stopping training.")
            logger.error(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.error("This usually indicates gradient explosion. Model will use best checkpoint.")
            break

        history['head'].append((train_loss, val_loss, val_acc, train_acc, val_f1))

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            ckpt_path = CKPT_DIR / f"{backbone_name}_head_best.pth"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, extra={
                'epoch': epoch,
                'accuracy': val_acc,
                'stage': 'head'
            })
        else:
            patience_counter += 1

        logger.info(f"HEAD Epoch {epoch+1:2d}/{epochs_head} | "
                   f"LR: {current_lr:.2e} | "
                   f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Prec: {train_prec:.4f} F1: {train_f1:.4f} | "
                   f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Prec: {val_prec:.4f} F1: {val_f1:.4f} | "
                   f"Best: {best_acc:.4f} | Patience: {patience_counter}/{PATIENCE_HEAD}")

        if patience_counter >= PATIENCE_HEAD:
            logger.info(f"Early stopping triggered for head training at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best head model with accuracy: {best_acc:.4f}")

    # Stage 2: FINE-TUNING
    logger.info(f"Stage 2: Fine-tuning for {backbone_name} (Epochs: {epochs_finetune})")

    for param in model.parameters():
        param.requires_grad = True
    model.train()

    optimizer = create_optimized_optimizer(model, lr=BACKBONE_LR, backbone_name=backbone_name)
    scheduler = create_improved_scheduler(optimizer, epochs_finetune, len(train_loader), backbone_name)

    patience_counter = 0

    for epoch in range(epochs_finetune):
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc, train_prec, train_rec, train_f1, _, _, _ = train_epoch_optimized(
            model, train_loader, optimizer, criterion
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = validate_epoch_optimized(
            model, val_loader, criterion
        )

        scheduler.step()

        # Check for NaN/Inf in metrics
        if np.isnan(train_loss) or np.isinf(train_loss) or np.isnan(val_loss) or np.isinf(val_loss):
            logger.error(f"NaN/Inf detected in epoch {epoch+1} metrics. Stopping fine-tuning.")
            logger.error(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.error("Model will use best checkpoint from before NaN occurred.")
            break

        history['finetune'].append((train_loss, val_loss, val_acc, train_acc, val_f1))

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            ckpt_path = CKPT_DIR / f"{backbone_name}_finetune_best.pth"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, extra={
                'epoch': epoch,
                'accuracy': val_acc,
                'stage': 'finetune'
            })
        else:
            patience_counter += 1

        logger.info(f"FINE Epoch {epoch+1:2d}/{epochs_finetune} | "
                   f"LR: {current_lr:.2e} | "
                   f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Prec: {train_prec:.4f} F1: {train_f1:.4f} | "
                   f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Prec: {val_prec:.4f} F1: {val_f1:.4f} | "
                   f"Best: {best_acc:.4f} | Patience: {patience_counter}/{PATIENCE_FT}")

        if patience_counter >= PATIENCE_FT:
            logger.info(f"Early stopping triggered for fine-tuning at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best fine-tuned model with accuracy: {best_acc:.4f}")

    # Stage 3: FINAL EVALUATION
    logger.info(f"Stage 3: Final evaluation for {backbone_name}")

    model.eval()
    final_val_loss, final_val_acc, final_val_prec, final_val_rec, final_val_f1, val_preds, val_labels, val_probs = validate_epoch_optimized(
        model, val_loader, criterion
    )

    metrics = {
        'best_accuracy': float(best_acc),
        'final_accuracy': float(final_val_acc),
        'final_precision': float(final_val_prec),
        'final_recall': float(final_val_rec),
        'final_f1_score': float(final_val_f1),
        'final_val_loss': float(final_val_loss),
        'backbone_name': backbone_name,
        'epochs_head': len(history['head']),
        'epochs_finetune': len(history['finetune'])
    }

    # Generate visualizations
    logger.info(f"Generating comprehensive visualizations for {backbone_name}...")

    try:
        from utils.visualization import generate_all_visualizations, save_visualization_summary

        plot_paths = generate_all_visualizations(
            model=model,
            backbone_name=backbone_name,
            history=history,
            val_loader=val_loader,
            class_names=class_names,
            criterion=criterion,
            device=DEVICE
        )

        viz_summary_path = save_visualization_summary(plot_paths, backbone_name, METRICS_DIR)

        metrics['visualization_paths'] = plot_paths
        metrics['visualization_summary'] = viz_summary_path

        logger.info(f"[OK] Generated {len(plot_paths)} visualizations for {backbone_name}")

    except Exception as e:
        logger.error(f"x Visualization generation failed for {backbone_name}: {e}")
        logger.exception("Full traceback:")
        metrics['visualization_paths'] = {}
        metrics['visualization_summary'] = None

    final_ckpt_path = CKPT_DIR / f"{backbone_name}_final.pth"
    save_checkpoint(final_ckpt_path, model, extra=metrics)

    metrics_file = METRICS_DIR / f"{backbone_name}_training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"[OK] {backbone_name} training completed: Best accuracy = {best_acc:.4f}")
    logger.info(f"  Final checkpoint: {final_ckpt_path}")
    logger.info(f"  Metrics saved: {metrics_file}")

    return model, best_acc, history, metrics

def k_fold_cross_validation(backbone_name, full_dataset, k_folds=K_FOLDS):
    """Perform K-fold cross validation"""
    logger.info(f"Starting {k_folds}-fold CV for {backbone_name}")

    if hasattr(full_dataset, 'samples'):
        samples = full_dataset.samples
        labels = [s[1] for s in samples]
    else:
        samples = [(full_dataset[i][0], full_dataset[i][1]) for i in range(len(full_dataset))]
        labels = [s[1] for s in samples]

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(samples, labels)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {backbone_name} - Fold {fold + 1}/{k_folds}")
        logger.info(f"{'='*60}")

        try:
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            class_names = full_dataset.classes if hasattr(full_dataset, 'classes') else [f'Class_{i}' for i in range(NUM_CLASSES)]

            fold_train_ds = OptimizedTempDataset(
                train_samples, class_names,
                transform=create_optimized_transforms(IMG_SIZE, is_training=True)
            )
            fold_val_ds = OptimizedTempDataset(
                val_samples, class_names,
                transform=create_optimized_transforms(IMG_SIZE, is_training=False)
            )

            model = create_custom_backbone_safe(backbone_name, NUM_CLASSES)
            model.to(DEVICE)

            train_loader = create_optimized_dataloader(fold_train_ds, BATCH_SIZE, shuffle=True)
            val_loader = create_optimized_dataloader(fold_val_ds, BATCH_SIZE, shuffle=False)

            criterion = nn.CrossEntropyLoss()
            best_fold_acc = 0.0

            # Reduced epochs for K-fold
            head_epochs = min(30, EPOCHS_HEAD)
            finetune_epochs = min(20, EPOCHS_FINETUNE)

            # Stage 1: Head training (frozen backbone)
            logger.info(f"Fold {fold+1} - Stage 1: Head training")

            # Freeze backbone parameters
            if hasattr(model, 'backbone'):
                for param in model.backbone.parameters():
                    param.requires_grad = False

            optimizer = create_optimized_optimizer(model, lr=HEAD_LR, backbone_name=backbone_name)

            for epoch in range(head_epochs):
                current_lr = optimizer.param_groups[0]['lr']

                train_loss, train_acc, train_prec, train_rec, train_f1, _, _, _ = train_epoch_optimized(
                    model, train_loader, optimizer, criterion
                )
                val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = validate_epoch_optimized(
                    model, val_loader, criterion
                )

                # Check for NaN/Inf - break out of head training if detected
                if np.isnan(train_loss) or np.isinf(train_loss) or np.isnan(val_loss) or np.isinf(val_loss):
                    logger.error(f"NaN/Inf detected in fold {fold+1} epoch {epoch+1}. Skipping to next fold.")
                    best_fold_acc = 0.0  # Mark this fold as failed
                    break

                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc

                # Log every 5 epochs
                if epoch % 5 == 0 or epoch == head_epochs - 1:
                    logger.info(f"K-fold HEAD Epoch {epoch+1:2d}/{head_epochs} | "
                               f"LR: {current_lr:.2e} | "
                               f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                               f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

            # Stage 2: Fine-tuning (unfreeze all parameters)
            logger.info(f"Fold {fold+1} - Stage 2: Fine-tuning")

            # Unfreeze all parameters
            for param in model.parameters():
                param.requires_grad = True

            optimizer = create_optimized_optimizer(model, lr=BACKBONE_LR, backbone_name=backbone_name)

            for epoch in range(finetune_epochs):
                current_lr = optimizer.param_groups[0]['lr']

                train_loss, train_acc, train_prec, train_rec, train_f1, _, _, _ = train_epoch_optimized(
                    model, train_loader, optimizer, criterion
                )
                val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = validate_epoch_optimized(
                    model, val_loader, criterion
                )

                # Check for NaN/Inf - break out of fine-tuning if detected
                if np.isnan(train_loss) or np.isinf(train_loss) or np.isnan(val_loss) or np.isinf(val_loss):
                    logger.error(f"NaN/Inf detected in fold {fold+1} fine-tuning epoch {epoch+1}.")
                    break

                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc

                # Log every 3 epochs
                if epoch % 3 == 0 or epoch == finetune_epochs - 1:
                    logger.info(f"K-fold FINE Epoch {epoch+1:2d}/{finetune_epochs} | "
                               f"LR: {current_lr:.2e} | "
                               f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                               f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

            fold_results.append(best_fold_acc)
            logger.info(f"[OK] Fold {fold+1} completed: Best accuracy = {best_fold_acc:.4f}")

            del model, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        except Exception as e:
            logger.error(f"x Fold {fold+1} failed: {e}")
            fold_results.append(0.0)

    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)

    kfold_summary = {
        'backbone': backbone_name,
        'k_folds': k_folds,
        'fold_accuracies': [float(acc) for acc in fold_results],
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc)
    }

    kfold_file = KFOLD_DIR / f"{backbone_name}_kfold_results.json"
    with open(kfold_file, 'w') as f:
        json.dump(kfold_summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"K-fold CV completed for {backbone_name}")
    logger.info(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"Results saved: {kfold_file}")
    logger.info(f"{'='*60}\n")

    return mean_acc, std_acc, kfold_summary

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline():
    """
    Complete training pipeline matching original Base_backbones.py workflow:
    - Stage 0: Unit tests + Dataset preparation + Model verification
    - Stage 1: K-fold Cross Validation (if enabled)
    - Stage 2: Final model training
    - Stage 3: Test set evaluation
    - Stage 4: Model export (if enabled)
    """
    time.time()
    logger.log_system_info_once()
    logger.info(f"Device info: {json.dumps(get_device_info(), indent=2, default=str)}")

    logger.info("="*80)
    logger.info("DISEASE CLASSIFICATION BACKBONE TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Backbones to train: {len(BACKBONES)}")

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager()

    # Check for recovery
    recovery_status = checkpoint_mgr.get_recovery_status()
    if recovery_status['recovery_available']:
        logger.info("\n" + "="*80)
        logger.info("RECOVERY DETECTED - RESUMING FROM CHECKPOINT")
        logger.info("="*80)
        logger.info(f"Completed: {recovery_status['completed']} / {recovery_status['total_backbones']}")
        logger.info(f"Next backbone: {recovery_status['next_backbone']}")
        logger.info(f"Remaining: {recovery_status['remaining']}")

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    Path.cwd() / f'pipeline_run_{timestamp}.log'

    set_seed(SEED)

    # STAGE 0.1: Unit Tests
    logger.info("\n" + "="*80)
    logger.info("STAGE 0.1: RUNNING UNIT TESTS")
    logger.info("="*80)
    try:
        from tests.test_suite import run_all_unit_tests
        passed, failed = run_all_unit_tests()
        logger.info(f"[PASS] Unit Tests: {passed} passed, {failed} failed")
        if failed > 0:
            logger.warning(f"[WARN] {failed} unit tests failed, but continuing pipeline...")
    except ImportError:
        logger.warning("[WARN] Unit test module not found, skipping tests")
    except Exception as e:
        logger.warning(f"[WARN] Unit tests failed to run: {e}")

    # Stage 0.2: Dataset preparation
    logger.info("\n" + "="*80)
    logger.info("STAGE 0.2: DATASET PREPARATION")
    logger.info("="*80)

    try:
        prepare_optimized_datasets()
        verify_dataset_split()
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return {}

    # Stage 1: Model verification (only for unverified backbones)
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: MODEL VERIFICATION")
    logger.info("="*80)

    verification_results = {}
    for i, backbone_name in enumerate(BACKBONES):
        logger.info(f"\nVerifying {i+1}/{len(BACKBONES)}: {backbone_name}")
        try:
            model = create_custom_backbone_safe(backbone_name, NUM_CLASSES)
            verification_results[backbone_name] = {
                'status': 'verified',
                'num_params': sum(p.numel() for p in model.parameters()),
            }
            logger.info(f"[OK] {backbone_name}: verified")
            del model
        except Exception as e:
            logger.error(f"[FAIL] {backbone_name}: {e}")
            verification_results[backbone_name] = {'status': 'failed', 'error': str(e)}

    verified_backbones = [name for name, result in verification_results.items()
                         if result['status'] == 'verified']
    logger.info(f"\nVerified {len(verified_backbones)}/{len(BACKBONES)} models")

    # Stage 2: Training with checkpoint recovery
    logger.info("\n" + "="*80)
    logger.info("STAGE 2: MODEL TRAINING (with K-fold CV and Export)")
    logger.info("="*80)

    # Load full dataset for K-fold
    raw_dir = RAW_DIR

    if not raw_dir.exists():
        logger.warning(f"Raw dataset not found at {raw_dir}, using split dataset only")
        full_dataset = None
    else:
        try:
            from torchvision import transforms as T
            full_dataset = WindowsCompatibleImageFolder(
                str(raw_dir),
                transform=T.ToTensor()
            )
            logger.info(f"Loaded full dataset: {len(full_dataset)} samples for K-fold CV")
        except Exception as e:
            logger.warning(f"Could not load full dataset: {e}")
            full_dataset = None

    results = {}
    pipeline_start = time.time()
    completed_backbones = checkpoint_mgr.get_completed_backbones()

    for i, backbone_name in enumerate(verified_backbones):
        # Skip already completed backbones
        if backbone_name in completed_backbones:
            logger.info(f"\n[SKIP] {backbone_name}: Already completed (skipping)")
            continue

        # Track per-model timing
        model_start_time = time.time()

        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING BACKBONE {i+1}/{len(verified_backbones)}: {backbone_name.upper()}")
        logger.info(f"Progress: {i}/{len(verified_backbones)} completed ({(i/len(verified_backbones)*100):.1f}%)")

        # Calculate and display estimated time remaining
        if i > 0:
            elapsed = time.time() - pipeline_start
            avg_time_per_model = elapsed / i
            remaining = avg_time_per_model * (len(verified_backbones) - i)
            logger.info(f"Estimated time remaining: {remaining/60:.1f} min")
        else:
            logger.info("Estimated time: Calculating...")

        logger.info(f"{'='*80}")

        try:
            # STAGE 2.1: K-fold Cross Validation (if enabled)
            if ENABLE_KFOLD_CV and full_dataset is not None:
                logger.info(f"\nStage 2.1: K-fold Cross Validation for {backbone_name}")
                mean_acc, std_acc, kfold_summary = k_fold_cross_validation(
                    backbone_name, full_dataset, k_folds=K_FOLDS
                )
                logger.info(f"K-fold CV Results: {mean_acc:.4f} ± {std_acc:.4f}")
            else:
                logger.info("\nStage 2.1: K-fold Cross Validation SKIPPED")
                mean_acc, std_acc, _kfold_summary = 0.0, 0.0, {'skipped': True}

            # STAGE 2.2: Train final model
            logger.info(f"\nStage 2.2: Training final {backbone_name} model")

            train_ds, val_ds = prepare_datasets_for_backbone(backbone_name, IMG_SIZE)
            model = create_custom_backbone_safe(backbone_name, NUM_CLASSES)

            final_model, final_acc, history, metrics = train_backbone_with_metrics(
                backbone_name, model, train_ds, val_ds,
                epochs_head=EPOCHS_HEAD,
                epochs_finetune=EPOCHS_FINETUNE
            )

            # STAGE 2.3: Test set evaluation
            logger.info(f"\nStage 2.3: Test set evaluation for {backbone_name}")
            test_dir = SPLIT_DIR / 'test'

            if test_dir.exists():
                test_ds = WindowsCompatibleImageFolder(
                    str(test_dir),
                    transform=create_optimized_transforms(IMG_SIZE, is_training=False)
                )
                test_loader = create_optimized_dataloader(test_ds, BATCH_SIZE, shuffle=False)

                test_loss, test_acc, test_prec, test_rec, test_f1, _, _, _ = validate_epoch_optimized(
                    final_model, test_loader, nn.CrossEntropyLoss(), device=DEVICE
                )

                logger.info("Test Results:")
                logger.info(f"  Accuracy:  {test_acc:.4f}")
                logger.info(f"  Precision: {test_prec:.4f}")
                logger.info(f"  Recall:    {test_rec:.4f}")
                logger.info(f"  F1 Score:  {test_f1:.4f}")

                metrics['test_accuracy'] = float(test_acc)
                metrics['test_precision'] = float(test_prec)
                metrics['test_recall'] = float(test_rec)
                metrics['test_f1'] = float(test_f1)
            else:
                logger.warning(f"Test directory not found: {test_dir}")

            # STAGE 2.4: Model Export (if enabled)
            if ENABLE_EXPORT:
                logger.info(f"\nStage 2.4: Exporting {backbone_name} model")
                try:
                    from export.export_engine import export_model

                    deploy_dir = DEPLOY_DIR
                    model_dir = deploy_dir / backbone_name
                    model_dir.mkdir(parents=True, exist_ok=True)

                    # Prepare training metadata for export
                    export_metadata = {
                        'best_accuracy': float(final_acc),
                        'epochs_head': EPOCHS_HEAD,
                        'epochs_finetune': EPOCHS_FINETUNE,
                        'img_size': IMG_SIZE,
                        'batch_size': BATCH_SIZE,
                        'num_classes': NUM_CLASSES or len(train_ds.classes) if hasattr(train_ds, 'classes') else 13
                    }
                    if 'test_accuracy' in metrics:
                        export_metadata['test_accuracy'] = metrics['test_accuracy']
                        export_metadata['test_precision'] = metrics.get('test_precision', 0.0)
                        export_metadata['test_recall'] = metrics.get('test_recall', 0.0)
                        export_metadata['test_f1'] = metrics.get('test_f1', 0.0)

                    # Get class names from dataset
                    if hasattr(train_ds, 'classes'):
                        export_class_names = train_ds.classes
                    elif hasattr(train_ds, 'dataset') and hasattr(train_ds.dataset, 'classes'):
                        export_class_names = train_ds.dataset.classes
                    else:
                        export_class_names = [f'class_{i}' for i in range(NUM_CLASSES or 13)]

                    # Export to multiple formats
                    export_results = export_model(
                        model=final_model,
                        model_name=backbone_name,
                        output_dir=model_dir,
                        input_shape=(1, 3, IMG_SIZE, IMG_SIZE),
                        formats=['pytorch', 'onnx', 'torchscript'],
                        class_names=export_class_names,
                        training_metadata=export_metadata
                    )

                    logger.info(f"[OK] Export completed: {len(export_results)} formats")
                    for fmt, path in export_results.items():
                        logger.info(f"  - {fmt}: {path}")

                    metrics['exported_formats'] = list(export_results.keys())
                except Exception as e:
                    logger.warning(f"Export failed for {backbone_name}: {e}")
                    metrics['export_error'] = str(e)
            else:
                logger.info("\nStage 2.4: Model Export SKIPPED (disabled)")

            # Calculate model training time
            model_time = time.time() - model_start_time
            logger.info(f"\n[OK] {backbone_name} completed in {model_time/60:.1f} min")

            # Save checkpoint after successful training
            checkpoint_mgr.mark_backbone_complete(backbone_name, {
                'accuracy': final_acc,
                'history': history,
                'metrics': metrics,
                'kfold_cv': {'mean': mean_acc, 'std': std_acc} if ENABLE_KFOLD_CV else None,
                'training_time_seconds': model_time
            })

            results[backbone_name] = {
                'verification': verification_results[backbone_name],
                'final_accuracy': final_acc,
                'final_metrics': metrics,
                'kfold_results': {'mean': mean_acc, 'std': std_acc},
                'training_time': model_time,
                'status': 'success'
            }

            logger.info(f"[OK] {backbone_name} completed: {final_acc:.4f} accuracy")

            del model, final_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        except Exception as e:
            logger.error(f"x {backbone_name} failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            results[backbone_name] = {
                'verification': verification_results[backbone_name],
                'error': str(e),
                'status': 'failed'
            }

    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start

    # Final summary - matching original Base_backbones.py format
    logger.info("\n" + "="*80)
    logger.info("FINAL PIPELINE SUMMARY")
    logger.info("="*80)
    logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total execution time: {total_time/3600:.2f} hours ({total_time/60:.1f} min)")

    successful = sum(1 for r in results.values() if r.get('status') == 'success')
    failed = len(verified_backbones) - successful
    logger.info("\nTraining Results:")
    logger.info(f"  [OK] Successfully trained: {successful}/{len(verified_backbones)} models")
    logger.info(f"  x Failed: {failed} models")

    # Per-backbone summary
    logger.info("\nPer-Backbone Results:")
    for backbone_name, result in results.items():
        if result.get('status') == 'success':
            acc = result.get('final_accuracy', 0.0)
            kfold = result.get('kfold_results', {})
            time_min = result.get('training_time', 0) / 60
            logger.info(f"  {backbone_name:25s} | Acc: {acc:.4f} | K-fold: {kfold.get('mean', 0):.4f}±{kfold.get('std', 0):.4f} | Time: {time_min:.1f}min")
        else:
            error_msg = result.get('error', 'Unknown error')[:50]
            logger.info(f"  {backbone_name:25s} | FAILED: {error_msg}")

    # Export recovery status
    final_status = checkpoint_mgr.get_recovery_status()
    logger.info("\nCheckpoint Recovery Status:")
    logger.info(f"  Completed: {final_status['completed']} / {final_status['total_backbones']}")
    logger.info(f"  Remaining: {final_status['remaining']}")

    # Save recovery summary
    checkpoint_mgr.export_recovery_summary()

    # Save results
    summary_file = METRICS_DIR / 'pipeline_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'verification_results': verification_results,
            'training_results': results,
            'total_execution_time_seconds': total_time,
            'final_checkpoint_status': final_status,
            'timestamp': timestamp,
            'statistics': {
                'successful': successful,
                'failed': failed,
                'total_backbones': len(verified_backbones)
            }
        }, f, indent=2, default=str)

    logger.info("\n" + "="*80)
    logger.info(f"[DONE] Pipeline completed! Results saved to: {summary_file}")
    logger.info(f"[DONE] Recovery status saved to: {checkpoint_mgr.checkpoint_dir / 'recovery_status.txt'}")
    logger.info("="*80)

    return results

if __name__ == '__main__':
    run_full_pipeline()
