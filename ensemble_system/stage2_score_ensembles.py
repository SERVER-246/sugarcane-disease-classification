"""
Stage 2: Score-Level Ensembles
Soft Voting, Hard Voting, Weighted Voting, Logit Averaging
All using predictions from Stage 1
"""

import sys
from pathlib import Path


# Add BASE-BACK to path BEFORE other imports
BASE_BACK_PATH = Path(__file__).parent.parent / 'BASE-BACK' / 'src'
if str(BASE_BACK_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_BACK_PATH))

import json
from typing import Any

import numpy as np
from config.settings import BACKBONES, NUM_CLASSES
from ensemble_plots import create_all_plots
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import logger


def load_stage1_predictions(stage1_dir: Path, split: str = 'val') -> dict[str, dict[str, np.ndarray]]:
    """Load predictions from Stage 1 for all backbones"""

    stage1_dir = Path(stage1_dir)
    predictions = {}

    for backbone_name in BACKBONES:
        pred_file = stage1_dir / f'{backbone_name}_{split}_predictions.npz'

        if not pred_file.exists():
            logger.warning(f"  x Missing {split} predictions for {backbone_name}")
            continue

        data = np.load(pred_file)
        predictions[backbone_name] = {
            'logits': data['logits'],
            'probabilities': data['probabilities'],
            'predictions': data['predictions'],
            'labels': data['labels']
        }

    logger.info(f"Loaded {split} predictions from {len(predictions)} backbones")
    return predictions


def soft_voting_ensemble(predictions: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    """
    Soft Voting: Average probabilities from all models
    
    Args:
        predictions: Dict of predictions from each backbone
        
    Returns:
        Final predictions (N,)
    """

    probs_list = [pred['probabilities'] for pred in predictions.values()]
    avg_probs = np.mean(probs_list, axis=0)  # (N, num_classes)
    final_preds = avg_probs.argmax(axis=1)

    return final_preds, avg_probs


def hard_voting_ensemble(predictions: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    """
    Hard Voting: Majority vote from all models
    
    Args:
        predictions: Dict of predictions from each backbone
        
    Returns:
        Final predictions (N,)
    """

    preds_list = [pred['predictions'] for pred in predictions.values()]
    preds_array = np.stack(preds_list, axis=0)  # (num_models, N)

    # Count votes for each class
    num_samples = preds_array.shape[1]
    final_preds = np.zeros(num_samples, dtype=int)

    for i in range(num_samples):
        votes = preds_array[:, i]
        # Get most common prediction
        counts = np.bincount(votes, minlength=NUM_CLASSES)
        final_preds[i] = counts.argmax()

    return final_preds


def weighted_voting_ensemble(
    predictions: dict[str, dict[str, np.ndarray]],
    weights: dict[str, float]
) -> np.ndarray:
    """
    Weighted Voting: Weighted average of probabilities
    
    Args:
        predictions: Dict of predictions from each backbone
        weights: Dict of weights for each backbone (e.g., validation accuracy)
        
    Returns:
        Final predictions (N,)
    """

    weighted_probs = None
    total_weight = 0.0

    for backbone_name, pred in predictions.items():
        weight = weights.get(backbone_name, 1.0)
        probs = pred['probabilities']

        if weighted_probs is None:
            weighted_probs = probs * weight
        else:
            weighted_probs += probs * weight

        total_weight += weight

    weighted_probs /= total_weight
    final_preds = weighted_probs.argmax(axis=1)

    return final_preds, weighted_probs


def logit_averaging_ensemble(predictions: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    """
    Logit Averaging: Average logits before softmax
    
    Args:
        predictions: Dict of predictions from each backbone
        
    Returns:
        Final predictions (N,)
    """

    logits_list = [pred['logits'] for pred in predictions.values()]
    avg_logits = np.mean(logits_list, axis=0)  # (N, num_classes)

    # Apply softmax
    exp_logits = np.exp(avg_logits - np.max(avg_logits, axis=1, keepdims=True))
    avg_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    final_preds = avg_probs.argmax(axis=1)

    return final_preds, avg_probs


def evaluate_ensemble(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Calculate metrics for ensemble predictions"""

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }


def train_all_score_ensembles(
    stage1_dir: Path,
    val_loader,
    test_loader,
    output_dir: Path,
    class_names: list[str],
    train_loader=None
) -> dict[str, Any]:
    """
    Train all score-level ensemble methods
    
    Returns:
        Dictionary with results for each ensemble type
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("Training Score-Level Ensembles")
    logger.info("="*80)

    # Load predictions from Stage 1
    train_preds = load_stage1_predictions(stage1_dir, 'train')  # For Stage 6 meta-learning
    val_preds = load_stage1_predictions(stage1_dir, 'val')
    test_preds = load_stage1_predictions(stage1_dir, 'test')

    # Get labels (same for all backbones)
    train_labels = list(train_preds.values())[0]['labels']
    val_labels = list(val_preds.values())[0]['labels']
    test_labels = list(test_preds.values())[0]['labels']

    # Load validation accuracies for weighted voting
    catalog_file = stage1_dir / 'catalog.json'
    with open(catalog_file) as f:
        catalog = json.load(f)

    weights = {name: info['val_accuracy'] for name, info in catalog['backbones'].items()
               if info.get('status') == 'success'}

    results = {}

    # 1. Soft Voting
    logger.info("\n1. Soft Voting Ensemble")
    soft_train_preds, soft_train_probs = soft_voting_ensemble(train_preds)
    soft_val_preds, soft_val_probs = soft_voting_ensemble(val_preds)
    soft_test_preds, soft_test_probs = soft_voting_ensemble(test_preds)

    soft_val_metrics = evaluate_ensemble(soft_val_preds, val_labels)
    soft_test_metrics = evaluate_ensemble(soft_test_preds, test_labels)

    logger.info(f"  Val Acc: {soft_val_metrics['accuracy']:.4f}")
    logger.info(f"  Test Acc: {soft_test_metrics['accuracy']:.4f}")

    # Save
    soft_dir = output_dir / 'soft_voting'
    soft_dir.mkdir(exist_ok=True)

    # Save as npz for backward compatibility
    np.savez_compressed(
        soft_dir / 'predictions.npz',
        train_predictions=soft_train_preds,
        train_probabilities=soft_train_probs,
        val_predictions=soft_val_preds,
        val_probabilities=soft_val_probs,
        test_predictions=soft_test_preds,
        test_probabilities=soft_test_probs,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels
    )

    # Also save individual .npy files for Stage 6 compatibility
    np.save(soft_dir / 'train_predictions.npy', soft_train_probs)
    np.save(soft_dir / 'val_predictions.npy', soft_val_probs)
    np.save(soft_dir / 'test_predictions.npy', soft_test_probs)

    with open(soft_dir / 'metrics.json', 'w') as f:
        json.dump({
            'val_metrics': soft_val_metrics,
            'test_metrics': soft_test_metrics,
            'ensemble_type': 'soft_voting',
            'num_backbones': len(val_preds)
        }, f, indent=2)

    # Generate plots
    create_all_plots(
        y_true=test_labels,
        y_pred=soft_test_preds,
        y_probs=soft_test_probs,
        class_names=class_names,
        output_dir=soft_dir,
        prefix='soft_voting'
    )

    results['soft_voting'] = {
        'val_metrics': soft_val_metrics,
        'test_metrics': soft_test_metrics,
        'test_accuracy': soft_test_metrics['accuracy'],
        'output_dir': str(soft_dir)
    }

    # 2. Hard Voting
    logger.info("\n2. Hard Voting Ensemble")
    hard_train_preds = hard_voting_ensemble(train_preds)
    hard_val_preds = hard_voting_ensemble(val_preds)
    hard_test_preds = hard_voting_ensemble(test_preds)

    hard_val_metrics = evaluate_ensemble(hard_val_preds, val_labels)
    hard_test_metrics = evaluate_ensemble(hard_test_preds, test_labels)

    logger.info(f"  Val Acc: {hard_val_metrics['accuracy']:.4f}")
    logger.info(f"  Test Acc: {hard_test_metrics['accuracy']:.4f}")

    # Save
    hard_dir = output_dir / 'hard_voting'
    hard_dir.mkdir(exist_ok=True)

    np.savez_compressed(
        hard_dir / 'predictions.npz',
        train_predictions=hard_train_preds,
        val_predictions=hard_val_preds,
        test_predictions=hard_test_preds,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels
    )

    # Also save individual .npy files - use soft probs for hard voting (for meta-learning)
    np.save(hard_dir / 'train_predictions.npy', soft_train_probs)
    np.save(hard_dir / 'val_predictions.npy', soft_val_probs)
    np.save(hard_dir / 'test_predictions.npy', soft_test_probs)

    with open(hard_dir / 'metrics.json', 'w') as f:
        json.dump({
            'val_metrics': hard_val_metrics,
            'test_metrics': hard_test_metrics,
            'ensemble_type': 'hard_voting',
            'num_backbones': len(val_preds)
        }, f, indent=2)

    # Generate plots (use soft_val_probs for ROC since hard voting doesn't produce probs)
    create_all_plots(
        y_true=test_labels,
        y_pred=hard_test_preds,
        y_probs=soft_test_probs,  # Use soft voting probs for ROC
        class_names=class_names,
        output_dir=hard_dir,
        prefix='hard_voting'
    )

    results['hard_voting'] = {
        'val_metrics': hard_val_metrics,
        'test_metrics': hard_test_metrics,
        'test_accuracy': hard_test_metrics['accuracy'],
        'output_dir': str(hard_dir)
    }

    # 3. Weighted Voting
    logger.info("\n3. Weighted Voting Ensemble")
    weighted_train_preds, weighted_train_probs = weighted_voting_ensemble(train_preds, weights)
    weighted_val_preds, weighted_val_probs = weighted_voting_ensemble(val_preds, weights)
    weighted_test_preds, weighted_test_probs = weighted_voting_ensemble(test_preds, weights)

    weighted_val_metrics = evaluate_ensemble(weighted_val_preds, val_labels)
    weighted_test_metrics = evaluate_ensemble(weighted_test_preds, test_labels)

    logger.info(f"  Val Acc: {weighted_val_metrics['accuracy']:.4f}")
    logger.info(f"  Test Acc: {weighted_test_metrics['accuracy']:.4f}")

    # Save
    weighted_dir = output_dir / 'weighted_voting'
    weighted_dir.mkdir(exist_ok=True)

    np.savez_compressed(
        weighted_dir / 'predictions.npz',
        train_predictions=weighted_train_preds,
        train_probabilities=weighted_train_probs,
        val_predictions=weighted_val_preds,
        val_probabilities=weighted_val_probs,
        test_predictions=weighted_test_preds,
        test_probabilities=weighted_test_probs,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels
    )

    # Also save individual .npy files
    np.save(weighted_dir / 'train_predictions.npy', weighted_train_probs)
    np.save(weighted_dir / 'val_predictions.npy', weighted_val_probs)
    np.save(weighted_dir / 'test_predictions.npy', weighted_test_probs)

    with open(weighted_dir / 'metrics.json', 'w') as f:
        json.dump({
            'val_metrics': weighted_val_metrics,
            'test_metrics': weighted_test_metrics,
            'ensemble_type': 'weighted_voting',
            'num_backbones': len(val_preds),
            'weights': weights
        }, f, indent=2)

    # Generate plots
    create_all_plots(
        y_true=test_labels,
        y_pred=weighted_test_preds,
        y_probs=weighted_test_probs,
        class_names=class_names,
        output_dir=weighted_dir,
        prefix='weighted_voting'
    )

    results['weighted_voting'] = {
        'val_metrics': weighted_val_metrics,
        'test_metrics': weighted_test_metrics,
        'test_accuracy': weighted_test_metrics['accuracy'],
        'output_dir': str(weighted_dir)
    }

    # 4. Logit Averaging
    logger.info("\n4. Logit Averaging Ensemble")
    logit_train_preds, logit_train_probs = logit_averaging_ensemble(train_preds)
    logit_val_preds, logit_val_probs = logit_averaging_ensemble(val_preds)
    logit_test_preds, logit_test_probs = logit_averaging_ensemble(test_preds)

    logit_val_metrics = evaluate_ensemble(logit_val_preds, val_labels)
    logit_test_metrics = evaluate_ensemble(logit_test_preds, test_labels)

    logger.info(f"  Val Acc: {logit_val_metrics['accuracy']:.4f}")
    logger.info(f"  Test Acc: {logit_test_metrics['accuracy']:.4f}")

    # Save
    logit_dir = output_dir / 'logit_averaging'
    logit_dir.mkdir(exist_ok=True)

    np.savez_compressed(
        logit_dir / 'predictions.npz',
        train_predictions=logit_train_preds,
        train_probabilities=logit_train_probs,
        val_predictions=logit_val_preds,
        val_probabilities=logit_val_probs,
        test_predictions=logit_test_preds,
        test_probabilities=logit_test_probs,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels
    )

    # Also save individual .npy files
    np.save(logit_dir / 'train_predictions.npy', logit_train_probs)
    np.save(logit_dir / 'val_predictions.npy', logit_val_probs)
    np.save(logit_dir / 'test_predictions.npy', logit_test_probs)

    with open(logit_dir / 'metrics.json', 'w') as f:
        json.dump({
            'val_metrics': logit_val_metrics,
            'test_metrics': logit_test_metrics,
            'ensemble_type': 'logit_averaging',
            'num_backbones': len(val_preds)
        }, f, indent=2)

    # Generate plots
    create_all_plots(
        y_true=test_labels,
        y_pred=logit_test_preds,
        y_probs=logit_test_probs,
        class_names=class_names,
        output_dir=logit_dir,
        prefix='logit_averaging'
    )

    results['logit_averaging'] = {
        'val_metrics': logit_val_metrics,
        'test_metrics': logit_test_metrics,
        'test_accuracy': logit_test_metrics['accuracy'],
        'output_dir': str(logit_dir)
    }

    # Generate ensemble comparison plot
    logger.info("\n[*] Generating ensemble comparison plot...")
    from ensemble_plots import plot_ensemble_comparison

    ensemble_names = ['Soft Voting', 'Hard Voting', 'Weighted Voting', 'Logit Averaging']
    ensemble_accuracies = [
        results['soft_voting']['test_accuracy'],
        results['hard_voting']['test_accuracy'],
        results['weighted_voting']['test_accuracy'],
        results['logit_averaging']['test_accuracy']
    ]

    plot_ensemble_comparison(
        ensemble_names=ensemble_names,
        ensemble_accuracies=ensemble_accuracies,
        output_dir=output_dir,
        prefix='stage2_comparison'
    )
    logger.info(f"[OK] Ensemble comparison plot saved to {output_dir}")

    logger.info("\n[OK] Stage 2 completed: 4 score-level ensembles trained")

    return {
        'ensembles': results,
        'num_ensembles': 4,
        'best_ensemble': max(results.keys(), key=lambda k: results[k]['test_accuracy']),
        'best_test_accuracy': max([r['test_accuracy'] for r in results.values()])
    }
