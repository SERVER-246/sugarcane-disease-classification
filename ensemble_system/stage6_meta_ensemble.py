"""
Stage 6: Meta-Ensemble Controller
Combines all ensemble types from Stages 2-5
"""

import sys
from pathlib import Path


# Add BASE-BACK to path BEFORE other imports
BASE_BACK_PATH = Path(__file__).parent.parent / 'BASE-BACK' / 'src'
if str(BASE_BACK_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_BACK_PATH))

import json
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xgboost as xgb
from config.settings import BACKBONES, NUM_CLASSES, SEED
from ensemble_plots import create_all_plots, plot_ensemble_comparison
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from utils import DEVICE, logger


def load_stage2_predictions(stage2_dir: Path, split: str) -> dict[str, np.ndarray]:
    """Load predictions from Stage 2 (score ensembles)"""
    predictions = {}
    ensemble_types = ['soft_voting', 'hard_voting', 'weighted_voting', 'logit_averaging']

    for ens_type in ensemble_types:
        # First try .npy file
        pred_file = stage2_dir / ens_type / f'{split}_predictions.npy'
        if pred_file.exists():
            predictions[f'stage2_{ens_type}'] = np.load(pred_file)
        else:
            # Fallback to .npz file
            npz_file = stage2_dir / ens_type / 'predictions.npz'
            if npz_file.exists():
                data = np.load(npz_file)
                key = f'{split}_probabilities'
                if key in data:
                    predictions[f'stage2_{ens_type}'] = data[key]
                else:
                    logger.warning(f"Key {key} not found in {npz_file}")
            else:
                logger.warning(f"No prediction files found for Stage 2 {ens_type}")

    return predictions


def load_stage3_predictions(stage3_dir: Path, split: str) -> dict[str, np.ndarray]:
    """Load predictions from Stage 3 (stackers)"""

    predictions = {}
    stacker_types = ['logistic_regression', 'xgboost', 'mlp']

    for stacker_type in stacker_types:
        pred_file = stage3_dir / stacker_type / f'{split}_predictions.npy'
        if pred_file.exists():
            predictions[f'stage3_{stacker_type}'] = np.load(pred_file)
        else:
            logger.warning(f"Stage 3 prediction file not found: {pred_file}")

    return predictions


def load_stage4_predictions(stage4_dir: Path, split: str) -> dict[str, np.ndarray]:
    """Load predictions from Stage 4 (feature fusion)"""

    predictions = {}
    # Correct fusion type names matching actual directory structure
    fusion_types = ['concat_mlp', 'attention_fusion', 'bilinear_pooling']

    for fusion_type in fusion_types:
        pred_file = stage4_dir / fusion_type / f'{split}_predictions.npy'
        if pred_file.exists():
            predictions[f'stage4_{fusion_type}'] = np.load(pred_file)
        else:
            logger.warning(f"Stage 4 prediction file not found: {pred_file}")

    return predictions


def load_stage5_predictions(stage5_dir: Path, split: str) -> dict[str, np.ndarray]:
    """Load predictions from Stage 5 (MoE)"""

    predictions = {}
    pred_file = stage5_dir / f'{split}_predictions.npy'

    if pred_file.exists():
        predictions['stage5_moe'] = np.load(pred_file)
    else:
        logger.warning(f"Stage 5 prediction file not found: {pred_file}")

    return predictions


def load_all_ensemble_predictions(
    stage2_dir: Path,
    stage3_dir: Path,
    stage4_dir: Path,
    stage5_dir: Path,
    split: str
) -> tuple[np.ndarray, list[str]]:
    """
    Load predictions from all stages and combine
    
    Returns:
        features: (N, num_ensembles * num_classes) - concatenated predictions
        ensemble_names: list of ensemble names
    """

    all_predictions = {}

    # Load from each stage
    all_predictions.update(load_stage2_predictions(stage2_dir, split))
    all_predictions.update(load_stage3_predictions(stage3_dir, split))
    all_predictions.update(load_stage4_predictions(stage4_dir, split))
    all_predictions.update(load_stage5_predictions(stage5_dir, split))

    if not all_predictions:
        logger.error(f"No predictions found for {split} split!")
        return None, []

    # Ensure all have same length
    first_key = list(all_predictions.keys())[0]
    num_samples = len(all_predictions[first_key])

    # Filter out mismatched lengths
    filtered_preds = {}
    for name, preds in all_predictions.items():
        if len(preds) == num_samples:
            filtered_preds[name] = preds
        else:
            logger.warning(f"  x Skipping {name}: length mismatch ({len(preds)} vs {num_samples})")

    # Sort for consistency
    ensemble_names = sorted(filtered_preds.keys())

    # Concatenate all predictions
    features = np.concatenate([filtered_preds[name] for name in ensemble_names], axis=1)

    logger.info(f"Loaded {len(ensemble_names)} ensemble predictions for {split}")
    logger.info(f"  Ensembles: {', '.join(ensemble_names)}")
    logger.info(f"  Feature shape: {features.shape}")

    return features, ensemble_names


class MetaMLPController(nn.Module):
    """Meta-controller using MLP"""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def train_xgboost_meta_controller(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    output_dir: Path
) -> dict[str, Any]:
    """Train XGBoost meta-controller"""

    logger.info("\nTraining XGBoost Meta-Controller...")

    # Train model
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        eval_metric='mlogloss'
    )

    model.fit(
        train_features, train_labels,
        eval_set=[(val_features, val_labels)],
        verbose=False
    )

    # Evaluate
    val_preds = model.predict(val_features)
    val_acc = accuracy_score(val_labels, val_preds)

    test_preds = model.predict(test_features)
    test_probs = model.predict_proba(test_features)
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_rec = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

    logger.info(f"  Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

    # Save model
    xgb_dir = output_dir / 'xgboost'
    xgb_dir.mkdir(exist_ok=True)
    joblib.dump(model, xgb_dir / 'xgboost_meta.pkl')

    # Save predictions for plotting
    np.save(xgb_dir / 'test_predictions.npy', test_preds)
    np.save(xgb_dir / 'test_probabilities.npy', test_probs)
    np.save(xgb_dir / 'test_labels.npy', test_labels)

    # Generate plots
    logger.info("  Generating plots for XGBoost meta-controller...")
    class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]
    create_all_plots(
        y_true=test_labels,
        y_pred=test_preds,
        y_probs=test_probs,
        class_names=class_names,
        output_dir=xgb_dir,
        prefix='xgboost_meta'
    )
    logger.info(f"  Plots saved to {xgb_dir}")

    return {
        'method': 'xgboost',
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1)
    }


def train_mlp_meta_controller(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    output_dir: Path,
    epochs: int = 50
) -> dict[str, Any]:
    """Train MLP meta-controller"""

    logger.info("\nTraining MLP Meta-Controller...")

    input_dim = train_features.shape[1]

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_features),
        torch.LongTensor(train_labels)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_features),
        torch.LongTensor(val_labels)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_features),
        torch.LongTensor(test_labels)
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Create model
    model = MetaMLPController(input_dim, NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(features)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        scheduler.step()

        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    model.eval()

    test_preds = []
    test_probs = []
    test_labels_list = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels_list)

    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_rec = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

    logger.info(f"  Best Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")

    # Save model
    mlp_dir = output_dir / 'mlp'
    mlp_dir.mkdir(exist_ok=True)
    torch.save(best_model_state, mlp_dir / 'mlp_meta.pth')

    # Save predictions for plotting
    np.save(mlp_dir / 'test_predictions.npy', test_preds)
    np.save(mlp_dir / 'test_probabilities.npy', test_probs)
    np.save(mlp_dir / 'test_labels.npy', test_labels)

    # Generate plots
    logger.info("  Generating plots for MLP meta-controller...")
    class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]
    create_all_plots(
        y_true=test_labels,
        y_pred=test_preds,
        y_probs=test_probs,
        class_names=class_names,
        output_dir=mlp_dir,
        prefix='mlp_meta'
    )
    logger.info(f"  Plots saved to {mlp_dir}")

    return {
        'method': 'mlp',
        'val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1)
    }


def train_meta_ensemble_controller(
    stage2_dir: Path,
    stage3_dir: Path,
    stage4_dir: Path,
    stage5_dir: Path,
    val_loader,
    test_loader,
    output_dir: Path,
    class_names: list[str]
) -> dict[str, Any]:
    """Train meta-ensemble combining all previous stages"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("Training Meta-Ensemble Controller")
    logger.info("="*80)

    # Load predictions from all stages
    train_features, ensemble_names = load_all_ensemble_predictions(
        stage2_dir, stage3_dir, stage4_dir, stage5_dir, 'train'
    )
    val_features, _ = load_all_ensemble_predictions(
        stage2_dir, stage3_dir, stage4_dir, stage5_dir, 'val'
    )
    test_features, _ = load_all_ensemble_predictions(
        stage2_dir, stage3_dir, stage4_dir, stage5_dir, 'test'
    )

    if train_features is None:
        logger.error("Failed to load ensemble predictions!")
        return {'error': 'Failed to load predictions'}

    # Load labels from Stage 1 (most reliable source)
    stage1_dir = stage2_dir.parent / 'stage1_individual'
    train_labels_file = stage1_dir / f'{BACKBONES[0]}_train_predictions.npz'
    val_labels_file = stage1_dir / f'{BACKBONES[0]}_val_predictions.npz'
    test_labels_file = stage1_dir / f'{BACKBONES[0]}_test_predictions.npz'

    if not train_labels_file.exists():
        logger.error(f"Cannot find training labels at {train_labels_file}")
        return {'error': 'No labels found'}

    # Load labels from .npz files
    train_labels = np.load(train_labels_file)['labels']
    val_labels = np.load(val_labels_file)['labels'] if val_labels_file.exists() else None
    test_labels = np.load(test_labels_file)['labels'] if test_labels_file.exists() else None

    if val_labels is None or test_labels is None:
        logger.error("Missing validation or test labels!")
        return {'error': 'Missing labels'}

    # Train both meta-controllers
    results = {}

    # 1. XGBoost meta-controller
    xgb_results = train_xgboost_meta_controller(
        train_features, train_labels,
        val_features, val_labels,
        test_features, test_labels,
        output_dir
    )
    results['xgboost'] = xgb_results

    # 2. MLP meta-controller
    mlp_results = train_mlp_meta_controller(
        train_features, train_labels,
        val_features, val_labels,
        test_features, test_labels,
        output_dir,
        epochs=50
    )
    results['mlp'] = mlp_results

    # Find best controller
    best_method = 'xgboost' if xgb_results['test_accuracy'] > mlp_results['test_accuracy'] else 'mlp'
    best_acc = results[best_method]['test_accuracy']

    # Generate meta-controller comparison plot
    logger.info("\n[*] Generating meta-controller comparison plot...")
    controller_names = ['XGBoost Meta', 'MLP Meta']
    controller_accuracies = [
        results['xgboost']['test_accuracy'],
        results['mlp']['test_accuracy']
    ]

    plot_ensemble_comparison(
        ensemble_names=controller_names,
        ensemble_accuracies=controller_accuracies,
        output_dir=output_dir,
        prefix='stage6_meta_comparison'
    )
    logger.info(f"[OK] Meta-controller comparison plot saved to {output_dir}")

    logger.info(f"\n[OK] Stage 6 completed: Best controller={best_method}, accuracy={best_acc:.4f}")
    logger.info(f"  Combined {len(ensemble_names)} ensembles from stages 2-5")

    # Save overall metrics
    overall_metrics = {
        'test_accuracy': float(best_acc),  # For pipeline compatibility
        'num_ensembles_combined': len(ensemble_names),
        'ensemble_names': ensemble_names,
        'best_method': best_method,
        'best_accuracy': float(best_acc),
        'controllers': results
    }

    with open(output_dir / 'meta_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=2)

    return overall_metrics
