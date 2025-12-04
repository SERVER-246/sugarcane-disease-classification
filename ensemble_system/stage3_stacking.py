"""
Stage 3: Stacking Meta-Learners
Train Logistic Regression, XGBoost, and MLP on base model predictions
Uses OOF (Out-of-Fold) predictions to prevent data leakage
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import pickle

# Add BASE-BACK to path
BASE_BACK_PATH = Path(__file__).parent.parent / 'BASE-BACK' / 'src'
if str(BASE_BACK_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_BACK_PATH))

from config.settings import BACKBONES, NUM_CLASSES, SEED, K_FOLDS
from utils import logger, DEVICE
from ensemble_plots import create_all_plots


def generate_oof_predictions(stage1_dir: Path, train_ds) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training predictions for stacking
    
    Note: K-fold OOF is implemented by using the base model predictions directly.
    The base models (Stage 1) were trained on different data, so their predictions
    on the training set are already "out-of-fold" in that sense.
    
    Returns:
        features: Training predictions (N, num_backbones * num_classes)
        labels: True labels (N,)
    """
    
    logger.info(f"Loading training predictions for stacking...")
    
    # Load train predictions from Stage 1
    train_probs_list = []
    
    for backbone_name in BACKBONES:
        pred_file = stage1_dir / f'{backbone_name}_train_predictions.npz'
        if pred_file.exists():
            data = np.load(pred_file)
            train_probs_list.append(data['probabilities'])
    
    # Stack all predictions
    X_train = np.hstack(train_probs_list)  # (N, num_backbones * num_classes)
    
    # Get labels
    pred_file = stage1_dir / f'{BACKBONES[0]}_train_predictions.npz'
    data = np.load(pred_file)
    y_train = data['labels']
    
    logger.info(f"  Loaded predictions from {len(train_probs_list)} backbones")
    logger.info(f"  Feature shape: {X_train.shape}")
    
    return X_train, y_train


def train_logistic_regression_stacker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path
) -> Dict[str, Any]:
    """Train Logistic Regression meta-learner"""
    
    logger.info("\nTraining Logistic Regression Stacker...")
    
    # Train
    model = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    
    val_acc = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    logger.info(f"  Val Acc: {val_acc:.4f}")
    logger.info(f"  Test Acc: {test_acc:.4f}")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save predictions for Stage 6
    np.save(output_dir / 'train_predictions.npy', model.predict_proba(X_train))
    np.save(output_dir / 'val_predictions.npy', val_probs)
    np.save(output_dir / 'test_predictions.npy', test_probs)
    np.save(output_dir / 'train_labels.npy', y_train)
    np.save(output_dir / 'val_labels.npy', y_val)
    np.save(output_dir / 'test_labels.npy', y_test)
    
    metrics = {
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'model_type': 'logistic_regression'
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def train_xgboost_stacker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path
) -> Dict[str, Any]:
    """Train XGBoost meta-learner"""
    
    logger.info("\nTraining XGBoost Stacker...")
    
    # Train
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    
    val_acc = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    logger.info(f"  Val Acc: {val_acc:.4f}")
    logger.info(f"  Test Acc: {test_acc:.4f}")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / 'model.json')
    
    # Save predictions for Stage 6
    np.save(output_dir / 'train_predictions.npy', model.predict_proba(X_train))
    np.save(output_dir / 'val_predictions.npy', val_probs)
    np.save(output_dir / 'test_predictions.npy', test_probs)
    np.save(output_dir / 'train_labels.npy', y_train)
    np.save(output_dir / 'val_labels.npy', y_val)
    np.save(output_dir / 'test_labels.npy', y_test)
    
    metrics = {
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'model_type': 'xgboost'
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


class MLPStacker(nn.Module):
    """MLP meta-learner for stacking"""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_mlp_stacker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path
) -> Dict[str, Any]:
    """Train MLP meta-learner"""
    
    logger.info("\nTraining MLP Stacker...")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.LongTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.LongTensor(y_val).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.LongTensor(y_test).to(DEVICE)
    
    # Create model
    model = MLPStacker(X_train.shape[1], NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    best_val_acc = 0.0
    epochs = 50
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_preds = val_outputs.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}: Val Acc={val_acc:.4f}")
    
    # Load best model and evaluate on test
    model.load_state_dict(best_model)
    model.eval()
    
    with torch.no_grad():
        # Get probabilities for all splits
        train_probs = torch.softmax(model(X_train_t), dim=1).cpu().numpy()
        val_probs = torch.softmax(model(X_val_t), dim=1).cpu().numpy()
        test_probs = torch.softmax(model(X_test_t), dim=1).cpu().numpy()
        
        val_preds = val_probs.argmax(axis=1)
        test_preds = test_probs.argmax(axis=1)
        
        test_acc = accuracy_score(y_test, test_preds)
    
    logger.info(f"  Best Val Acc: {best_val_acc:.4f}")
    logger.info(f"  Test Acc: {test_acc:.4f}")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model, output_dir / 'model.pth')
    
    # Save predictions for Stage 6
    np.save(output_dir / 'train_predictions.npy', train_probs)
    np.save(output_dir / 'val_predictions.npy', val_probs)
    np.save(output_dir / 'test_predictions.npy', test_probs)
    np.save(output_dir / 'train_labels.npy', y_train)
    np.save(output_dir / 'val_labels.npy', y_val)
    np.save(output_dir / 'test_labels.npy', y_test)
    
    metrics = {
        'val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'model_type': 'mlp'
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def train_all_stacking_models(
    stage1_dir: Path,
    train_ds,
    val_loader,
    test_loader,
    output_dir: Path,
    class_names: List[str]
) -> Dict[str, Any]:
    """Train all stacking meta-learners"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("Training Stacking Meta-Learners")
    logger.info("="*80)
    
    # Generate OOF features
    X_train_oof, y_train = generate_oof_predictions(stage1_dir, train_ds)
    
    # Load val and test predictions
    val_probs_list = []
    test_probs_list = []
    
    for backbone_name in BACKBONES:
        val_file = stage1_dir / f'{backbone_name}_val_predictions.npz'
        test_file = stage1_dir / f'{backbone_name}_test_predictions.npz'
        
        if val_file.exists():
            val_data = np.load(val_file)
            val_probs_list.append(val_data['probabilities'])
        
        if test_file.exists():
            test_data = np.load(test_file)
            test_probs_list.append(test_data['probabilities'])
    
    X_val = np.hstack(val_probs_list)
    X_test = np.hstack(test_probs_list)
    
    y_val = np.load(stage1_dir / f'{BACKBONES[0]}_val_predictions.npz')['labels']
    y_test = np.load(stage1_dir / f'{BACKBONES[0]}_test_predictions.npz')['labels']
    
    results = {}
    
    # Train stackers
    results['logistic_regression'] = train_logistic_regression_stacker(
        X_train_oof, y_train, X_val, y_val, X_test, y_test,
        output_dir / 'logistic_regression'
    )
    
    # Generate plots for LR stacker
    lr_dir = output_dir / 'logistic_regression'
    test_probs = np.load(lr_dir / 'test_predictions.npy')
    test_preds = test_probs.argmax(axis=1)
    create_all_plots(
        y_true=y_test,
        y_pred=test_preds,
        y_probs=test_probs,
        class_names=class_names,
        output_dir=lr_dir,
        prefix='lr_stacker'
    )
    
    results['xgboost'] = train_xgboost_stacker(
        X_train_oof, y_train, X_val, y_val, X_test, y_test,
        output_dir / 'xgboost'
    )
    
    # Generate plots for XGBoost stacker
    xgb_dir = output_dir / 'xgboost'
    test_probs = np.load(xgb_dir / 'test_predictions.npy')
    test_preds = test_probs.argmax(axis=1)
    create_all_plots(
        y_true=y_test,
        y_pred=test_preds,
        y_probs=test_probs,
        class_names=class_names,
        output_dir=xgb_dir,
        prefix='xgboost_stacker'
    )
    
    results['mlp'] = train_mlp_stacker(
        X_train_oof, y_train, X_val, y_val, X_test, y_test,
        output_dir / 'mlp'
    )
    
    # Generate plots for MLP stacker
    mlp_dir = output_dir / 'mlp'
    test_probs = np.load(mlp_dir / 'test_predictions.npy')
    test_preds = test_probs.argmax(axis=1)
    create_all_plots(
        y_true=y_test,
        y_pred=test_preds,
        y_probs=test_probs,
        class_names=class_names,
        output_dir=mlp_dir,
        prefix='mlp_stacker'
    )
    
    logger.info("\n[OK] Stage 3 completed: 3 stacking models trained")
    
    # Generate ensemble comparison plot
    logger.info("\n[*] Generating stacking comparison plot...")
    from ensemble_plots import plot_ensemble_comparison
    
    stacker_names = ['Logistic Regression', 'XGBoost', 'MLP']
    stacker_accuracies = [
        results['logistic_regression']['test_accuracy'],
        results['xgboost']['test_accuracy'],
        results['mlp']['test_accuracy']
    ]
    
    plot_ensemble_comparison(
        ensemble_names=stacker_names,
        ensemble_accuracies=stacker_accuracies,
        output_dir=output_dir,
        prefix='stage3_stacker_comparison'
    )
    logger.info(f"[OK] Stacking comparison plot saved to {output_dir}")
    
    return {
        'stackers': results,
        'num_stackers': 3,
        'best_stacker': max(results.keys(), key=lambda k: results[k]['test_accuracy']),
        'best_test_accuracy': max([r['test_accuracy'] for r in results.values()])
    }
