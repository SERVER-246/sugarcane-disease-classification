"""
Stage 5: Mixture of Experts (MoE)
Gating network with Top-K routing
"""

import sys
from pathlib import Path

# Add BASE-BACK to path BEFORE other imports
BASE_BACK_PATH = Path(__file__).parent.parent / 'BASE-BACK' / 'src'
if str(BASE_BACK_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_BACK_PATH))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple
import json
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config.settings import BACKBONES, NUM_CLASSES, SEED
from utils import logger, DEVICE
from ensemble_plots import create_all_plots


class GatingNetwork(nn.Module):
    """
    Gating network that learns which experts to use for each sample
    """
    
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 5):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_experts)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            gate_scores: (B, num_experts) - gating scores for all experts
            top_k_indices: (B, top_k) - indices of top-k experts
        """
        gate_logits = self.gate(x)  # (B, num_experts)
        gate_scores = F.softmax(gate_logits, dim=1)
        
        # Select top-k experts
        top_k_values, top_k_indices = torch.topk(gate_scores, self.top_k, dim=1)
        
        # Renormalize top-k scores
        top_k_scores = top_k_values / top_k_values.sum(dim=1, keepdim=True)
        
        return gate_scores, top_k_indices, top_k_scores


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts ensemble using predictions from Stage 1
    """
    
    def __init__(self, num_experts: int, num_classes: int, top_k: int = 5):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.top_k = top_k
        
        # Gating network takes concatenated probabilities as input
        self.gating = GatingNetwork(
            input_dim=num_experts * num_classes,
            num_experts=num_experts,
            top_k=top_k
        )
    
    def forward(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_probs: (B, num_experts, num_classes) - predictions from all experts
        
        Returns:
            final_probs: (B, num_classes) - weighted ensemble prediction
        """
        batch_size = expert_probs.size(0)
        
        # Flatten expert predictions for gating network
        flat_probs = expert_probs.view(batch_size, -1)  # (B, num_experts * num_classes)
        
        # Get gating scores and top-k experts
        gate_scores, top_k_indices, top_k_scores = self.gating(flat_probs)
        
        # Weighted combination using top-k experts
        final_probs = torch.zeros(batch_size, self.num_classes).to(expert_probs.device)
        
        for i in range(batch_size):
            for j, expert_idx in enumerate(top_k_indices[i]):
                weight = top_k_scores[i, j]
                final_probs[i] += weight * expert_probs[i, expert_idx]
        
        return final_probs


def load_predictions_from_stage1(stage1_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load probability predictions from Stage 1
    
    Returns:
        all_probs: (N, num_experts, num_classes)
        labels: (N,)
    """
    
    all_probs = []
    labels = None
    
    for backbone_name in BACKBONES:
        pred_file = stage1_dir / f'{backbone_name}_{split}_predictions.npz'
        
        if not pred_file.exists():
            logger.warning(f"  x Missing {split} predictions for {backbone_name}")
            continue
        
        data = np.load(pred_file)
        probs = data['probabilities']  # (N, num_classes)
        
        if labels is None:
            labels = data['labels']
        
        all_probs.append(probs)
    
    # Stack: (N, num_experts, num_classes)
    all_probs = np.stack(all_probs, axis=1)
    
    logger.info(f"Loaded predictions from {len(all_probs[0])} experts for {split}")
    return all_probs, labels


def train_moe_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: Path,
    epochs: int = 30
) -> Dict[str, Any]:
    """Train the MoE model"""
    
    logger.info("\nTraining Mixture of Experts...")
    
    model = model.to(DEVICE)
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
        
        for expert_probs, labels in train_loader:
            expert_probs = expert_probs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(expert_probs)
            
            # CrossEntropy expects logits, so convert probs to logits
            logits = torch.log(outputs + 1e-8)
            loss = criterion(logits, labels)
            
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
            for expert_probs, labels in val_loader:
                expert_probs = expert_probs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(expert_probs)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        scheduler.step()
        
        if epoch % 5 == 0:
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
    
    # Generate predictions for all splits (needed for Stage 6)
    all_preds = {}
    all_labels = {}
    
    for split_name, split_loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        split_probs = []
        split_labels = []
        
        with torch.no_grad():
            for expert_probs, labels in split_loader:
                expert_probs = expert_probs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(expert_probs)
                
                split_probs.extend(outputs.cpu().numpy())
                split_labels.extend(labels.cpu().numpy())
        
        all_preds[split_name] = np.array(split_probs)
        all_labels[split_name] = np.array(split_labels)
    
    # Calculate metrics on test split
    test_preds = all_preds['test'].argmax(axis=1)
    test_labels = all_labels['test']
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_rec = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    
    logger.info(f"  Best Val Acc: {best_val_acc:.4f}")
    logger.info(f"  Test Acc: {test_acc:.4f}")
    
    # Save model
    torch.save(best_model_state, output_dir / 'moe_model.pth')
    
    # Save predictions for Stage 6
    for split_name in ['train', 'val', 'test']:
        np.save(output_dir / f'{split_name}_predictions.npy', all_preds[split_name])
        np.save(output_dir / f'{split_name}_labels.npy', all_labels[split_name])
    
    logger.info(f"  Saved predictions to {output_dir}")
    
    # Generate plots
    logger.info(f"  Generating plots for MoE...")
    class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]  # Replace with actual class names if available
    create_all_plots(
        y_true=test_labels,
        y_pred=test_preds,
        y_probs=all_preds['test'],
        class_names=class_names,
        output_dir=output_dir,
        prefix='moe'
    )
    logger.info(f"  Plots saved to {output_dir}")
    
    # Save metrics
    metrics = {
        'val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1),
        'num_experts': model.num_experts,
        'top_k': model.top_k
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def train_mixture_of_experts(
    stage1_dir: Path,
    train_loader,
    val_loader,
    test_loader,
    output_dir: Path,
    class_names: List[str]
) -> Dict[str, Any]:
    """Train Mixture of Experts with gating"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("Training Mixture of Experts")
    logger.info("="*80)
    
    # Load predictions from Stage 1
    train_probs, train_labels = load_predictions_from_stage1(stage1_dir, 'train')
    val_probs, val_labels = load_predictions_from_stage1(stage1_dir, 'val')
    test_probs, test_labels = load_predictions_from_stage1(stage1_dir, 'test')
    
    if len(train_probs) == 0:
        logger.error("No predictions found from Stage 1!")
        return {'error': 'No predictions from Stage 1'}
    
    num_experts = train_probs.shape[1]
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_probs),
        torch.LongTensor(train_labels)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_probs),
        torch.LongTensor(val_labels)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_probs),
        torch.LongTensor(test_labels)
    )
    
    # Create dataloaders
    train_loader_moe = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader_moe = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader_moe = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Create and train MoE model
    moe_model = MixtureOfExperts(
        num_experts=num_experts,
        num_classes=NUM_CLASSES,
        top_k=min(5, num_experts)  # Use top-5 experts
    )
    
    metrics = train_moe_model(
        moe_model, train_loader_moe, val_loader_moe, test_loader_moe,
        output_dir, epochs=30
    )
    
    logger.info(f"\n[OK] Stage 5 completed: MoE accuracy={metrics['test_accuracy']:.4f}")
    
    return metrics
