"""
Stage 4: Feature-Level Fusion
Concatenation+MLP, Attention Fusion, Bilinear Pooling
Uses embeddings extracted in Stage 1
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config.settings import BACKBONES, NUM_CLASSES
from ensemble_plots import create_all_plots, plot_ensemble_comparison
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from utils import DEVICE, logger


class ConcatMLPFusion(nn.Module):
    """Simple concatenation of embeddings + MLP classifier"""

    def __init__(self, embedding_dims: list[int], num_classes: int, hidden_dim: int = 512):
        super().__init__()
        total_dim = sum(embedding_dims)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        # Concatenate all embeddings
        x = torch.cat(embeddings, dim=1)  # (B, total_dim)
        return self.fusion_mlp(x)


class AttentionFusion(nn.Module):
    """Attention-weighted fusion of embeddings"""

    def __init__(self, embedding_dims: list[int], num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.num_backbones = len(embedding_dims)

        # Project each embedding to common dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in embedding_dims
        ])

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        # Project all embeddings to common space
        projected = []
        for emb, proj in zip(embeddings, self.projections):
            projected.append(proj(emb))  # (B, hidden_dim)

        # Stack: (B, num_backbones, hidden_dim)
        stacked = torch.stack(projected, dim=1)

        # Compute attention weights: (B, num_backbones, 1)
        attn_scores = self.attention(stacked)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum: (B, hidden_dim)
        fused = (stacked * attn_weights).sum(dim=1)

        return self.classifier(fused)


class BilinearFusion(nn.Module):
    """Bilinear pooling fusion (pairwise interactions)"""

    def __init__(self, embedding_dims: list[int], num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.num_backbones = len(embedding_dims)

        # Project embeddings to common dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in embedding_dims
        ])

        # Bilinear interaction
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        # Project all embeddings
        projected = [proj(emb) for emb, proj in zip(embeddings, self.projections)]

        # Average all projected embeddings
        avg_proj = torch.stack(projected, dim=0).mean(dim=0)  # (B, hidden_dim)

        # Bilinear interaction between pairs (simplified: use first and average)
        if len(projected) > 0:
            bilinear_out = self.bilinear(projected[0], avg_proj)
        else:
            bilinear_out = avg_proj

        return self.classifier(bilinear_out)


def load_embeddings_from_stage1(stage1_dir: Path, split: str) -> tuple[list[np.ndarray], np.ndarray, list[int]]:
    """
    Load embeddings from Stage 1 predictions
    
    Returns:
        embeddings_list: List of embeddings per backbone (N, embedding_dim)
        labels: Ground truth labels (N,)
        embedding_dims: List of embedding dimensions
    """

    embeddings_list = []
    labels = None
    embedding_dims = []

    for backbone_name in BACKBONES:
        pred_file = stage1_dir / f'{backbone_name}_{split}_predictions.npz'

        if not pred_file.exists():
            logger.warning(f"  x Missing {split} embeddings for {backbone_name}")
            continue

        data = np.load(pred_file)
        embeddings = data['embeddings']

        if labels is None:
            labels = data['labels']

        embeddings_list.append(embeddings)
        embedding_dims.append(embeddings.shape[1])

    logger.info(f"Loaded {len(embeddings_list)} backbone embeddings for {split}")
    return embeddings_list, labels, embedding_dims


def train_fusion_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
    output_dir: Path,
    epochs: int = 50
) -> dict[str, Any]:
    """Train a fusion model"""

    logger.info(f"\nTraining {model_name}...")

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

        for batch in train_loader:
            embeddings_list = [emb.to(DEVICE) for emb in batch[:-1]]
            labels = batch[-1].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(embeddings_list)
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
            for batch in val_loader:
                embeddings_list = [emb.to(DEVICE) for emb in batch[:-1]]
                labels = batch[-1].to(DEVICE)

                outputs = model(embeddings_list)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        # Learning rate scheduling
        scheduler.step()

        # Log progress
        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
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
        split_preds = []
        split_labels = []
        split_probs = []

        with torch.no_grad():
            for batch in split_loader:
                embeddings_list = [emb.to(DEVICE) for emb in batch[:-1]]
                labels = batch[-1].to(DEVICE)

                outputs = model(embeddings_list)
                probs = F.softmax(outputs, dim=1)

                split_probs.extend(probs.cpu().numpy())
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
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_model_state, model_dir / 'model.pth')

    # Save predictions for Stage 6
    for split_name in ['train', 'val', 'test']:
        np.save(model_dir / f'{split_name}_predictions.npy', all_preds[split_name])
        np.save(model_dir / f'{split_name}_labels.npy', all_labels[split_name])

    logger.info(f"  Saved predictions to {model_dir}")

    # Generate plots
    logger.info(f"  Generating plots for {model_name}...")
    class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]  # Replace with actual class names if available
    create_all_plots(
        y_true=test_labels,
        y_pred=test_preds,
        y_probs=all_preds['test'],
        class_names=class_names,
        output_dir=model_dir,
        prefix=model_name
    )
    logger.info(f"  Plots saved to {model_dir}")

    # Save metrics
    metrics = {
        'val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1),
        'model_type': model_name
    }

    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def train_all_fusion_models(
    stage1_dir: Path,
    train_loader,
    val_loader,
    test_loader,
    output_dir: Path,
    class_names: list[str]
) -> dict[str, Any]:
    """Train all three fusion models"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("Training Feature-Level Fusion Models")
    logger.info("="*80)

    # Load embeddings from Stage 1
    train_embeddings, train_labels, embedding_dims = load_embeddings_from_stage1(stage1_dir, 'train')
    val_embeddings, val_labels, _ = load_embeddings_from_stage1(stage1_dir, 'val')
    test_embeddings, test_labels, _ = load_embeddings_from_stage1(stage1_dir, 'test')

    if len(train_embeddings) == 0:
        logger.error("No embeddings found from Stage 1!")
        return {'error': 'No embeddings from Stage 1'}

    # Create datasets
    train_dataset = TensorDataset(
        *[torch.FloatTensor(emb) for emb in train_embeddings],
        torch.LongTensor(train_labels)
    )
    val_dataset = TensorDataset(
        *[torch.FloatTensor(emb) for emb in val_embeddings],
        torch.LongTensor(val_labels)
    )
    test_dataset = TensorDataset(
        *[torch.FloatTensor(emb) for emb in test_embeddings],
        torch.LongTensor(test_labels)
    )

    # Create dataloaders
    train_loader_fusion = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader_fusion = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader_fusion = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    results = {}

    # 1. Concat+MLP
    logger.info("\n1. Concat+MLP Fusion")
    concat_model = ConcatMLPFusion(embedding_dims, NUM_CLASSES, hidden_dim=512)
    concat_metrics = train_fusion_model(
        concat_model, train_loader_fusion, val_loader_fusion, test_loader_fusion,
        'concat_mlp', output_dir, epochs=50
    )
    results['concat_mlp'] = concat_metrics

    # 2. Attention Fusion
    logger.info("\n2. Attention Fusion")
    attention_model = AttentionFusion(embedding_dims, NUM_CLASSES, hidden_dim=256)
    attention_metrics = train_fusion_model(
        attention_model, train_loader_fusion, val_loader_fusion, test_loader_fusion,
        'attention_fusion', output_dir, epochs=50
    )
    results['attention_fusion'] = attention_metrics

    # 3. Bilinear Pooling
    logger.info("\n3. Bilinear Pooling Fusion")
    bilinear_model = BilinearFusion(embedding_dims, NUM_CLASSES, hidden_dim=256)
    bilinear_metrics = train_fusion_model(
        bilinear_model, train_loader_fusion, val_loader_fusion, test_loader_fusion,
        'bilinear_pooling', output_dir, epochs=50
    )
    results['bilinear_pooling'] = bilinear_metrics

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])

    # Generate fusion model comparison plot
    logger.info("\n[*] Generating fusion model comparison plot...")
    fusion_names = ['Concat+MLP', 'Attention Fusion', 'Bilinear Pooling']
    fusion_accuracies = [
        results['concat_mlp']['test_accuracy'],
        results['attention_fusion']['test_accuracy'],
        results['bilinear_pooling']['test_accuracy']
    ]

    plot_ensemble_comparison(
        ensemble_names=fusion_names,
        ensemble_accuracies=fusion_accuracies,
        output_dir=output_dir,
        prefix='stage4_fusion_comparison'
    )
    logger.info(f"[OK] Fusion comparison plot saved to {output_dir}")

    logger.info("\n[OK] Stage 4 completed: 3 fusion models trained")
    logger.info(f"Best model: {best_model[0]} with {best_model[1]['test_accuracy']:.4f} test accuracy")

    return {
        'fusion_models': results,
        'num_models': 3,
        'best_model': best_model[0],
        'best_test_accuracy': best_model[1]['test_accuracy']
    }
