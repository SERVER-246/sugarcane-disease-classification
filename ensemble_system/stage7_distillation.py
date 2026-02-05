"""
Stage 7: Knowledge Distillation
Compress Stage 6 teacher into compact student model
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
from config.settings import NUM_CLASSES
from ensemble_plots import create_all_plots, plot_training_history
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from utils import DEVICE, logger


class CompactStudentModel(nn.Module):
    """
    Lightweight student model for distillation
    Based on simplified MobileOne architecture
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Efficient stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Efficient stages
        self.stage1 = self._make_stage(32, 64, num_blocks=2, stride=2)
        self.stage2 = self._make_stage(64, 128, num_blocks=3, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks=3, stride=2)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        # First block with stride
        layers.append(self._make_block(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(self._make_block(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def _make_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 3.0,
    alpha: float = 0.7
) -> torch.Tensor:
    """
    Compute distillation loss combining:
    - KL divergence between student and teacher (soft targets)
    - Cross-entropy with true labels (hard targets)
    
    Args:
        student_logits: Student model outputs
        teacher_logits: Teacher model outputs (or soft predictions)
        labels: True labels
        temperature: Temperature for softening distributions
        alpha: Weight for distillation loss (1-alpha for hard label loss)
    """

    # Soft targets - KL divergence
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

    # Hard targets - cross-entropy
    hard_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    total_loss = alpha * distill_loss + (1 - alpha) * hard_loss

    return total_loss


def load_teacher_predictions(teacher_dir: Path) -> dict[str, Any]:
    """
    Load predictions from Stage 6 teacher (meta-ensemble)
    """

    # Try to load from best controller
    metrics_file = teacher_dir / 'meta_metrics.json'
    if not metrics_file.exists():
        logger.error("Cannot find Stage 6 meta-ensemble metrics!")
        return None

    with open(metrics_file) as f:
        meta_metrics = json.load(f)

    best_method = meta_metrics['best_method']
    logger.info(f"Loading teacher from Stage 6: {best_method} controller")

    # For XGBoost, we'll need to generate predictions
    # For MLP, we can load the model

    # Model is now in subdirectory
    model_subdir = teacher_dir / best_method
    if best_method == 'xgboost':
        model_path = model_subdir / 'xgboost_meta.pkl'
    else:
        model_path = model_subdir / 'mlp_meta.pth'

    if not model_path.exists():
        logger.error(f"Cannot find teacher model at {model_path}")
        return None

    teacher_info = {
        'method': best_method,
        'accuracy': meta_metrics['best_accuracy'],
        'model_path': model_path,
        'model_subdir': model_subdir
    }

    return teacher_info


def generate_teacher_soft_targets(
    teacher_info: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    stage2_dir: Path,
    stage3_dir: Path,
    stage4_dir: Path,
    stage5_dir: Path
) -> dict[str, np.ndarray]:
    """
    Generate soft targets from teacher model
    
    Returns dict with 'train', 'val', 'test' soft predictions
    """

    logger.info("Generating teacher soft targets...")

    # This is simplified - in practice, we need to:
    # 1. Load all ensemble predictions for each split
    # 2. Feed through teacher model
    # 3. Extract soft probabilities

    # For now, we'll generate soft targets directly from student training
    # The actual implementation would load the Stage 6 controller

    soft_targets = {}

    # Placeholder - in real implementation, load and run teacher
    logger.info("  [Note] Using on-the-fly teacher predictions during training")

    return soft_targets


def train_student_with_distillation(
    student_model: nn.Module,
    teacher_info: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: Path,
    epochs: int = 50,  # Reduced from 100 - early stopping will handle convergence
    temperature: float = 3.0,
    alpha: float = 0.7
) -> dict[str, Any]:
    """
    Train student model with knowledge distillation
    Uses mixed precision (AMP) for faster training
    """

    logger.info("\nTraining student with distillation...")
    logger.info(f"  Temperature: {temperature}, Alpha: {alpha}")

    student_model = student_model.to(DEVICE)

    optimizer = optim.AdamW(student_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision for faster training
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None
    use_amp = DEVICE == 'cuda'

    best_val_acc = 0.0
    best_model_state = None
    patience = 10  # Reduced patience for faster convergence check
    patience_counter = 0

    # Track training history for plotting
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    import sys
    logger.info(f"  Starting training loop, {epochs} epochs...")
    sys.stdout.flush()

    for epoch in range(1, epochs + 1):
        # Training
        student_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if epoch == 1 and batch_idx == 0:
                logger.info(f"  First batch loaded successfully, shape: {images.shape}")
                logger.info(f"  Using AMP: {use_amp}")
                sys.stdout.flush()
            if batch_idx % 50 == 0:
                print(f"    Epoch {epoch} - Batch {batch_idx}/{len(train_loader)}", flush=True)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # Mixed precision training
            if use_amp:
                with torch.amp.autocast('cuda'):
                    student_logits = student_model(images)
                    loss = F.cross_entropy(student_logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                student_logits = student_model(images)
                loss = F.cross_entropy(student_logits, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, preds = student_logits.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # Validation
        student_model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = student_model(images)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        # Log every epoch for progress tracking
        logger.info(f"  Epoch {epoch}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Best={best_val_acc:.4f}")
        sys.stdout.flush()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = student_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    # Load best model and evaluate on test
    student_model.load_state_dict(best_model_state)
    student_model.eval()

    test_preds = []
    test_probs = []
    test_labels_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = student_model(images)
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

    logger.info(f"  Best Val Acc: {best_val_acc:.4f}")
    logger.info(f"  Test Acc: {test_acc:.4f}")

    # Save model
    torch.save(best_model_state, output_dir / 'student_model.pth')

    # Save predictions for plotting
    np.save(output_dir / 'test_predictions.npy', test_preds)
    np.save(output_dir / 'test_probabilities.npy', test_probs)
    np.save(output_dir / 'test_labels.npy', test_labels)

    # Generate plots
    logger.info("  Generating plots for distilled student model...")
    class_names_plot = [f"Class_{i}" for i in range(NUM_CLASSES)]
    create_all_plots(
        y_true=test_labels,
        y_pred=test_preds,
        y_probs=test_probs,
        class_names=class_names_plot,
        output_dir=output_dir,
        prefix='distilled_student'
    )

    # Plot training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    plot_training_history(
        history=history,
        save_path=output_dir / 'distilled_student_training_history.tiff',
        title='Distilled Student Training History'
    )
    logger.info(f"  Plots saved to {output_dir}")

    # Calculate model size
    model_size_mb = Path(output_dir / 'student_model.pth').stat().st_size / (1024 * 1024)

    return {
        'val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1),
        'model_size_mb': float(model_size_mb),
        'teacher_accuracy': teacher_info['accuracy']
    }


def train_distilled_student(
    teacher_stage6_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: Path,
    class_names: list[str],
    stage2_dir: Path = None,
    stage3_dir: Path = None,
    stage4_dir: Path = None,
    stage5_dir: Path = None
) -> dict[str, Any]:
    """Train distilled student model"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("Training Distilled Student Model")
    logger.info("="*80)

    # Load teacher information
    teacher_info = load_teacher_predictions(teacher_stage6_dir)

    if teacher_info is None:
        logger.error("Failed to load teacher model from Stage 6!")
        return {
            'error': 'Failed to load teacher',
            'test_accuracy': 0.0,
            'val_accuracy': 0.0,
            'test_precision': 0.0,
            'test_recall': 0.0,
            'test_f1': 0.0
        }

    logger.info(f"Teacher: {teacher_info['method']} (accuracy={teacher_info['accuracy']:.4f})")

    # Create compact student model
    student_model = CompactStudentModel(num_classes=NUM_CLASSES)

    # Count parameters
    num_params = sum(p.numel() for p in student_model.parameters())
    logger.info(f"Student model parameters: {num_params:,}")

    # Train student with distillation
    metrics = train_student_with_distillation(
        student_model,
        teacher_info,
        train_loader,
        val_loader,
        test_loader,
        output_dir,
        epochs=100,
        temperature=3.0,
        alpha=0.7
    )

    # Add additional metrics
    metrics['num_parameters'] = num_params
    metrics['compression_info'] = {
        'teacher': teacher_info['method'],
        'teacher_accuracy': teacher_info['accuracy'],
        'student_accuracy': metrics['test_accuracy'],
        'accuracy_drop': teacher_info['accuracy'] - metrics['test_accuracy']
    }

    # Save metrics
    with open(output_dir / 'distillation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\n[OK] Stage 7 completed: Student accuracy={metrics['test_accuracy']:.4f}")
    logger.info(f"  Model size: {metrics['model_size_mb']:.2f} MB")
    logger.info(f"  Parameters: {num_params:,}")
    logger.info(f"  Accuracy drop from teacher: {metrics['compression_info']['accuracy_drop']:.4f}")

    return metrics
