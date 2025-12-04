"""Training pipeline - model training, validation, and metrics"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
import numpy as np
from pathlib import Path
import json
import time
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# Handle both package imports and direct sys.path imports
try:
    from ..config.settings import (
        BATCH_SIZE, IMG_SIZE, WEIGHT_DECAY, BACKBONE_LR, HEAD_LR,
        EPOCHS_HEAD, EPOCHS_FINETUNE, PATIENCE_HEAD, PATIENCE_FT,
        CKPT_DIR, METRICS_DIR
    )
    from ..utils import logger, DEVICE, get_device
except ImportError:
    from config.settings import (
        BATCH_SIZE, IMG_SIZE, WEIGHT_DECAY, BACKBONE_LR, HEAD_LR,
        EPOCHS_HEAD, EPOCHS_FINETUNE, PATIENCE_HEAD, PATIENCE_FT,
        CKPT_DIR, METRICS_DIR
    )
    from utils import logger, DEVICE, get_device

# =============================================================================
# TRAINING HELPERS
# =============================================================================

def _unwrap_logits(outputs):
    """Handle inception aux_logits and various output formats"""
    if isinstance(outputs, torch.Tensor):
        return outputs, None
    if hasattr(outputs, 'logits'):
        return outputs.logits, getattr(outputs, 'aux_logits', None)
    if isinstance(outputs, (tuple, list)):
        main = None; aux = None
        for o in outputs:
            if isinstance(o, torch.Tensor) and main is None:
                main = o
            if hasattr(o, 'logits') and main is None:
                main = o.logits
            if isinstance(o, torch.Tensor) and main is not None and aux is None:
                aux = o
        if main is not None:
            return main, aux
    if isinstance(outputs, dict) and 'logits' in outputs:
        return outputs['logits'], outputs.get('aux_logits', None)
    return outputs, None

def get_loss_function_for_backbone(backbone_name, num_classes):
    """Get optimized loss function per backbone"""
    transformer_models = ['CustomSwinTransformer', 'CustomViTHybrid', 
                         'CustomDeiTStyle', 'CustomCoAtNet', 'CustomMaxViT']
    
    if backbone_name in transformer_models:
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        return nn.CrossEntropyLoss(label_smoothing=0.05)

def create_optimized_optimizer(model, lr, backbone_name):
    """Create optimized optimizer for specific backbone
    
    Args:
        model: PyTorch model
        lr: base learning rate
        backbone_name: name of the backbone
    """
    lr_configs = {
        'CustomConvNeXt': lr * 1.2,
        'CustomEfficientNetV4': lr * 1.1,
        'CustomGhostNetV2': lr * 1.5,
        'CustomResNetMish': lr * 1.2,
        'CustomCSPDarkNet': lr * 1.2,
        'CustomInceptionV4': lr * 1.0,
        'CustomViTHybrid': lr * 1.2,  # Match Base_backbones.py - higher LR works!
        'CustomSwinTransformer': lr * 0.5,
        'CustomCoAtNet': lr * 0.6,
        'CustomRegNet': lr * 1.2,
        'CustomDenseNetHybrid': lr * 1.1,
        'CustomDeiTStyle': lr * 0.5,
        'CustomMaxViT': lr * 0.6,
        'CustomMobileOne': lr * 1.0,
        'CustomDynamicConvNet': lr * 1.0
    }
    
    actual_lr = lr_configs.get(backbone_name, lr)
    
    # Standard optimizer for all backbones (no special handling)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=actual_lr,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer

def create_improved_scheduler(optimizer, epochs, steps_per_epoch, backbone_name):
    """Create learning rate scheduler"""
    transformer_models = ['CustomSwinTransformer', 'CustomViTHybrid', 
                         'CustomDeiTStyle', 'CustomCoAtNet', 'CustomMaxViT']
    
    if backbone_name in transformer_models:
        def lr_lambda(current_step):
            warmup_steps = steps_per_epoch * 3
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            progress = float(current_step - warmup_steps) / float(max(1, epochs * steps_per_epoch - warmup_steps))
            return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=optimizer.param_groups[0]['lr'] * 5,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.2,
            div_factor=20,
            final_div_factor=1000
        )

def save_checkpoint(path: Path, model, optimizer=None, scheduler=None, extra=None):
    """Save model checkpoint with detailed information"""
    path = Path(path)
    state = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        try:
            state["optimizer_state_dict"] = optimizer.state_dict()
        except Exception:
            state["optimizer_state_dict"] = None
    if scheduler is not None:
        try:
            state["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            state["scheduler_state_dict"] = None
    if extra is not None:
        state["extra"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
    logger.info(f"Checkpoint saved to {path}")

# =============================================================================
# TRAINING LOOPS
# =============================================================================

def train_epoch_optimized(model, loader, optimizer, criterion, device=None):
    """Enhanced training epoch with detailed metrics collection"""
    if device is None:
        device = DEVICE
    
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    current_lr = optimizer.param_groups[0]['lr']
    scaler = amp.GradScaler() if device.type == 'cuda' else None
    
    pbar = tqdm(loader, desc=f"Train (LR: {current_lr:.2e})", leave=False)
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Input validation: Check for NaN/Inf in input data
        if torch.isnan(images).any() or torch.isinf(images).any():
            logger.warning(f"Batch {batch_idx}: NaN/Inf in input images, skipping")
            optimizer.zero_grad()
            continue
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with amp.autocast(device_type='cuda'):
                outputs = model(images)
                logits, aux_logits = _unwrap_logits(outputs)
                loss_main = criterion(logits, targets)
                loss = loss_main
                
                if aux_logits is not None and isinstance(aux_logits, torch.Tensor):
                    aux_loss = criterion(aux_logits, targets)
                    loss += 0.4 * aux_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            logits, aux_logits = _unwrap_logits(outputs)
            loss_main = criterion(logits, targets)
            loss = loss_main
            
            if aux_logits is not None and isinstance(aux_logits, torch.Tensor):
                aux_loss = criterion(aux_logits, targets)
                loss += 0.4 * aux_loss
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(targets.cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())
        
        if batch_idx % 10 == 0:
            current_samples = sum([len(p) for p in all_preds])
            avg_loss = running_loss / current_samples if current_samples > 0 else 0
            pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
    
    all_preds_cat = np.concatenate(all_preds) if len(all_preds) > 0 else np.array([])
    all_labels_cat = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])
    all_probs_cat = np.concatenate(all_probs) if len(all_probs) > 0 else np.array([])
    
    acc = (all_preds_cat == all_labels_cat).mean() if all_preds_cat.size > 0 else 0.0
    prec = precision_score(all_labels_cat, all_preds_cat, average='macro', zero_division=0) if all_preds_cat.size > 0 else 0.0
    rec = recall_score(all_labels_cat, all_preds_cat, average='macro', zero_division=0) if all_preds_cat.size > 0 else 0.0
    f1 = f1_score(all_labels_cat, all_preds_cat, average='macro', zero_division=0) if all_preds_cat.size > 0 else 0.0

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, acc, prec, rec, f1, all_preds_cat, all_labels_cat, all_probs_cat

def validate_epoch_optimized(model, loader, criterion, device=None):
    """Enhanced validation epoch with detailed metrics collection"""
    if device is None:
        device = DEVICE
    
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    
    pbar = tqdm(loader, desc="Validate", leave=False)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if device.type == 'cuda':
                with amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    logits, _ = _unwrap_logits(outputs)
                    loss = criterion(logits, targets)
            else:
                outputs = model(images)
                logits, _ = _unwrap_logits(outputs)
                loss = criterion(logits, targets)
            
            running_loss += loss.item() * images.size(0)
            
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(targets.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            
            if batch_idx % 5 == 0:
                current_samples = sum([len(p) for p in all_preds]) if all_preds else 1
                current_loss = running_loss / current_samples
                pbar.set_postfix({'loss': f"{current_loss:.4f}"})
    
    all_preds = np.concatenate(all_preds) if len(all_preds)>0 else np.array([])
    all_labels = np.concatenate(all_labels) if len(all_labels)>0 else np.array([])
    all_probs = np.concatenate(all_probs) if len(all_probs)>0 else np.array([])
    
    acc = (all_preds == all_labels).mean() if all_preds.size>0 else 0.0
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) if all_preds.size>0 else 0.0
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0) if all_preds.size>0 else 0.0
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if all_preds.size>0 else 0.0
    
    total = len(loader.dataset) if hasattr(loader, 'dataset') else (all_labels.size if all_labels.size > 0 else 1)
    return running_loss / total, acc, prec, rec, f1, all_preds, all_labels, all_probs

__all__ = [
    '_unwrap_logits',
    'get_loss_function_for_backbone',
    'create_optimized_optimizer',
    'create_improved_scheduler',
    'save_checkpoint',
    'train_epoch_optimized',
    'validate_epoch_optimized',
]
