"""
Stage 1: Individual Models
Extract predictions, logits, and embeddings from all 15 trained backbones
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
import numpy as np
from typing import Dict, List, Tuple, Any
from tqdm.auto import tqdm
import json

from config.settings import BACKBONES, NUM_CLASSES, CKPT_DIR
from utils import logger, DEVICE
from models import create_custom_backbone_safe


def load_trained_backbone(backbone_name: str) -> nn.Module:
    """Load a trained backbone from checkpoint"""
    
    # Try different checkpoint files (prioritize final, then finetune, then head)
    checkpoint_candidates = [
        CKPT_DIR / f'{backbone_name}_final.pth',
        CKPT_DIR / f'{backbone_name}_finetune_best.pth',
        CKPT_DIR / f'{backbone_name}_head_best.pth'
    ]
    
    checkpoint_path = None
    for ckpt in checkpoint_candidates:
        if ckpt.exists():
            checkpoint_path = ckpt
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found for {backbone_name}")
    
    # Create model
    model = create_custom_backbone_safe(backbone_name, NUM_CLASSES)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    logger.info(f"  [OK] Loaded {backbone_name} from {checkpoint_path.name}")
    
    return model


def extract_embeddings_and_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    backbone_name: str
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings and predictions from a model.
    
    Returns:
        Dictionary with logits, probabilities, predictions, embeddings, labels
    """
    
    all_logits = []
    all_probs = []
    all_preds = []
    all_embeddings = []
    all_labels = []
    
    # Hook to extract embeddings (before final classifier)
    embeddings_hook = []
    
    def hook_fn(module, input, output):
        # Store the input to the final layer (embeddings)
        if isinstance(input, tuple):
            embeddings_hook.append(input[0].detach().cpu())
        else:
            embeddings_hook.append(input.detach().cpu())
    
    # Register hook on classifier layer
    if hasattr(model, 'classifier'):
        hook_handle = model.classifier.register_forward_hook(hook_fn)
    elif hasattr(model, 'head'):
        hook_handle = model.head.register_forward_hook(hook_fn)
    elif hasattr(model, 'fc'):
        hook_handle = model.fc.register_forward_hook(hook_fn)
    else:
        # No hook, embeddings will be None
        hook_handle = None
    
    model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Extracting {backbone_name}", leave=False):
            images = images.to(DEVICE)
            
            # Clear embeddings from previous batch
            embeddings_hook.clear()
            
            # Forward pass
            outputs = model(images)
            
            # Handle tuple outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Collect outputs
            logits = outputs.cpu().numpy()
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_logits.append(logits)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            
            # Collect embeddings
            if len(embeddings_hook) > 0:
                emb = embeddings_hook[0]
                # Flatten if needed
                if emb.dim() > 2:
                    emb = emb.view(emb.size(0), -1)
                all_embeddings.append(emb.numpy())
    
    # Remove hook
    if hook_handle is not None:
        hook_handle.remove()
    
    # Concatenate all batches
    result = {
        'logits': np.concatenate(all_logits, axis=0),
        'probabilities': np.concatenate(all_probs, axis=0),
        'predictions': np.concatenate(all_preds, axis=0),
        'labels': np.concatenate(all_labels, axis=0)
    }
    
    if len(all_embeddings) > 0:
        result['embeddings'] = np.concatenate(all_embeddings, axis=0)
    else:
        logger.warning(f"  x No embeddings extracted for {backbone_name}")
        result['embeddings'] = None
    
    return result


def extract_all_predictions_and_embeddings(
    train_loader,
    val_loader,
    test_loader,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Extract predictions and embeddings from all 15 backbones.
    
    Saves:
        - {backbone}_train_predictions.npz
        - {backbone}_val_predictions.npz
        - {backbone}_test_predictions.npz
        - catalog.json (summary of all extractions)
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Extracting predictions and embeddings from all 15 backbones...")
    
    catalog = {}
    successful = 0
    failed = []
    
    for i, backbone_name in enumerate(BACKBONES):
        logger.info(f"\n[{i+1}/15] Processing {backbone_name}")
        
        try:
            # Load model
            model = load_trained_backbone(backbone_name)
            
            # Extract on train set
            logger.info(f"  Extracting train predictions...")
            train_data = extract_embeddings_and_predictions(model, train_loader, backbone_name)
            train_file = output_dir / f'{backbone_name}_train_predictions.npz'
            np.savez_compressed(train_file, **train_data)
            
            # Extract on val set
            logger.info(f"  Extracting val predictions...")
            val_data = extract_embeddings_and_predictions(model, val_loader, backbone_name)
            val_file = output_dir / f'{backbone_name}_val_predictions.npz'
            np.savez_compressed(val_file, **val_data)
            
            # Extract on test set
            logger.info(f"  Extracting test predictions...")
            test_data = extract_embeddings_and_predictions(model, test_loader, backbone_name)
            test_file = output_dir / f'{backbone_name}_test_predictions.npz'
            np.savez_compressed(test_file, **test_data)
            
            # Calculate accuracy
            train_acc = (train_data['predictions'] == train_data['labels']).mean()
            val_acc = (val_data['predictions'] == val_data['labels']).mean()
            test_acc = (test_data['predictions'] == test_data['labels']).mean()
            
            catalog[backbone_name] = {
                'train_file': str(train_file),
                'val_file': str(val_file),
                'test_file': str(test_file),
                'train_accuracy': float(train_acc),
                'val_accuracy': float(val_acc),
                'test_accuracy': float(test_acc),
                'embedding_dim': train_data['embeddings'].shape[1] if train_data['embeddings'] is not None else None,
                'num_classes': NUM_CLASSES,
                'status': 'success'
            }
            
            logger.info(f"  [OK] {backbone_name}: Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}")
            successful += 1
            
            # Free memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"  x Failed to process {backbone_name}: {e}")
            failed.append(backbone_name)
            catalog[backbone_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Save catalog
    catalog_file = output_dir / 'catalog.json'
    full_catalog = {
        'num_backbones': len(BACKBONES),
        'successful': successful,
        'failed': len(failed),
        'failed_backbones': failed,
        'backbones': catalog
    }
    
    with open(catalog_file, 'w') as f:
        json.dump(full_catalog, f, indent=2, default=str)
    
    logger.info(f"\n[OK] Stage 1 completed: {successful}/15 backbones")
    logger.info(f"Catalog saved: {catalog_file}")
    
    return full_catalog
