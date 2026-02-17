"""
V2_segmentation/evaluation/oof_generator.py
============================================
Out-of-Fold (OOF) prediction generator for leakage-free evaluation.

Per Section 8.4:
  - K-Fold applied to TRAINING SET ONLY (never val/test)
  - StratifiedKFold, n_splits=5, seed=42
  - Each sample predicted by a model that NEVER saw it during training
  - OOF predictions used for stacking, cascaded training, adversarial boosting

Supports both classification-only and dual-head (cls + seg) OOF.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from V2_segmentation.config import (
    DEVICE, IMG_SIZE, K_FOLDS, NUM_CLASSES,
    OOF_DIR, RUN_SEED, SPLIT_DIR,
)

logger = logging.getLogger(__name__)


class OOFGenerator:
    """Generate out-of-fold predictions for the training set.

    Workflow:
      1. Load training dataset
      2. For each fold k in 1..K:
           - Train model on folds != k
           - Predict on fold k
      3. Concatenate → complete OOF predictions for all training samples

    Supports two modes:
      - ``v1``: Load pre-trained V1 checkpoints and just generate predictions
      - ``v2``: Train V2 dual-head models per fold (expensive but proper)

    Parameters
    ----------
    n_folds : int
        Number of cross-validation folds (default 5).
    seed : int
        Random seed for fold splits.
    output_dir : Path
        Where to save OOF predictions.
    """

    def __init__(
        self,
        n_folds: int = K_FOLDS,
        seed: int = RUN_SEED,
        output_dir: Path | None = None,
    ) -> None:
        self.n_folds = n_folds
        self.seed = seed
        self.output_dir = output_dir or OOF_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def generate_v1_oof(
        self,
        backbone_name: str,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Generate OOF predictions using V1 backbone with K-fold.

        This trains V1 models from scratch per fold. For pre-trained V1
        inference-only OOF, use ``generate_inference_oof()``.

        Returns dict with OOF arrays and metadata.
        """
        from V2_segmentation.models.model_factory import create_v1_backbone

        train_dir = SPLIT_DIR / "train"
        if not train_dir.exists():
            return {"status": "FAIL", "reason": "Train directory not found"}

        dataset = ImageFolder(str(train_dir), transform=self._transform)
        labels = np.array([s[1] for s in dataset.samples])
        n_samples = len(dataset)

        # Initialize OOF arrays
        oof_probs = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
        oof_indices = np.full(n_samples, -1, dtype=np.int64)

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        for fold_idx, (train_idx, hold_idx) in enumerate(skf.split(np.arange(n_samples), labels)):
            logger.info(
                f"OOF fold {fold_idx+1}/{self.n_folds} for {backbone_name}: "
                f"train={len(train_idx)}, holdout={len(hold_idx)}"
            )

            # Create fold data loaders
            train_subset = Subset(dataset, train_idx.tolist())
            hold_subset = Subset(dataset, hold_idx.tolist())

            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True,
            )
            hold_loader = DataLoader(
                hold_subset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
            )

            # Create and train model for this fold
            model = create_v1_backbone(backbone_name, NUM_CLASSES)
            model.to(DEVICE)

            # Quick training: reduced epochs for OOF (not full training)
            model = self._quick_train(model, train_loader, epochs=15)

            # Predict on holdout
            preds = self._predict(model, hold_loader)
            oof_probs[hold_idx] = preds
            oof_indices[hold_idx] = fold_idx

            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Verify all samples were predicted
        assert (oof_indices >= 0).all(), "Not all samples received OOF predictions!"

        # Save
        out_path = self.output_dir / f"{backbone_name}_oof_probs.npy"
        np.save(str(out_path), oof_probs)

        idx_path = self.output_dir / f"{backbone_name}_oof_fold_indices.npy"
        np.save(str(idx_path), oof_indices)

        # Compute accuracy
        oof_preds = oof_probs.argmax(axis=1)
        accuracy = float((oof_preds == labels).mean())

        result = {
            "status": "PASS",
            "backbone": backbone_name,
            "n_samples": n_samples,
            "n_folds": self.n_folds,
            "oof_accuracy": round(accuracy, 4),
            "output_path": str(out_path),
        }
        logger.info(f"OOF complete for {backbone_name}: accuracy={accuracy:.4f}")
        return result

    def generate_inference_oof(
        self,
        backbone_name: str,
        checkpoint_path: Path,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Generate OOF-like predictions from a single pre-trained model.

        WARNING: This is NOT true OOF — the model saw all training data.
        Use only for quick debugging / stacking feature extraction.
        """
        from V2_segmentation.models.model_factory import create_v1_backbone, load_v1_weights

        train_dir = SPLIT_DIR / "train"
        dataset = ImageFolder(str(train_dir), transform=self._transform)
        labels = np.array([s[1] for s in dataset.samples])

        model = create_v1_backbone(backbone_name, NUM_CLASSES)
        load_v1_weights(model, checkpoint_path, strict=False)
        model.to(DEVICE)

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        probs = self._predict(model, loader)

        out_path = self.output_dir / f"{backbone_name}_inference_probs.npy"
        np.save(str(out_path), probs)

        preds = probs.argmax(axis=1)
        accuracy = float((preds == labels).mean())

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "PASS",
            "backbone": backbone_name,
            "n_samples": len(dataset),
            "accuracy": round(accuracy, 4),
            "warning": "NOT true OOF — model saw all training data",
            "output_path": str(out_path),
        }

    def _quick_train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        epochs: int = 15,
        lr: float = 1e-3,
    ) -> torch.nn.Module:
        """Quick training loop for OOF fold models."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=False)  # type: ignore[attr-defined]

        model.train()
        for _epoch in range(epochs):
            for images, targets in train_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # type: ignore[attr-defined]
                    outputs = model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs.get("cls_logits", outputs.get("logits"))
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        model.eval()
        return model

    def _predict(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
    ) -> np.ndarray:
        """Generate softmax predictions for a dataset."""
        all_probs = []
        model.eval()
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(DEVICE)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # type: ignore[attr-defined]
                    outputs = model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs.get("cls_logits", outputs.get("logits"))
                    assert outputs is not None
                    probs = F.softmax(outputs, dim=1)
                all_probs.append(probs.float().cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def get_fold_splits(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Get the fold split indices (for reproducibility verification)."""
        train_dir = SPLIT_DIR / "train"
        dataset = ImageFolder(str(train_dir))
        labels = np.array([s[1] for s in dataset.samples])
        n_samples = len(dataset)

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )
        return list(skf.split(np.arange(n_samples), labels))
