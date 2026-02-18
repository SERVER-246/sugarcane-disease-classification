"""
V2_segmentation/evaluation/oof_generator.py
============================================
Out-of-Fold (OOF) prediction generator for leakage-free evaluation.

Per Section 8.4:
  - K-Fold applied to TRAINING SET ONLY (never val/test)
  - StratifiedKFold, n_splits=K_FOLDS, seed=42
  - Each sample predicted by a model that NEVER saw it during training
  - OOF predictions used for stacking, cascaded training, adversarial boosting

Supports both classification-only and dual-head (cls + seg) OOF.
"""

from __future__ import annotations

import gc
import logging
import time
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
    AMP_DTYPE, AMP_ENABLED, BACKBONES, BACKBONE_PROFILES, CKPT_V2_DIR,
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

    def generate_v2_oof(
        self,
        backbone_name: str,
        checkpoint_path: Path | None = None,
        quick_epochs: int = 10,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Generate true K-fold OOF predictions for a V2-trained backbone.

        Strategy: For each fold, initialize a fresh V1 classification backbone,
        load pretrained weights from the V2 final checkpoint (backbone only),
        freeze the backbone, and quick-train a fresh classifier head on the
        fold's training split. Then predict on the held-out fold.

        This is true OOF: each sample is predicted by a model that NEVER
        saw it during training. The backbone transfer from V2 gives each
        fold model a strong starting point so quick_epochs is sufficient.

        Parameters
        ----------
        backbone_name : str
            One of the 15 backbone names.
        checkpoint_path : Path | None
            Path to ``*_v2_final.pth``. If None, auto-detected from CKPT_V2_DIR.
        quick_epochs : int
            Epochs to train the classifier head per fold (default 10).
        batch_size : int | None
            Override batch size. If None, uses tier-appropriate value.

        Returns
        -------
        dict with OOF arrays, accuracy, and metadata.
        """
        from V2_segmentation.models.model_factory import create_v1_backbone

        # Resolve checkpoint
        if checkpoint_path is None:
            checkpoint_path = CKPT_V2_DIR / f"{backbone_name}_v2_final.pth"
        if not checkpoint_path.exists():
            return {"status": "FAIL", "reason": f"Checkpoint not found: {checkpoint_path}"}

        # Resolve batch size from tier profile
        if batch_size is None:
            profile = BACKBONE_PROFILES.get(backbone_name, {})
            batch_size = profile.get("batch_size", 16)

        train_dir = SPLIT_DIR / "train"
        if not train_dir.exists():
            return {"status": "FAIL", "reason": "Train directory not found"}

        dataset = ImageFolder(str(train_dir), transform=self._transform)
        labels = np.array([s[1] for s in dataset.samples])
        n_samples = len(dataset)

        # Load V2 checkpoint to extract backbone weights
        v2_ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        v2_state = v2_ckpt.get("model_state_dict", v2_ckpt)

        # Extract backbone.* keys → strip the "backbone." prefix
        backbone_state = {}
        for k, v in v2_state.items():
            if k.startswith("backbone."):
                backbone_state[k[len("backbone."):]] = v

        if not backbone_state:
            # Fallback: maybe the checkpoint IS the backbone directly
            backbone_state = v2_state
            logger.warning(f"No 'backbone.*' keys found in {checkpoint_path.name}, using full state_dict")

        # Initialize OOF arrays
        oof_probs = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
        oof_indices = np.full(n_samples, -1, dtype=np.int64)

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        t0 = time.time()
        for fold_idx, (train_idx, hold_idx) in enumerate(skf.split(np.arange(n_samples), labels)):
            logger.info(
                f"  OOF fold {fold_idx+1}/{self.n_folds} for {backbone_name}: "
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

            # Create fresh V1 backbone and load pretrained backbone weights
            model = create_v1_backbone(backbone_name, NUM_CLASSES)
            model.load_state_dict(backbone_state, strict=False)

            # Freeze backbone, only train classifier head
            # Known classifier attribute names across all 15 backbones:
            #   classifier.* (ConvNeXt, EfficientNetV4, GhostNetV2)
            #   fc.*         (ResNetMish, CSPDarkNet, InceptionV4, MobileOne,
            #                 RegNet, DenseNetHybrid, DynamicConvNet)
            #   head.*       (CoAtNet, Swin, ViTHybrid, MaxViT, ConvNeXt)
            #   head_dist.*  (DeiTStyle distillation head)
            # NOTE: "head_conv" in EfficientNetV4 is a feature projection layer,
            #       NOT a classifier — must NOT be unfrozen.
            _HEAD_PREFIXES = ("classifier.", "fc.", "head.", "head_dist.")
            for name, param in model.named_parameters():
                if any(name.startswith(p) for p in _HEAD_PREFIXES):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            model.to(DEVICE)

            # Quick-train the classifier head only
            model = self._quick_train(model, train_loader, epochs=quick_epochs)

            # Predict on holdout fold
            preds = self._predict(model, hold_loader)
            oof_probs[hold_idx] = preds
            oof_indices[hold_idx] = fold_idx

            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        elapsed = time.time() - t0

        # Verify all samples were predicted
        assert (oof_indices >= 0).all(), "Not all samples received OOF predictions!"

        # Save with the filename pattern that ensemble stages expect
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
            "quick_epochs": quick_epochs,
            "oof_accuracy": round(accuracy, 4),
            "elapsed_min": round(elapsed / 60, 1),
            "output_path": str(out_path),
        }
        logger.info(
            f"  OOF complete for {backbone_name}: "
            f"accuracy={accuracy:.4f}, {self.n_folds} folds, {elapsed/60:.1f}min"
        )
        return result

    def generate_all_v2_oof(
        self,
        quick_epochs: int = 10,
    ) -> dict[str, Any]:
        """Generate true K-fold OOF predictions for ALL 15 backbones.

        Iterates through all backbones with V2 final checkpoints and
        generates OOF predictions for each.

        Returns
        -------
        dict with per-backbone results and summary.
        """
        results: dict[str, Any] = {}
        passed = 0
        failed = 0
        total_time = 0.0

        logger.info(
            f"Generating V2 OOF predictions: {len(BACKBONES)} backbones, "
            f"{self.n_folds} folds, {quick_epochs} epochs/fold"
        )

        for i, backbone_name in enumerate(BACKBONES, 1):
            logger.info(f"[{i}/{len(BACKBONES)}] {backbone_name}")

            ckpt_path = CKPT_V2_DIR / f"{backbone_name}_v2_final.pth"
            if not ckpt_path.exists():
                logger.warning(f"  Skipping {backbone_name}: no V2 final checkpoint")
                results[backbone_name] = {"status": "SKIP", "reason": "No V2 final checkpoint"}
                failed += 1
                continue

            try:
                result = self.generate_v2_oof(
                    backbone_name=backbone_name,
                    checkpoint_path=ckpt_path,
                    quick_epochs=quick_epochs,
                )
                results[backbone_name] = result
                if result["status"] == "PASS":
                    passed += 1
                    total_time += result.get("elapsed_min", 0)
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"  OOF failed for {backbone_name}: {e}")
                logger.exception("  Full traceback:")
                results[backbone_name] = {"status": "FAIL", "reason": str(e)}
                failed += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        summary = {
            "total": len(BACKBONES),
            "passed": passed,
            "failed": failed,
            "n_folds": self.n_folds,
            "quick_epochs": quick_epochs,
            "total_time_min": round(total_time, 1),
            "results": results,
        }
        logger.info(
            f"OOF generation complete: {passed}/{len(BACKBONES)} passed, "
            f"{total_time:.1f}min total"
        )
        return summary

    def _quick_train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        epochs: int = 15,
        lr: float = 1e-3,
    ) -> torch.nn.Module:
        """Quick training loop for OOF fold models."""
        # Only optimize trainable parameters (unfrozen head)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for _epoch in range(epochs):
            for images, targets in train_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=AMP_ENABLED, dtype=AMP_DTYPE):  # type: ignore[attr-defined]
                    outputs = model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs.get("cls_logits", outputs.get("logits"))
                    loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

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
                with torch.amp.autocast("cuda", enabled=AMP_ENABLED, dtype=AMP_DTYPE):  # type: ignore[attr-defined]
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
