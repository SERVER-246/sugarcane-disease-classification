"""
V2_segmentation/ensemble_v2/stage4_feature_fusion_v2.py
========================================================
Stage 4: Feature-Level Fusion (V2 Native).

Three fusion strategies applied to backbone probability features:
  1. ConcatMLPFusion  — concatenate all backbone probs → MLP classifier
  2. AttentionFusion   — project probs to common space → attention-weighted sum
  3. BilinearFusion    — pairwise bilinear interaction of projected features

TRAINING: OOF predictions (leakage-safe).
VALIDATION / TEST: V2 Stage 1 cls_probs.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder

from V2_segmentation.config import (
    BACKBONES,
    DEVICE,
    ENSEMBLE_V2_DIR,
    NUM_CLASSES,
    OOF_DIR,
    SPLIT_DIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Neural-network fusion models
# ---------------------------------------------------------------------------


class ConcatMLPFusion(nn.Module):
    """Concatenate backbone probability vectors → MLP classifier."""

    def __init__(self, num_backbones: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        total_dim = num_backbones * num_classes
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_backbones * num_classes) flattened probs."""
        return self.mlp(x)


class AttentionFusion(nn.Module):
    """Project each backbone's probs to a common space → attention-weighted sum."""

    def __init__(self, num_backbones: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.num_backbones = num_backbones
        self.projections = nn.ModuleList(
            [nn.Linear(num_classes, hidden_dim) for _ in range(num_backbones)]
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_backbones, num_classes) per-backbone probs."""
        projected = [
            proj(x[:, i, :]) for i, proj in enumerate(self.projections)
        ]
        stacked = torch.stack(projected, dim=1)  # (B, K, H)
        attn = F.softmax(self.attention(stacked), dim=1)  # (B, K, 1)
        fused = (stacked * attn).sum(dim=1)  # (B, H)
        return self.classifier(fused)


class BilinearFusion(nn.Module):
    """Bilinear pairwise interaction of projected backbone features."""

    def __init__(self, num_backbones: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(num_classes, hidden_dim) for _ in range(num_backbones)]
        )
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_backbones, num_classes) per-backbone probs."""
        projected = [
            proj(x[:, i, :]) for i, proj in enumerate(self.projections)
        ]
        avg_proj = torch.stack(projected, dim=0).mean(dim=0)  # (B, H)
        bilinear_out = self.bilinear(projected[0], avg_proj)
        return self.classifier(bilinear_out)


# ---------------------------------------------------------------------------
#  Helper: train one fusion model
# ---------------------------------------------------------------------------


def _train_fusion_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    output_dir: Path,
    epochs: int = 50,
    patience: int = 10,
) -> dict[str, Any]:
    """Train a single fusion model; return metrics dict."""

    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state: dict[str, Any] | None = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        correct = total = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / max(total, 1)

        # --- val ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                logits = model(feats)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(total, 1)

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  [{model_name}] Epoch {epoch}/{epochs}: "
                f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(f"  [{model_name}] Early stop at epoch {epoch}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Save model
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_dir / "model.pth")

    return {"model": model, "best_val_acc": best_val_acc}


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    split: str,
    output_dir: Path,
    model_name: str,
) -> dict[str, Any]:
    """Run inference and compute metrics for a split."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(DEVICE)
            logits = model(feats)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    probs_arr = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    preds_arr = probs_arr.argmax(axis=1)

    acc = accuracy_score(labels_arr, preds_arr)
    f1 = f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)
    prec = precision_score(labels_arr, preds_arr, average="weighted", zero_division=0)
    rec = recall_score(labels_arr, preds_arr, average="weighted", zero_division=0)

    # Save predictions (needed by Stage 6)
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(model_dir / f"{split}_predictions.npy"), probs_arr)
    np.save(str(model_dir / f"{split}_labels.npy"), labels_arr)

    return {
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
    }


# ---------------------------------------------------------------------------
#  Stage 4 orchestrator class
# ---------------------------------------------------------------------------


class Stage4FeatureFusionV2:
    """V2-native Feature-Level Fusion using backbone probability features.

    Trains three fusion models on OOF predictions (leakage-safe),
    validates on val set, and evaluates on test set.

    Parameters
    ----------
    epochs : int
        Max training epochs per model (default 50).
    batch_size : int
        Batch size for dataloaders (default 64).
    output_dir : Path | None
        Directory for saved models and predictions.
    """

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 64,
        output_dir: Path | None = None,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = output_dir or (ENSEMBLE_V2_DIR / "stage4_feature_fusion")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage1_dir = ENSEMBLE_V2_DIR / "stage1_individual"

    # ------------------------------------------------------------------ data
    def _load_oof_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Load OOF probabilities → (N_train, K, C) and labels.

        Returns
        -------
        features : (N_train, num_backbones, NUM_CLASSES)
        labels   : (N_train,)
        """
        probs_list: list[np.ndarray] = []
        available_backbones: list[str] = []

        for backbone in BACKBONES:
            path = OOF_DIR / f"{backbone}_oof_probs.npy"
            if path.exists():
                probs_list.append(np.load(str(path)))
                available_backbones.append(backbone)

        if not probs_list:
            raise FileNotFoundError("No OOF predictions found in " + str(OOF_DIR))

        features = np.stack(probs_list, axis=1)  # (N, K, C)

        # Labels from training set folder structure
        train_ds = ImageFolder(str(SPLIT_DIR / "train"))
        labels = np.array([s[1] for s in train_ds.samples])

        logger.info(
            f"Stage 4: Loaded OOF features from {len(available_backbones)} "
            f"backbones → shape {features.shape}"
        )
        return features, labels

    def _load_split_features(
        self, split: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Stage 1 cls_probs for val / test → (N, K, C) and labels."""
        probs_list: list[np.ndarray] = []
        labels: np.ndarray | None = None

        for backbone in BACKBONES:
            path = self.stage1_dir / f"{backbone}_{split}_cls_probs.npy"
            if not path.exists():
                logger.warning(f"Stage 4: missing {path.name}")
                continue
            probs_list.append(np.load(str(path)))
            if labels is None:
                lbl_path = self.stage1_dir / f"{backbone}_{split}_labels.npy"
                if lbl_path.exists():
                    labels = np.load(str(lbl_path))

        if not probs_list:
            raise FileNotFoundError(f"No Stage 1 {split} predictions found")
        if labels is None:
            labels = np.zeros(probs_list[0].shape[0], dtype=np.int64)
        assert labels is not None

        features = np.stack(probs_list, axis=1)
        return features, labels

    def _make_loaders(
        self,
        train_feats: np.ndarray,
        train_labels: np.ndarray,
        val_feats: np.ndarray,
        val_labels: np.ndarray,
        *,
        flatten: bool = False,
    ) -> tuple[DataLoader, DataLoader]:
        """Create train/val DataLoaders.

        If flatten=True, reshape (N, K, C) → (N, K*C) for ConcatMLP.
        """
        if flatten:
            tf = train_feats.reshape(train_feats.shape[0], -1)
            vf = val_feats.reshape(val_feats.shape[0], -1)
        else:
            tf, vf = train_feats, val_feats

        train_ds = TensorDataset(
            torch.FloatTensor(tf), torch.LongTensor(train_labels)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(vf), torch.LongTensor(val_labels)
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0,
        )
        return train_loader, val_loader

    def _make_eval_loader(
        self,
        feats: np.ndarray,
        labels: np.ndarray,
        *,
        flatten: bool = False,
    ) -> DataLoader:
        if flatten:
            feats = feats.reshape(feats.shape[0], -1)
        ds = TensorDataset(torch.FloatTensor(feats), torch.LongTensor(labels))
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, num_workers=0,
        )

    # -------------------------------------------------------------- main run
    def run(self) -> dict[str, Any]:
        """Run Stage 4: train three fusion models, evaluate, return results."""
        logger.info("=" * 60)
        logger.info("  STAGE 4 — Feature-Level Fusion (V2)")
        logger.info("=" * 60)

        # Load data
        train_feats, train_labels = self._load_oof_features()
        val_feats, val_labels = self._load_split_features("val")
        test_feats, test_labels = self._load_split_features("test")

        num_backbones = train_feats.shape[1]
        model_results: dict[str, Any] = {}

        # ── 1. Concat MLP ────────────────────────────────────────────
        logger.info("\n[1/3] ConcatMLPFusion")
        train_ldr, val_ldr = self._make_loaders(
            train_feats, train_labels, val_feats, val_labels, flatten=True,
        )
        model = ConcatMLPFusion(num_backbones, NUM_CLASSES, hidden_dim=512)
        info = _train_fusion_model(
            model, train_ldr, val_ldr, "concat_mlp", self.output_dir,
            epochs=self.epochs,
        )
        # Evaluate on all splits for Stage 6 downstream
        for split, feats, labels in [
            ("train", train_feats, train_labels),
            ("val", val_feats, val_labels),
            ("test", test_feats, test_labels),
        ]:
            ldr = self._make_eval_loader(feats, labels, flatten=True)
            metrics = _evaluate_model(
                info["model"], ldr, split, self.output_dir, "concat_mlp",
            )
            if split == "test":
                model_results["concat_mlp"] = {
                    "val_accuracy": info["best_val_acc"],
                    **metrics,
                }
        del info["model"]
        self._cleanup()

        # ── 2. Attention Fusion ───────────────────────────────────────
        logger.info("\n[2/3] AttentionFusion")
        train_ldr, val_ldr = self._make_loaders(
            train_feats, train_labels, val_feats, val_labels, flatten=False,
        )
        model = AttentionFusion(num_backbones, NUM_CLASSES, hidden_dim=256)
        info = _train_fusion_model(
            model, train_ldr, val_ldr, "attention_fusion", self.output_dir,
            epochs=self.epochs,
        )
        for split, feats, labels in [
            ("train", train_feats, train_labels),
            ("val", val_feats, val_labels),
            ("test", test_feats, test_labels),
        ]:
            ldr = self._make_eval_loader(feats, labels, flatten=False)
            metrics = _evaluate_model(
                info["model"], ldr, split, self.output_dir, "attention_fusion",
            )
            if split == "test":
                model_results["attention_fusion"] = {
                    "val_accuracy": info["best_val_acc"],
                    **metrics,
                }
        del info["model"]
        self._cleanup()

        # ── 3. Bilinear Fusion ────────────────────────────────────────
        logger.info("\n[3/3] BilinearFusion")
        train_ldr, val_ldr = self._make_loaders(
            train_feats, train_labels, val_feats, val_labels, flatten=False,
        )
        model = BilinearFusion(num_backbones, NUM_CLASSES, hidden_dim=256)
        info = _train_fusion_model(
            model, train_ldr, val_ldr, "bilinear_fusion", self.output_dir,
            epochs=self.epochs,
        )
        for split, feats, labels in [
            ("train", train_feats, train_labels),
            ("val", val_feats, val_labels),
            ("test", test_feats, test_labels),
        ]:
            ldr = self._make_eval_loader(feats, labels, flatten=False)
            metrics = _evaluate_model(
                info["model"], ldr, split, self.output_dir, "bilinear_fusion",
            )
            if split == "test":
                model_results["bilinear_fusion"] = {
                    "val_accuracy": info["best_val_acc"],
                    **metrics,
                }
        del info["model"]
        self._cleanup()

        # ── Summary ───────────────────────────────────────────────────
        best_name = max(
            model_results, key=lambda k: model_results[k]["accuracy"],
        )
        best_test_acc = model_results[best_name]["accuracy"]

        # Save summary metrics
        summary = {
            "status": "PASS",
            "fusion_models": model_results,
            "best_model": best_name,
            "best_test_accuracy": best_test_acc,
            "num_backbones": num_backbones,
        }
        with open(self.output_dir / "stage4_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"\nStage 4 complete — best: {best_name} "
            f"(test acc={best_test_acc:.4f})"
        )
        return summary

    @staticmethod
    def _cleanup() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
