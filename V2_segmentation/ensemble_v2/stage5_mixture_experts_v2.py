"""
V2_segmentation/ensemble_v2/stage5_mixture_experts_v2.py
=========================================================
Stage 5: Mixture of Experts (V2 Native).

A learned gating network routes each sample to the Top-K most relevant
backbone "experts", producing a weighted combination of their predictions.

Architecture:
  GatingNetwork  — MLP that scores each backbone for a given input
  MixtureOfExperts — Top-K sparse gating over backbone predictions

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
#  Neural-network components
# ---------------------------------------------------------------------------


class GatingNetwork(nn.Module):
    """Learns per-sample routing weights over K backbone experts.

    Input : flattened expert probabilities (B, K*C)
    Output: gate_scores (B, K), top_k_indices (B, top_k), top_k_scores (B, top_k)
    """

    def __init__(self, num_experts: int, num_classes: int, top_k: int = 5):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        input_dim = num_experts * num_classes

        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_experts),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(x)
        gate_scores = F.softmax(gate_logits, dim=1)
        top_k_vals, top_k_idx = torch.topk(gate_scores, self.top_k, dim=1)
        top_k_scores = top_k_vals / top_k_vals.sum(dim=1, keepdim=True)
        return gate_scores, top_k_idx, top_k_scores


class MixtureOfExperts(nn.Module):
    """Sparse MoE that weights backbone expert predictions via a gating network.

    Forward input: expert_probs (B, K, C) — per-backbone class probabilities.
    Output:        final_probs  (B, C)    — gated ensemble prediction.
    """

    def __init__(self, num_experts: int, num_classes: int, top_k: int = 5):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.top_k = min(top_k, num_experts)

        self.gating = GatingNetwork(num_experts, num_classes, self.top_k)

    def forward(self, expert_probs: torch.Tensor) -> torch.Tensor:
        B = expert_probs.size(0)
        flat = expert_probs.reshape(B, -1)  # (B, K*C)
        _gate_all, top_k_idx, top_k_scores = self.gating(flat)

        # Gather top-K expert probs  → (B, top_k, C)
        # expand indices for gather
        idx_exp = top_k_idx.unsqueeze(-1).expand(-1, -1, self.num_classes)
        selected = torch.gather(expert_probs, 1, idx_exp)  # (B, top_k, C)

        # Weighted sum
        weights = top_k_scores.unsqueeze(-1)  # (B, top_k, 1)
        final = (selected * weights).sum(dim=1)  # (B, C)
        return final


# ---------------------------------------------------------------------------
#  Stage 5 orchestrator class
# ---------------------------------------------------------------------------


class Stage5MixtureExpertsV2:
    """V2-native Mixture-of-Experts ensemble.

    Trains a gating network on OOF predictions (leakage-safe),
    evaluates on val and test Stage 1 cls_probs.

    Parameters
    ----------
    top_k : int
        Number of experts selected per sample (default 5).
    epochs : int
        Max training epochs (default 30).
    batch_size : int
        DataLoader batch size (default 64).
    patience : int
        Early-stopping patience (default 10).
    output_dir : Path | None
        Directory for saved models and predictions.
    """

    def __init__(
        self,
        top_k: int = 5,
        epochs: int = 30,
        batch_size: int = 64,
        patience: int = 10,
        output_dir: Path | None = None,
    ) -> None:
        self.top_k = top_k
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.output_dir = output_dir or (ENSEMBLE_V2_DIR / "stage5_mixture_experts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage1_dir = ENSEMBLE_V2_DIR / "stage1_individual"

    # -------------------------------------------------------------- data I/O
    def _load_oof(self) -> tuple[np.ndarray, np.ndarray]:
        """Load OOF probs → (N_train, K, C) and labels."""
        probs: list[np.ndarray] = []
        for backbone in BACKBONES:
            path = OOF_DIR / f"{backbone}_oof_probs.npy"
            if path.exists():
                probs.append(np.load(str(path)))
        if not probs:
            raise FileNotFoundError("No OOF predictions in " + str(OOF_DIR))

        features = np.stack(probs, axis=1)
        train_ds = ImageFolder(str(SPLIT_DIR / "train"))
        labels = np.array([s[1] for s in train_ds.samples])
        logger.info(f"Stage 5: OOF shape {features.shape}")
        return features, labels

    def _load_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Load Stage 1 cls_probs for val / test."""
        probs: list[np.ndarray] = []
        labels: np.ndarray | None = None
        for backbone in BACKBONES:
            p = self.stage1_dir / f"{backbone}_{split}_cls_probs.npy"
            if not p.exists():
                continue
            probs.append(np.load(str(p)))
            if labels is None:
                lp = self.stage1_dir / f"{backbone}_{split}_labels.npy"
                if lp.exists():
                    labels = np.load(str(lp))
        if not probs:
            raise FileNotFoundError(f"No Stage 1 {split} predictions found")
        if labels is None:
            labels = np.zeros(probs[0].shape[0], dtype=np.int64)
        assert labels is not None
        return np.stack(probs, axis=1), labels

    def _make_loader(
        self, feats: np.ndarray, labels: np.ndarray, shuffle: bool
    ) -> DataLoader:
        ds = TensorDataset(torch.FloatTensor(feats), torch.LongTensor(labels))
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=0,
        )

    # -------------------------------------------------------------- training
    def _train(
        self,
        model: MixtureOfExperts,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """Train gating network; return best model + metrics."""
        model = model.to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state: dict[str, Any] | None = None
        patience_ctr = 0

        for epoch in range(1, self.epochs + 1):
            # ---- train ----
            model.train()
            correct = total = 0
            for expert_probs, labels in train_loader:
                expert_probs = expert_probs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                out = model(expert_probs)  # (B, C) probs
                logits = torch.log(out + 1e-8)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
            train_acc = correct / max(total, 1)

            # ---- val ----
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for expert_probs, labels in val_loader:
                    expert_probs = expert_probs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    out = model(expert_probs)
                    correct += (out.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / max(total, 1)

            scheduler.step()

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"  [MoE] Epoch {epoch}/{self.epochs}: "
                    f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    logger.info(f"  [MoE] Early stop at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # Save model
        torch.save(best_state, self.output_dir / "moe_model.pth")

        return {"model": model, "best_val_acc": best_val_acc}

    # ------------------------------------------------------------ evaluation
    def _evaluate(
        self,
        model: MixtureOfExperts,
        loader: DataLoader,
        split: str,
    ) -> dict[str, Any]:
        """Run inference, compute metrics, save predictions."""
        model.eval()
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for expert_probs, labels in loader:
                expert_probs = expert_probs.to(DEVICE)
                out = model(expert_probs)
                all_probs.append(out.cpu().numpy())
                all_labels.append(labels.numpy())

        probs_arr = np.concatenate(all_probs)
        labels_arr = np.concatenate(all_labels)
        preds_arr = probs_arr.argmax(axis=1)

        acc = accuracy_score(labels_arr, preds_arr)
        f1 = f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)
        prec = precision_score(labels_arr, preds_arr, average="weighted", zero_division=0)
        rec = recall_score(labels_arr, preds_arr, average="weighted", zero_division=0)

        # Save predictions for Stage 6
        np.save(str(self.output_dir / f"{split}_predictions.npy"), probs_arr)
        np.save(str(self.output_dir / f"{split}_labels.npy"), labels_arr)

        return {
            "accuracy": round(float(acc), 4),
            "f1": round(float(f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
        }

    # -------------------------------------------------------------- main run
    def run(self) -> dict[str, Any]:
        """Run Stage 5: train MoE, evaluate on val/test, return results."""
        logger.info("=" * 60)
        logger.info("  STAGE 5 — Mixture of Experts (V2)")
        logger.info("=" * 60)

        # Data
        train_feats, train_labels = self._load_oof()
        val_feats, val_labels = self._load_split("val")
        test_feats, test_labels = self._load_split("test")

        num_experts = train_feats.shape[1]
        top_k = min(self.top_k, num_experts)

        train_ldr = self._make_loader(train_feats, train_labels, shuffle=True)
        val_ldr = self._make_loader(val_feats, val_labels, shuffle=False)
        test_ldr = self._make_loader(test_feats, test_labels, shuffle=False)

        # Build & train
        moe = MixtureOfExperts(num_experts, NUM_CLASSES, top_k)
        info = self._train(moe, train_ldr, val_ldr)
        model = info["model"]

        # Evaluate all splits (train predictions needed by Stage 6)
        split_metrics: dict[str, Any] = {}
        for split, ldr in [
            ("train", train_ldr),
            ("val", val_ldr),
            ("test", test_ldr),
        ]:
            split_metrics[split] = self._evaluate(model, ldr, split)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = {
            "status": "PASS",
            "num_experts": num_experts,
            "top_k": top_k,
            "best_val_accuracy": round(info["best_val_acc"], 4),
            "val": split_metrics["val"],
            "test": split_metrics["test"],
        }

        # Save summary
        with open(self.output_dir / "stage5_metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        logger.info(
            f"\nStage 5 complete — MoE (top-{top_k}): "
            f"val={split_metrics['val']['accuracy']:.4f}, "
            f"test={split_metrics['test']['accuracy']:.4f}"
        )
        return result
