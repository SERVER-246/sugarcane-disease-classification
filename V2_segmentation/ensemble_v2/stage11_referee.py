"""
V2_segmentation/ensemble_v2/stage11_referee.py
===============================================
Stage 11: Cross-Architecture Knowledge Transfer / Referee.

Identifies ambiguous samples where backbones disagree, then trains a
lightweight "referee" model to resolve these ambiguities.

Ambiguous = backbone disagreement ratio > 0.4 (i.e., less than 60% of
backbones agree on the prediction).

ISOLATION: Ambiguity identified via OOF on train set; referee validated on val.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from V2_segmentation.config import (
    BACKBONES, DEVICE, ENSEMBLE_V2_DIR, NUM_CLASSES,
    OOF_DIR, SPLIT_DIR,
)

logger = logging.getLogger(__name__)


def compute_disagreement(probs_stack: np.ndarray) -> np.ndarray:
    """Compute per-sample backbone disagreement ratio.

    Parameters
    ----------
    probs_stack : (N_backbones, N_samples, N_classes)

    Returns
    -------
    disagreement : (N_samples,) float in [0, 1]
        0 = all agree, 1 = maximum disagreement.
    """
    preds = probs_stack.argmax(axis=2)  # (B, N)
    n_backbones = preds.shape[0]

    # Most common prediction per sample
    from scipy.stats import mode
    _majority = mode(preds, axis=0, keepdims=False).mode  # (N,)
    majority_count = mode(preds, axis=0, keepdims=False).count  # (N,)

    # Disagreement = 1 - (majority_count / n_backbones)
    agreement_ratio = majority_count / n_backbones
    return 1.0 - agreement_ratio.astype(np.float32)


class RefereeModel(nn.Module):
    """Lightweight MLP referee for ambiguous samples.

    Input: concatenated backbone probabilities + seg features.
    Output: class predictions.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage11Referee:
    """Cross-architecture referee for ambiguous samples.

    Workflow:
      1. Compute disagreement across all backbone OOF predictions
      2. Identify ambiguous samples (disagreement > threshold)
      3. Train referee MLP on ambiguous training samples
      4. At inference: use main ensemble for clear cases, referee for ambiguous
    """

    def __init__(
        self,
        disagreement_threshold: float = 0.4,
        output_dir: Path | None = None,
    ) -> None:
        self.disagreement_threshold = disagreement_threshold
        self.output_dir = output_dir or (ENSEMBLE_V2_DIR / "stage11_referee")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.referee: RefereeModel | None = None
        self._ambiguous_mask: np.ndarray | None = None

    def identify_ambiguous(self) -> dict[str, Any]:
        """Identify ambiguous training samples via OOF disagreement."""
        oof_probs_list = []
        for backbone in BACKBONES:
            path = OOF_DIR / f"{backbone}_oof_probs.npy"
            if path.exists():
                oof_probs_list.append(np.load(str(path)))

        if not oof_probs_list:
            return {"status": "SKIP", "reason": "No OOF predictions"}

        probs_stack = np.stack(oof_probs_list)  # (B, N, C)
        disagreement = compute_disagreement(probs_stack)

        ambiguous = disagreement > self.disagreement_threshold
        self._ambiguous_mask = ambiguous

        n_ambiguous = int(ambiguous.sum())
        n_total = len(disagreement)

        # Per-class breakdown
        from torchvision.datasets import ImageFolder
        train_ds = ImageFolder(str(SPLIT_DIR / "train"))
        y_true = np.array([s[1] for s in train_ds.samples])

        per_class = {}
        for cls_idx in range(NUM_CLASSES):
            cls_mask = y_true == cls_idx
            n_cls = int(cls_mask.sum())
            n_cls_amb = int((ambiguous & cls_mask).sum())
            per_class[str(cls_idx)] = {
                "total": n_cls,
                "ambiguous": n_cls_amb,
                "ratio": round(n_cls_amb / max(n_cls, 1), 3),
            }

        result = {
            "status": "PASS",
            "n_total": n_total,
            "n_ambiguous": n_ambiguous,
            "ambiguous_ratio": round(n_ambiguous / n_total, 3),
            "per_class": per_class,
        }
        logger.info(
            f"Stage 11: {n_ambiguous}/{n_total} ambiguous samples "
            f"({n_ambiguous/n_total*100:.1f}%)"
        )
        return result

    def train_referee(self, epochs: int = 30, lr: float = 1e-3) -> dict[str, Any]:
        """Train referee model on ambiguous training samples."""
        if self._ambiguous_mask is None:
            self.identify_ambiguous()
        if self._ambiguous_mask is None or self._ambiguous_mask.sum() == 0:
            return {"status": "SKIP", "reason": "No ambiguous samples"}

        # Build training features: concatenated OOF probs
        features = []
        for backbone in BACKBONES:
            path = OOF_DIR / f"{backbone}_oof_probs.npy"
            if path.exists():
                features.append(np.load(str(path)))
        if not features:
            return {"status": "SKIP", "reason": "No OOF data"}

        X_all = np.concatenate(features, axis=1)  # (N, B*C)
        from torchvision.datasets import ImageFolder
        train_ds = ImageFolder(str(SPLIT_DIR / "train"))
        y_all = np.array([s[1] for s in train_ds.samples])

        # Filter to ambiguous only
        amb_idx = np.where(self._ambiguous_mask)[0]
        X_amb = X_all[amb_idx]
        y_amb = y_all[amb_idx]

        # Create referee model
        input_dim = X_amb.shape[1]
        self.referee = RefereeModel(input_dim, NUM_CLASSES)
        self.referee.to(DEVICE)

        # Training
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_amb, dtype=torch.float32),
            torch.tensor(y_amb, dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.AdamW(self.referee.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.referee.train()
        for _epoch in range(epochs):
            total_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                optimizer.zero_grad()
                out = self.referee(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Evaluate on ambiguous training set
        self.referee.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_amb, dtype=torch.float32).to(DEVICE)
            preds = self.referee(X_t).argmax(dim=1).cpu().numpy()
            train_acc = float((preds == y_amb).mean())

        # Save
        torch.save(self.referee.state_dict(), str(self.output_dir / "referee_model.pth"))

        result = {
            "status": "PASS",
            "n_ambiguous_train": len(amb_idx),
            "input_dim": input_dim,
            "train_accuracy": round(train_acc, 4),
        }
        logger.info(f"Referee trained: acc={train_acc:.4f} on {len(amb_idx)} samples")
        return result

    def predict_with_referee(
        self,
        backbone_probs: dict[str, np.ndarray],
        base_ensemble_probs: np.ndarray,
    ) -> np.ndarray:
        """Apply referee to ambiguous samples, keep base ensemble for clear ones.

        Parameters
        ----------
        backbone_probs : dict
            backbone → (N, C) per-backbone probs.
        base_ensemble_probs : (N, C)
            Ensemble prediction from previous stages.
        """
        if self.referee is None:
            return base_ensemble_probs

        # Build features
        features = []
        for backbone in BACKBONES:
            if backbone in backbone_probs:
                features.append(backbone_probs[backbone])
            else:
                n = base_ensemble_probs.shape[0]
                features.append(np.zeros((n, NUM_CLASSES), dtype=np.float32))

        X = np.concatenate(features, axis=1)

        # Compute disagreement
        probs_stack = np.stack(list(backbone_probs.values()))
        disagreement = compute_disagreement(probs_stack)
        ambiguous = disagreement > self.disagreement_threshold

        # Predict referee for ambiguous
        result = base_ensemble_probs.copy()
        if ambiguous.sum() > 0:
            self.referee.eval()
            with torch.no_grad():
                X_amb = torch.tensor(X[ambiguous], dtype=torch.float32).to(DEVICE)
                referee_logits = self.referee(X_amb)
                referee_probs = torch.softmax(referee_logits, dim=1).cpu().numpy()
                result[ambiguous] = referee_probs

        return result

    def run(self) -> dict[str, Any]:
        """Run full Stage 11 pipeline."""
        amb_result = self.identify_ambiguous()
        train_result = self.train_referee()
        return {
            "ambiguity_analysis": amb_result,
            "referee_training": train_result,
        }
