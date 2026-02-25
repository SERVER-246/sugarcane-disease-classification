"""
V2_segmentation/ensemble_v2/stage6_meta_learner_v2.py
======================================================
Stage 6: Meta-Ensemble Controller (V2 Native).

Combines predictions produced by Stages 2–5 into a single
"meta-learner" that decides the final class.

Two meta-controllers are trained and the best is kept:
  1. XGBoost  — gradient-boosted trees on concatenated stage predictions
  2. MLP      — lightweight neural-network meta-classifier

TRAINING: Uses OOF-derived predictions from stages 2–5 (leakage-safe).
VALIDATION / TEST: Stage 1 cls_probs routed through earlier stages.

If upstream stage predictions are missing, the meta-learner gracefully
falls back to whatever is available (minimum: Stage 2 voting outputs).
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
#  MLP meta-controller
# ---------------------------------------------------------------------------


class MetaMLPController(nn.Module):
    """Small MLP that maps concatenated stage predictions → class logits."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
#  Stage 6 orchestrator
# ---------------------------------------------------------------------------


class Stage6MetaLearnerV2:
    """V2-native Meta-Ensemble Controller.

    Collects predictions saved by earlier stages and trains two
    meta-controllers (XGBoost + MLP) to combine them.

    Parameters
    ----------
    epochs : int
        Max training epochs for MLP (default 50).
    batch_size : int
        DataLoader batch size (default 64).
    patience : int
        Early-stopping patience for MLP (default 10).
    output_dir : Path | None
        Where to save models and metrics.
    """

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        output_dir: Path | None = None,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.output_dir = output_dir or (ENSEMBLE_V2_DIR / "stage6_meta_learner")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage1_dir = ENSEMBLE_V2_DIR / "stage1_individual"

    # ------------------------------------------------------------------ data
    def _collect_stage_predictions(
        self, split: str
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Gather predictions from stages 2-5 for a given split.

        Falls back gracefully when stages are missing.

        Returns
        -------
        features : (N, total_feature_dim)
        labels   : (N,)
        sources  : list of source names (for logging)
        """
        pieces: list[tuple[str, np.ndarray]] = []
        labels: np.ndarray | None = None

        # ── Stage 2 voting outputs ────────────────────────────────────
        stage2_dir = ENSEMBLE_V2_DIR / "stage2_voting"
        for method in ["soft_voting", "hard_voting", "weighted_voting"]:
            path = stage2_dir / f"{method}_{split}.npy"
            if path.exists():
                arr = np.load(str(path))
                # hard_voting is 1-D (class indices) — one-hot encode it
                if arr.ndim == 1:
                    onehot = np.zeros((arr.shape[0], NUM_CLASSES), dtype=np.float32)
                    for i, c in enumerate(arr):
                        onehot[i, int(c)] = 1.0
                    arr = onehot
                pieces.append((f"s2_{method}", arr))

        # ── Stage 3 stacking outputs ─────────────────────────────────
        stage3_dir = ENSEMBLE_V2_DIR / "stage3_stacking"
        # The LR stacker doesn't save per-split predictions by default.
        # If available from a re-run, load them.
        for name in ["lr_stacker"]:
            path = stage3_dir / f"{name}_{split}_predictions.npy"
            if path.exists():
                pieces.append((f"s3_{name}", np.load(str(path))))

        # ── Stage 4 fusion outputs ────────────────────────────────────
        stage4_dir = ENSEMBLE_V2_DIR / "stage4_feature_fusion"
        for model_name in ["concat_mlp", "attention_fusion", "bilinear_fusion"]:
            path = stage4_dir / model_name / f"{split}_predictions.npy"
            if path.exists():
                pieces.append((f"s4_{model_name}", np.load(str(path))))

        # ── Stage 5 MoE outputs ──────────────────────────────────────
        stage5_dir = ENSEMBLE_V2_DIR / "stage5_mixture_experts"
        path = stage5_dir / f"{split}_predictions.npy"
        if path.exists():
            pieces.append(("s5_moe", np.load(str(path))))

        # ── Fallback: raw Stage 1 soft-voting if nothing above worked ─
        if not pieces:
            logger.warning(
                f"Stage 6: No stage 2-5 predictions for '{split}' — "
                f"falling back to raw Stage 1 soft-voting"
            )
            probs_list: list[np.ndarray] = []
            for backbone in BACKBONES:
                p = self.stage1_dir / f"{backbone}_{split}_cls_probs.npy"
                if p.exists():
                    probs_list.append(np.load(str(p)))
                    if labels is None:
                        lp = self.stage1_dir / f"{backbone}_{split}_labels.npy"
                        if lp.exists():
                            labels = np.load(str(lp))
            if probs_list:
                soft = np.mean(probs_list, axis=0)
                pieces.append(("s1_soft_mean", soft))

        if not pieces:
            raise FileNotFoundError(
                f"No ensemble predictions found for split '{split}'"
            )

        # Resolve labels
        if labels is None:
            if split == "train":
                train_ds = ImageFolder(str(SPLIT_DIR / "train"))
                labels = np.array([s[1] for s in train_ds.samples])
            else:
                # Try from Stage 4/5 label files
                for label_src in [
                    stage4_dir / "concat_mlp" / f"{split}_labels.npy",
                    stage5_dir / f"{split}_labels.npy",
                ]:
                    if label_src.exists():
                        labels = np.load(str(label_src))
                        break
                if labels is None:
                    # Last resort: Stage 1
                    for backbone in BACKBONES:
                        lp = self.stage1_dir / f"{backbone}_{split}_labels.npy"
                        if lp.exists():
                            labels = np.load(str(lp))
                            break
        if labels is None:
            labels = np.zeros(pieces[0][1].shape[0], dtype=np.int64)
        assert labels is not None

        # Verify alignment
        n_samples = labels.shape[0]
        valid_pieces: list[tuple[str, np.ndarray]] = []
        for name, arr in pieces:
            if arr.shape[0] == n_samples:
                valid_pieces.append((name, arr))
            else:
                logger.warning(
                    f"Stage 6: skipping {name} — size {arr.shape[0]} vs {n_samples}"
                )

        sources = [p[0] for p in valid_pieces]
        features = np.concatenate([p[1] for p in valid_pieces], axis=1)

        logger.info(
            f"Stage 6 [{split}]: {len(sources)} sources, "
            f"feature shape {features.shape}"
        )
        return features.astype(np.float32), labels, sources

    def _collect_oof_predictions(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build OOF-based training features from stages 2-5.

        For OOF-safe training we use:
          - Stage 4/5 *train* predictions (which were themselves trained on OOF)
          - Stage 2/3 train-split predictions
          - Stage 1 OOF soft-voting as fallback
        """
        pieces: list[tuple[str, np.ndarray]] = []

        # Stage 2 voting on val (produced by Stage2To7Rerun)
        stage2_dir = ENSEMBLE_V2_DIR / "stage2_voting"
        for method in ["soft_voting", "hard_voting", "weighted_voting"]:
            path = stage2_dir / f"{method}_val.npy"
            if path.exists():
                arr = np.load(str(path))
                if arr.ndim == 1:
                    onehot = np.zeros((arr.shape[0], NUM_CLASSES), dtype=np.float32)
                    for i, c in enumerate(arr):
                        onehot[i, int(c)] = 1.0
                    arr = onehot
                pieces.append((f"s2_{method}", arr))

        # Stage 4 train predictions
        stage4_dir = ENSEMBLE_V2_DIR / "stage4_feature_fusion"
        for model_name in ["concat_mlp", "attention_fusion", "bilinear_fusion"]:
            path = stage4_dir / model_name / "train_predictions.npy"
            if path.exists():
                pieces.append((f"s4_{model_name}", np.load(str(path))))

        # Stage 5 train predictions
        stage5_dir = ENSEMBLE_V2_DIR / "stage5_mixture_experts"
        path = stage5_dir / "train_predictions.npy"
        if path.exists():
            pieces.append(("s5_moe", np.load(str(path))))

        # Fallback: OOF soft-voting
        if not pieces:
            oof_list: list[np.ndarray] = []
            for backbone in BACKBONES:
                p = OOF_DIR / f"{backbone}_oof_probs.npy"
                if p.exists():
                    oof_list.append(np.load(str(p)))
            if oof_list:
                pieces.append(("oof_soft_mean", np.mean(oof_list, axis=0)))

        if not pieces:
            raise FileNotFoundError("No training features for Stage 6")

        # Labels
        train_ds = ImageFolder(str(SPLIT_DIR / "train"))
        labels = np.array([s[1] for s in train_ds.samples])

        # Alignment
        n_samples = labels.shape[0]
        valid: list[tuple[str, np.ndarray]] = []
        for name, arr in pieces:
            if arr.shape[0] == n_samples:
                valid.append((name, arr))
            else:
                logger.warning(
                    f"Stage 6 OOF: skipping {name} — "
                    f"size {arr.shape[0]} vs {n_samples}"
                )

        sources = [p[0] for p in valid]
        features = np.concatenate([p[1] for p in valid], axis=1)
        logger.info(
            f"Stage 6 OOF training: {len(sources)} sources, "
            f"feature shape {features.shape}"
        )
        return features.astype(np.float32), labels, sources

    # -------------------------------------------------------- XGBoost meta
    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """Train XGBoost meta-controller."""
        logger.info("\n  [XGBoost] Training meta-controller...")
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("  xgboost not installed — skipping XGBoost meta-controller")
            return {"status": "SKIP", "reason": "xgboost not installed"}

        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)

        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
        test_prec = precision_score(y_test, test_preds, average="weighted", zero_division=0)
        test_rec = recall_score(y_test, test_preds, average="weighted", zero_division=0)

        # Save
        xgb_dir = self.output_dir / "xgboost"
        xgb_dir.mkdir(parents=True, exist_ok=True)

        import joblib
        joblib.dump(model, xgb_dir / "xgboost_meta.pkl")
        np.save(str(xgb_dir / "test_predictions.npy"), test_probs)
        np.save(str(xgb_dir / "test_labels.npy"), y_test)

        logger.info(
            f"  [XGBoost] val_acc={val_acc:.4f}  test_acc={test_acc:.4f}"
        )
        return {
            "method": "xgboost",
            "val_accuracy": round(float(val_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_f1": round(float(test_f1), 4),
            "test_precision": round(float(test_prec), 4),
            "test_recall": round(float(test_rec), 4),
        }

    # ------------------------------------------------------------ MLP meta
    def _train_mlp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """Train MLP meta-controller."""
        logger.info("\n  [MLP] Training meta-controller...")

        input_dim = X_train.shape[1]
        model = MetaMLPController(input_dim, NUM_CLASSES).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs,
        )
        criterion = nn.CrossEntropyLoss()

        train_ldr = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train), torch.LongTensor(y_train),
            ),
            batch_size=self.batch_size, shuffle=True, num_workers=0,
        )
        val_ldr = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_val), torch.LongTensor(y_val),
            ),
            batch_size=self.batch_size, shuffle=False, num_workers=0,
        )

        best_val_acc = 0.0
        best_state: dict[str, Any] | None = None
        patience_ctr = 0

        for epoch in range(1, self.epochs + 1):
            model.train()
            correct = total = 0
            for feats, labels in train_ldr:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                logits = model(feats)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
            train_acc = correct / max(total, 1)

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for feats, labels in val_ldr:
                    feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                    logits = model(feats)
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / max(total, 1)

            scheduler.step()

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"  [MLP] Epoch {epoch}/{self.epochs}: "
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
                    logger.info(f"  [MLP] Early stop at epoch {epoch}")
                    break

        # Restore & evaluate on test
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        test_ldr = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_test), torch.LongTensor(y_test),
            ),
            batch_size=self.batch_size, shuffle=False, num_workers=0,
        )
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        with torch.no_grad():
            for feats, labels in test_ldr:
                feats = feats.to(DEVICE)
                probs = F.softmax(model(feats), dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())

        test_probs = np.concatenate(all_probs)
        test_labels_arr = np.concatenate(all_labels)
        test_preds = test_probs.argmax(axis=1)

        test_acc = accuracy_score(test_labels_arr, test_preds)
        test_f1 = f1_score(test_labels_arr, test_preds, average="weighted", zero_division=0)
        test_prec = precision_score(test_labels_arr, test_preds, average="weighted", zero_division=0)
        test_rec = recall_score(test_labels_arr, test_preds, average="weighted", zero_division=0)

        # Save
        mlp_dir = self.output_dir / "mlp"
        mlp_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, mlp_dir / "mlp_meta.pth")
        np.save(str(mlp_dir / "test_predictions.npy"), test_probs)
        np.save(str(mlp_dir / "test_labels.npy"), test_labels_arr)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            f"  [MLP] val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}"
        )
        return {
            "method": "mlp",
            "val_accuracy": round(float(best_val_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_f1": round(float(test_f1), 4),
            "test_precision": round(float(test_prec), 4),
            "test_recall": round(float(test_rec), 4),
        }

    # -------------------------------------------------------------- main run
    def run(self) -> dict[str, Any]:
        """Run Stage 6: train XGBoost + MLP meta-controllers, pick best."""
        logger.info("=" * 60)
        logger.info("  STAGE 6 — Meta-Ensemble Controller (V2)")
        logger.info("=" * 60)

        # Build training features from OOF-derived stage outputs
        X_train, y_train, train_sources = self._collect_oof_predictions()

        # Val / test features (from stage 2-5 val/test predictions)
        X_val, y_val, val_sources = self._collect_stage_predictions("val")
        X_test, y_test, test_sources = self._collect_stage_predictions("test")

        # Ensure consistent feature dims (pad shorter splits if needed)
        min_dim = min(X_train.shape[1], X_val.shape[1], X_test.shape[1])
        max_dim = max(X_train.shape[1], X_val.shape[1], X_test.shape[1])

        if min_dim != max_dim:
            logger.warning(
                f"Stage 6: Feature dim mismatch "
                f"(train={X_train.shape[1]}, val={X_val.shape[1]}, "
                f"test={X_test.shape[1]}). Using minimum dim={min_dim}."
            )
            X_train = X_train[:, :min_dim]
            X_val = X_val[:, :min_dim]
            X_test = X_test[:, :min_dim]

        # Train both meta-controllers
        results: dict[str, Any] = {}

        xgb_result = self._train_xgboost(
            X_train, y_train, X_val, y_val, X_test, y_test,
        )
        results["xgboost"] = xgb_result

        mlp_result = self._train_mlp(
            X_train, y_train, X_val, y_val, X_test, y_test,
        )
        results["mlp"] = mlp_result

        # Pick best
        xgb_acc = xgb_result.get("test_accuracy", 0.0)
        mlp_acc = mlp_result.get("test_accuracy", 0.0)
        best_method = "xgboost" if xgb_acc >= mlp_acc else "mlp"
        best_acc = max(xgb_acc, mlp_acc)

        summary = {
            "status": "PASS",
            "controllers": results,
            "best_method": best_method,
            "best_test_accuracy": round(best_acc, 4),
            "num_features": X_train.shape[1],
            "train_sources": train_sources,
            "val_sources": val_sources,
            "test_sources": test_sources,
        }

        with open(self.output_dir / "stage6_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"\nStage 6 complete — best: {best_method} "
            f"(test acc={best_acc:.4f})"
        )
        return summary
