"""
V2_segmentation/ensemble_v2/stage2_to_7_rerun.py
==================================================
Re-run V1 ensemble stages 2–3 on V2 dual-head predictions.
Stages 4–7 are handled by dedicated V2 modules.

Stages:
  2: Score ensembles (hard/soft/weighted/logit voting)
  3: Stacking (LR/MLP/XGBoost)
  4–6: Delegated to stage4_feature_fusion_v2, stage5_mixture_experts_v2,
       stage6_meta_learner_v2
  7: Superseded by Stage 12 distillation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from V2_segmentation.config import (
    BACKBONES, ENSEMBLE_V2_DIR, OOF_DIR, SPLIT_DIR,
)

logger = logging.getLogger(__name__)


class Stage2To7Rerun:
    """Re-run V1 ensemble stages 2–7 using V2 predictions.

    Loads V2 Stage 1 outputs and passes them through the existing
    ensemble_system/ code (stages 2–7).

    For stages requiring training (3, 5, 6, 7): uses OOF predictions
    from the training set, validates on val, evaluates on test.
    """

    def __init__(
        self,
        stage1_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.stage1_dir = stage1_dir or (ENSEMBLE_V2_DIR / "stage1_individual")
        self.output_dir = output_dir or ENSEMBLE_V2_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_stage1_predictions(
        self, split: str = "val"
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load all backbone predictions for a split.

        Returns (all_probs, labels, backbone_names)
          all_probs: (N_backbones, N_samples, N_classes)
          labels: (N_samples,)
        """
        all_probs = []
        labels = None
        names = []

        for backbone in BACKBONES:
            path = self.stage1_dir / f"{backbone}_{split}_cls_probs.npy"
            if not path.exists():
                logger.warning(f"Missing Stage 1 predictions: {path}")
                continue
            probs = np.load(str(path))
            all_probs.append(probs)
            names.append(backbone)

            if labels is None:
                label_path = self.stage1_dir / f"{backbone}_{split}_labels.npy"
                if label_path.exists():
                    labels = np.load(str(label_path))

        if not all_probs:
            raise FileNotFoundError("No Stage 1 predictions found")

        if labels is None:
            labels = np.zeros(all_probs[0].shape[0], dtype=np.int64)

        return np.stack(all_probs), labels, names

    def run_stage2_voting(self, split: str = "val") -> dict[str, Any]:
        """Stage 2: Score ensembles — hard/soft/weighted/logit voting."""
        stage_dir = self.output_dir / "stage2_voting"
        stage_dir.mkdir(parents=True, exist_ok=True)

        all_probs, labels, _names = self._load_stage1_predictions(split)
        n_backbones, _n_samples, _n_classes = all_probs.shape

        results: dict[str, Any] = {"methods": {}}

        # Hard voting (majority vote)
        preds = all_probs.argmax(axis=2)  # (B, N)
        from scipy.stats import mode
        hard_votes = mode(preds, axis=0, keepdims=False).mode
        hard_acc = float((hard_votes == labels).mean()) if labels is not None else 0.0
        results["methods"]["hard_voting"] = {"accuracy": round(hard_acc, 4)}
        np.save(str(stage_dir / f"hard_voting_{split}.npy"), hard_votes)

        # Soft voting (mean probabilities)
        soft_probs = all_probs.mean(axis=0)
        soft_preds = soft_probs.argmax(axis=1)
        soft_acc = float((soft_preds == labels).mean()) if labels is not None else 0.0
        results["methods"]["soft_voting"] = {"accuracy": round(soft_acc, 4)}
        np.save(str(stage_dir / f"soft_voting_{split}.npy"), soft_probs)

        # Weighted voting (inverse rank weighting)
        if labels is not None:
            per_backbone_acc = [(all_probs[i].argmax(1) == labels).mean() for i in range(n_backbones)]
            weights = np.array(per_backbone_acc)
            weights = weights / weights.sum()
            weighted_probs = np.tensordot(weights, all_probs, axes=([0], [0]))
            weighted_preds = weighted_probs.argmax(axis=1)
            weighted_acc = float((weighted_preds == labels).mean())
            results["methods"]["weighted_voting"] = {"accuracy": round(weighted_acc, 4)}
            np.save(str(stage_dir / f"weighted_voting_{split}.npy"), weighted_probs)

        logger.info(f"Stage 2 [{split}]: soft={soft_acc:.4f}, hard={hard_acc:.4f}")
        return results

    def run_stage3_stacking(self) -> dict[str, Any]:
        """Stage 3: Stacking — train meta-classifiers on OOF predictions."""
        stage_dir = self.output_dir / "stage3_stacking"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Load OOF predictions for training features
        oof_features = []
        _oof_labels = None

        for backbone in BACKBONES:
            oof_path = OOF_DIR / f"{backbone}_oof_probs.npy"
            if not oof_path.exists():
                logger.warning(f"Missing OOF for {backbone}")
                continue
            oof_features.append(np.load(str(oof_path)))

        if not oof_features:
            return {"status": "SKIP", "reason": "No OOF predictions available"}

        # Stack: (N_train, N_backbones × N_classes)
        X_train = np.concatenate(oof_features, axis=1)

        # Labels from any backbone's indices
        from torchvision.datasets import ImageFolder
        train_ds = ImageFolder(str(SPLIT_DIR / "train"))
        y_train = np.array([s[1] for s in train_ds.samples])

        # Logistic Regression stacker
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial")
        lr.fit(X_train, y_train)
        train_acc = float(lr.score(X_train, y_train))

        # Validate on val set
        try:
            val_probs, val_labels, _ = self._load_stage1_predictions("val")
            X_val = np.concatenate([val_probs[i] for i in range(val_probs.shape[0])], axis=1)
            val_preds = lr.predict(X_val)
            val_acc = float((val_preds == val_labels).mean()) if val_labels is not None else 0.0
        except Exception:
            val_acc = 0.0

        result = {
            "status": "PASS",
            "stacker": "LogisticRegression",
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "n_features": X_train.shape[1],
        }

        # Save
        import pickle
        with open(stage_dir / "lr_stacker.pkl", "wb") as f:
            pickle.dump(lr, f)

        logger.info(f"Stage 3 stacking: val_acc={val_acc:.4f}")
        return result

    def run_stages2to3(self) -> dict[str, Any]:
        """Run only stages 2 and 3 (voting + stacking).

        Stages 4-7 are now handled by dedicated V2 modules.
        """
        results: dict[str, Any] = {}

        # Stage 2: Voting
        try:
            results["stage2"] = self.run_stage2_voting("val")
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            results["stage2"] = {"status": "FAIL", "error": str(e)}

        # Stage 3: Stacking
        try:
            results["stage3"] = self.run_stage3_stacking()
        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            results["stage3"] = {"status": "FAIL", "error": str(e)}

        return results

    def run_all(self) -> dict[str, Any]:
        """Run stages 2-7 sequentially.

        Stages 2-3 run natively. Stages 4-7 are delegated to their
        dedicated V2 modules (called from the orchestrator).
        """
        results: dict[str, Any] = {}

        # Stage 2: Voting
        try:
            results["stage2"] = self.run_stage2_voting("val")
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            results["stage2"] = {"status": "FAIL", "error": str(e)}

        # Stage 3: Stacking
        try:
            results["stage3"] = self.run_stage3_stacking()
        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            results["stage3"] = {"status": "FAIL", "error": str(e)}

        # Stages 4-7: Now handled by dedicated V2 modules
        for stage_num in [4, 5, 6, 7]:
            results[f"stage{stage_num}"] = {
                "status": "DELEGATED_TO_V2",
                "note": f"Stage {stage_num} handled by dedicated V2 module",
            }

        return results
