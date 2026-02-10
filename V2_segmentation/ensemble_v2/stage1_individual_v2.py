"""
V2_segmentation/ensemble_v2/stage1_individual_v2.py
====================================================
Stage 1: Extract individual predictions from all 15 V2 dual-head models.

For each backbone:
  - Load trained V2 checkpoint (DualHeadModel)
  - Run inference on val and test sets
  - Save: class probabilities (N, 13) + seg masks (N, 5, H, W) + embeddings

Outputs: ensembles_v2/stage1_individual/{backbone_name}_{split}.npy
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from V2_segmentation.config import (
    BACKBONES, BACKBONE_PROFILES, CKPT_V2_DIR, DEVICE,
    ENSEMBLE_V2_DIR, IMG_SIZE, NUM_CLASSES, NUM_SEG_CHANNELS, SPLIT_DIR,
)

logger = logging.getLogger(__name__)


class Stage1IndividualV2:
    """Extract individual predictions from all V2 dual-head backbones.

    Each backbone produces:
      - cls_probs: (N, NUM_CLASSES) softmax probabilities
      - seg_masks: (N, NUM_SEG_CHANNELS, H, W) segmentation masks
      - embeddings: (N, D) feature embeddings from backbone
    """

    def __init__(
        self,
        backbones: list[str] | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.backbones = backbones or BACKBONES
        self.output_dir = output_dir or (ENSEMBLE_V2_DIR / "stage1_individual")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _find_v2_checkpoint(self, backbone_name: str) -> Path | None:
        """Find best V2 checkpoint for a backbone."""
        for suffix in ("_v2_best.pth", "_v2_final.pth"):
            path = CKPT_V2_DIR / f"{backbone_name}{suffix}"
            if path.exists():
                return path
        return None

    def _load_v2_model(self, backbone_name: str) -> torch.nn.Module | None:
        """Load a trained V2 DualHeadModel."""
        ckpt_path = self._find_v2_checkpoint(backbone_name)
        if ckpt_path is None:
            logger.warning(f"No V2 checkpoint found for {backbone_name}")
            return None

        from V2_segmentation.models.model_factory import build_v2_model

        profile = BACKBONE_PROFILES.get(backbone_name, {})
        model = build_v2_model(
            backbone_name,
            num_classes=NUM_CLASSES,
            num_seg_channels=NUM_SEG_CHANNELS,
            decoder_channels=profile.get("decoder_channels", 256),
            skip_channels=profile.get("skip_channels", 48),
        )

        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.to(DEVICE).eval()
        return model

    def extract_predictions(
        self,
        backbone_name: str,
        split: str = "val",
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Extract predictions from a single V2 backbone.

        Returns dict with cls_probs, seg_masks arrays and metadata.
        """
        model = self._load_v2_model(backbone_name)
        if model is None:
            return {"status": "SKIP", "backbone": backbone_name}

        profile = BACKBONE_PROFILES.get(backbone_name, {})
        if batch_size is None:
            batch_size = profile.get("batch_size", 16)

        split_dir = SPLIT_DIR / split
        if not split_dir.exists():
            return {"status": "FAIL", "reason": f"Split dir not found: {split_dir}"}

        dataset = ImageFolder(str(split_dir), transform=self._transform)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        all_cls_probs = []
        all_seg_masks = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(DEVICE)
                with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                    outputs = model(images)

                # Classification probabilities
                cls_logits = outputs.get("cls_logits")
                if cls_logits is not None:
                    cls_probs = F.softmax(cls_logits, dim=1)
                    all_cls_probs.append(cls_probs.cpu().numpy())

                # Segmentation masks
                seg_logits = outputs.get("seg_logits")
                if seg_logits is not None:
                    seg_probs = F.softmax(seg_logits, dim=1)
                    # Downsample to save memory
                    seg_probs = F.interpolate(
                        seg_probs, size=(56, 56),
                        mode="bilinear", align_corners=False,
                    )
                    all_seg_masks.append(seg_probs.cpu().numpy())

                all_labels.append(labels.numpy())

        # Concatenate
        cls_probs = np.concatenate(all_cls_probs) if all_cls_probs else np.array([])
        seg_masks = np.concatenate(all_seg_masks) if all_seg_masks else np.array([])
        labels = np.concatenate(all_labels)

        # Save
        np.save(str(self.output_dir / f"{backbone_name}_{split}_cls_probs.npy"), cls_probs)
        if seg_masks.size > 0:
            np.save(str(self.output_dir / f"{backbone_name}_{split}_seg_masks.npy"), seg_masks)
        np.save(str(self.output_dir / f"{backbone_name}_{split}_labels.npy"), labels)

        # Compute accuracy
        if cls_probs.size > 0:
            accuracy = float((cls_probs.argmax(axis=1) == labels).mean())
        else:
            accuracy = 0.0

        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        result = {
            "status": "PASS",
            "backbone": backbone_name,
            "split": split,
            "n_samples": len(dataset),
            "accuracy": round(accuracy, 4),
            "has_seg": seg_masks.size > 0,
        }
        logger.info(f"Stage 1 [{backbone_name}] {split}: acc={accuracy:.4f}")
        return result

    def run(self, splits: list[str] | None = None) -> dict[str, Any]:
        """Run Stage 1 for all backbones on specified splits.

        Default splits: ["val", "test"].
        """
        splits = splits or ["val", "test"]
        results: dict[str, Any] = {"backbones": {}}

        for backbone_name in self.backbones:
            results["backbones"][backbone_name] = {}
            for split in splits:
                r = self.extract_predictions(backbone_name, split)
                results["backbones"][backbone_name][split] = r

        # Summary
        n_pass = sum(
            1 for b in results["backbones"].values()
            for s in b.values() if s.get("status") == "PASS"
        )
        total = len(self.backbones) * len(splits)
        results["summary"] = {
            "total": total,
            "pass": n_pass,
            "status": "PASS" if n_pass == total else "PARTIAL",
        }
        return results
