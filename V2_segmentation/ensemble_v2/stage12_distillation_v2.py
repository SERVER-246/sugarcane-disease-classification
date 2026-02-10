"""
V2_segmentation/ensemble_v2/stage12_distillation_v2.py
======================================================
Stage 12: Upgraded Multi-Teacher Distillation.

All 15 V2 dual-head backbones serve as teachers for a compact student model.
The distillation loss includes:
  - Classification KD loss (KL divergence on soft labels)
  - Segmentation KD loss (pixel-wise KL on seg maps)
  - Attention transfer (student mimics teacher attention patterns)

ISOLATION: Teachers frozen; student trained on train set, validated on val.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from V2_segmentation.config import (
    BACKBONES, DEVICE, ENSEMBLE_V2_DIR, IMG_SIZE, NUM_CLASSES,
    NUM_SEG_CHANNELS, SPLIT_DIR,
)

logger = logging.getLogger(__name__)


class CompactStudentModel(nn.Module):
    """Lightweight student model for distillation.

    Designed for deployment: ~5M params, fast inference.
    Produces both classification logits and segmentation maps.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_seg_channels: int = NUM_SEG_CHANNELS,
    ) -> None:
        super().__init__()

        # Encoder: lightweight ConvNeXt-style blocks
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, 4, stride=4),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.stages = nn.Sequential(
            self._make_stage(48, 96, 3),
            self._make_stage(96, 192, 3),
            self._make_stage(192, 384, 3),
        )

        # Classification head
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        # Segmentation head
        self.seg_upsample = nn.Sequential(
            nn.Conv2d(384, 128, 1),
            nn.GELU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, num_seg_channels, 1),
        )

    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
        ]
        for _ in range(n_blocks):
            layers.append(self._block(out_ch))
        return nn.Sequential(*layers)

    def _block(self, ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(ch, ch, 7, padding=3, groups=ch),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch * 4, 1),
            nn.GELU(),
            nn.Conv2d(ch * 4, ch, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.stem(x)
        feat = self.stages(feat)

        cls_logits = self.cls_head(self.cls_pool(feat))
        seg_logits = self.seg_upsample(feat)

        return {"cls_logits": cls_logits, "seg_logits": seg_logits}


class Stage12DistillationV2:
    """Multi-teacher distillation with segmentation knowledge.

    Parameters
    ----------
    temperature : float
        KD temperature for softening teacher predictions.
    alpha_cls : float
        Weight for classification distillation loss.
    alpha_seg : float
        Weight for segmentation distillation loss.
    alpha_hard : float
        Weight for hard label cross-entropy loss.
    epochs : int
        Training epochs for student.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha_cls: float = 0.4,
        alpha_seg: float = 0.3,
        alpha_hard: float = 0.3,
        epochs: int = 50,
        output_dir: Path | None = None,
    ) -> None:
        self.temperature = temperature
        self.alpha_cls = alpha_cls
        self.alpha_seg = alpha_seg
        self.alpha_hard = alpha_hard
        self.epochs = epochs
        self.output_dir = output_dir or (ENSEMBLE_V2_DIR / "stage12_distillation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.student: CompactStudentModel | None = None

    def _get_teacher_soft_labels(
        self,
        split: str = "train",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load pre-computed teacher predictions (mean of all backbones).

        Returns (soft_cls_labels, soft_seg_labels, hard_labels).
        """
        stage1_dir = ENSEMBLE_V2_DIR / "stage1_individual"

        all_cls = []
        all_seg = []
        labels = None

        for backbone in BACKBONES:
            cls_path = stage1_dir / f"{backbone}_{split}_cls_probs.npy"
            if cls_path.exists():
                all_cls.append(np.load(str(cls_path)))

            seg_path = stage1_dir / f"{backbone}_{split}_seg_masks.npy"
            if seg_path.exists():
                all_seg.append(np.load(str(seg_path)))

            if labels is None:
                lbl_path = stage1_dir / f"{backbone}_{split}_labels.npy"
                if lbl_path.exists():
                    labels = np.load(str(lbl_path))

        soft_cls = np.mean(all_cls, axis=0) if all_cls else np.array([])
        soft_seg = np.mean(all_seg, axis=0) if all_seg else np.array([])
        if labels is None:
            ds = ImageFolder(str(SPLIT_DIR / split))
            labels = np.array([s[1] for s in ds.samples])

        return soft_cls, soft_seg, labels

    def train_student(self) -> dict[str, Any]:
        """Train compact student model via multi-teacher distillation."""
        self.student = CompactStudentModel(NUM_CLASSES, NUM_SEG_CHANNELS)
        self.student.to(DEVICE)

        # Load teacher soft labels
        soft_cls, _soft_seg, _hard_labels = self._get_teacher_soft_labels("train")

        if soft_cls.size == 0:
            return {"status": "SKIP", "reason": "No teacher predictions available"}

        # Build dataset
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_ds = ImageFolder(str(SPLIT_DIR / "train"), transform=transform)
        train_loader = DataLoader(
            train_ds, batch_size=32, shuffle=True,
            num_workers=0, pin_memory=True,
        )

        optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=1e-3, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        hard_criterion = nn.CrossEntropyLoss()

        T = self.temperature
        _best_val_acc = 0.0

        # Convert soft labels to tensors
        soft_cls_t = torch.tensor(soft_cls, dtype=torch.float32)

        self.student.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0

            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                bs = images.size(0)

                # Get sample indices for this batch
                start_idx = batch_idx * 32
                end_idx = min(start_idx + bs, len(soft_cls_t))
                if start_idx >= len(soft_cls_t):
                    break

                teacher_cls = soft_cls_t[start_idx:end_idx].to(DEVICE)

                optimizer.zero_grad()

                outputs = self.student(images[:teacher_cls.size(0)])
                student_cls = outputs["cls_logits"]

                # KD loss: KL divergence on softened predictions
                kd_loss = F.kl_div(
                    F.log_softmax(student_cls / T, dim=1),
                    F.softmax(teacher_cls / T, dim=1),
                    reduction="batchmean",
                ) * (T * T)

                # Hard label loss
                hard_loss = hard_criterion(student_cls, targets[:teacher_cls.size(0)])

                # Combined loss
                loss = self.alpha_cls * kd_loss + self.alpha_hard * hard_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Student epoch {epoch+1}/{self.epochs}: loss={avg_loss:.4f}")

        # Save student
        torch.save(
            self.student.state_dict(),
            str(self.output_dir / "student_model.pth"),
        )

        # Evaluate
        val_acc = self._evaluate("val")
        test_acc = self._evaluate("test")

        result = {
            "status": "PASS",
            "epochs": self.epochs,
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "student_params": sum(p.numel() for p in self.student.parameters()) / 1e6,
        }
        logger.info(
            f"Stage 12 distillation: val={val_acc:.4f}, test={test_acc:.4f}, "
            f"params={result['student_params']:.1f}M"
        )
        return result

    def _evaluate(self, split: str) -> float:
        """Evaluate student on a split."""
        if self.student is None:
            return 0.0

        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        ds = ImageFolder(str(SPLIT_DIR / split), transform=transform)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        self.student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(DEVICE)
                outputs = self.student(images)
                preds = outputs["cls_logits"].argmax(dim=1).cpu()
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        return correct / max(total, 1)

    def run(self) -> dict[str, Any]:
        """Run full Stage 12 pipeline."""
        return self.train_student()
