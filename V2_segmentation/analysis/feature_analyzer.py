"""
V2_segmentation/analysis/feature_analyzer.py
=============================================
Feature distribution analysis for V1 backbones.

Extracts high-level features from trained backbones, then:
  - Computes per-class feature means / covariance
  - Generates PCA / t-SNE projections
  - Identifies inter-class separability (which classes overlap?)
  - Saves feature embeddings for downstream use
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from V2_segmentation.config import (
    ANALYSIS_DIR, CLASS_NAMES, DEVICE, NUM_CLASSES,
)
from V2_segmentation.models.backbone_adapter import (
    BackboneFeatureExtractor,
)

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Analyze feature-space structure of V1 backbones.

    Parameters
    ----------
    model : nn.Module
        V1 backbone with trained weights.
    backbone_name : str
        Backbone identifier.
    device : torch.device
        Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        backbone_name: str,
        device: torch.device = DEVICE,
    ) -> None:
        self.model = model.to(device).eval()
        self.backbone_name = backbone_name
        self.device = device
        self.extractor = BackboneFeatureExtractor(model, backbone_name)

    def extract_features(
        self,
        dataloader: Any,
        max_samples: int = 2000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract high-level features and labels from a dataloader.

        Parameters
        ----------
        dataloader : DataLoader yielding (images, labels).
        max_samples : int
            Cap on total samples.

        Returns
        -------
        features : (N, C) float32 array — global-avg-pooled high-level features.
        labels : (N,) int array.
        """
        all_feats = []
        all_labels = []
        count = 0

        with torch.no_grad():
            for images, targets in dataloader:
                if count >= max_samples:
                    break
                images = images.to(self.device)
                feats = self.extractor.extract(images)
                hl = feats["high_level"]  # (B, C, H, W)

                # Global average pool to (B, C)
                pooled = hl.mean(dim=[2, 3]).cpu().numpy()
                all_feats.append(pooled)
                all_labels.append(targets.numpy())
                count += pooled.shape[0]

        features = np.concatenate(all_feats, axis=0)[:max_samples]
        labels = np.concatenate(all_labels, axis=0)[:max_samples]
        logger.info(f"  Extracted {features.shape[0]} features, dim={features.shape[1]}")
        return features, labels

    def compute_class_stats(
        self, features: np.ndarray, labels: np.ndarray
    ) -> dict[str, dict[str, np.ndarray]]:
        """Per-class mean and std of features.

        Returns
        -------
        dict mapping class_name → {"mean": (C,), "std": (C,), "count": int}
        """
        stats = {}
        for idx, name in enumerate(CLASS_NAMES):
            mask = labels == idx
            if mask.sum() == 0:
                continue
            class_feats = features[mask]
            stats[name] = {
                "mean": class_feats.mean(axis=0),
                "std": class_feats.std(axis=0),
                "count": int(mask.sum()),
            }
        return stats

    def compute_separability(
        self, features: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Inter-class separability matrix (cosine distance between means).

        Returns
        -------
        separability : (num_classes, num_classes) float32 array.
            Higher = more separable.
        """
        means = []
        for idx in range(NUM_CLASSES):
            mask = labels == idx
            if mask.sum() > 0:
                means.append(features[mask].mean(axis=0))
            else:
                means.append(np.zeros(features.shape[1]))
        means = np.stack(means)  # (num_classes, C)

        # Normalize
        norms = np.linalg.norm(means, axis=1, keepdims=True) + 1e-8
        means_normed = means / norms

        # Cosine similarity → distance
        cosine_sim = means_normed @ means_normed.T
        separability = 1.0 - cosine_sim
        return separability.astype(np.float32)

    def generate_projections(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        method: str = "pca",
        n_components: int = 2,
    ) -> np.ndarray:
        """Dimensionality reduction for visualization.

        Parameters
        ----------
        method : "pca" or "tsne".
        n_components : 2 or 3.

        Returns
        -------
        projections : (N, n_components) array.
        """
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=n_components, random_state=42,
                perplexity=min(30, features.shape[0] - 1),
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        projections = reducer.fit_transform(features)
        logger.info(f"  {method.upper()} projection: {projections.shape}")
        return projections

    def save_analysis(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        output_dir: Path | None = None,
    ) -> Path:
        """Run full analysis and save results.

        Saves:
          - features.npz (features + labels)
          - class_stats.npz
          - separability.npy
          - pca_projection.npy
          - Plots (PCA scatter, separability heatmap)

        Returns
        -------
        output_dir : Path where files were saved.
        """
        output_dir = Path(output_dir or ANALYSIS_DIR / self.backbone_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw
        np.savez(output_dir / "features.npz", features=features, labels=labels)

        # Class stats
        stats = self.compute_class_stats(features, labels)
        stats_flat: dict[str, np.ndarray] = {}
        for k, v in stats.items():
            stats_flat[f"{k}_mean"] = v["mean"]
            stats_flat[f"{k}_std"] = v["std"]
        np.savez(str(output_dir / "class_stats.npz"), **stats_flat)  # type: ignore[arg-type]

        # Separability
        sep = self.compute_separability(features, labels)
        np.save(output_dir / "separability.npy", sep)

        # PCA
        pca_proj = self.generate_projections(features, labels, "pca")
        np.save(output_dir / "pca_projection.npy", pca_proj)

        # Plots
        self._save_plots(pca_proj, labels, sep, output_dir)

        logger.info(f"  Analysis saved to {output_dir}")
        return output_dir

    def _save_plots(
        self,
        pca_proj: np.ndarray,
        labels: np.ndarray,
        separability: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Generate and save visualization plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # PCA scatter
            fig, ax = plt.subplots(figsize=(12, 10))
            for idx, name in enumerate(CLASS_NAMES):
                mask = labels == idx
                if mask.sum() > 0:
                    ax.scatter(
                        pca_proj[mask, 0], pca_proj[mask, 1],
                        label=name, alpha=0.6, s=20,
                    )
            ax.legend(fontsize=8, loc="best")
            ax.set_title(f"{self.backbone_name} — PCA Feature Space")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.tight_layout()
            plt.savefig(str(output_dir / "pca_scatter.png"), dpi=150)
            plt.close(fig)

            # Separability heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(separability, cmap="RdYlGn", vmin=0, vmax=2)
            ax.set_xticks(range(NUM_CLASSES))
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(NUM_CLASSES))
            ax.set_yticklabels(CLASS_NAMES, fontsize=7)
            ax.set_title(f"{self.backbone_name} — Class Separability")
            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(str(output_dir / "separability_heatmap.png"), dpi=150)
            plt.close(fig)

        except ImportError:
            logger.warning("  matplotlib/sklearn not available — skipping plots")

    def cleanup(self) -> None:
        """Remove hooks."""
        self.extractor.remove_hooks()
