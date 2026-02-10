"""
V2_segmentation/visualization/training_curves.py
==================================================
Training loss and metric curves for V2 3-phase training.

Shows Phase A / B / C transitions with vertical lines.
Includes seg loss, cls loss, joint loss, mIoU, val accuracy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DPI = 1200


class TrainingCurves:
    """Plot training curves with phase annotations."""

    def __init__(self, output_dir: Path | None = None) -> None:
        from V2_segmentation.config import PLOTS_V2_DIR
        self.output_dir = output_dir or PLOTS_V2_DIR / "training_curves"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_backbone_curves(
        self,
        backbone_name: str,
        history: dict[str, list[float]],
        phase_boundaries: list[int] | None = None,
    ) -> Path:
        """Plot training curves for a single backbone.

        Parameters
        ----------
        backbone_name : str
        history : dict with keys like 'train_loss', 'val_loss', 'seg_loss',
                  'cls_loss', 'mIoU', 'val_acc', etc.
        phase_boundaries : list of epoch indices where phases change.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n_plots = 0
            plot_specs = []

            # Loss subplot
            loss_keys = [k for k in history if "loss" in k.lower()]
            if loss_keys:
                plot_specs.append(("Loss", loss_keys))
                n_plots += 1

            # Accuracy subplot
            acc_keys = [k for k in history if "acc" in k.lower()]
            if acc_keys:
                plot_specs.append(("Accuracy", acc_keys))
                n_plots += 1

            # IoU subplot
            iou_keys = [k for k in history if "iou" in k.lower()]
            if iou_keys:
                plot_specs.append(("mIoU", iou_keys))
                n_plots += 1

            # LR subplot
            lr_keys = [k for k in history if "lr" in k.lower()]
            if lr_keys:
                plot_specs.append(("Learning Rate", lr_keys))
                n_plots += 1

            if n_plots == 0:
                logger.warning(f"No plottable metrics for {backbone_name}")
                return self.output_dir / f"{backbone_name}_EMPTY.txt"

            fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), squeeze=False)
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i / 10) for i in range(10)]

            for idx, (title, keys) in enumerate(plot_specs):
                ax = axes[idx][0]
                for ki, key in enumerate(keys):
                    vals = history[key]
                    ax.plot(vals, label=key, color=colors[ki % len(colors)], linewidth=1.2)

                # Phase boundaries
                if phase_boundaries:
                    phase_labels = ["A→B", "B→C"]
                    for pi, epoch in enumerate(phase_boundaries):
                        label = phase_labels[pi] if pi < len(phase_labels) else f"Phase {pi+1}"
                        ax.axvline(epoch, color="gray", linestyle="--", alpha=0.6, linewidth=0.8)
                        ax.text(
                            epoch, ax.get_ylim()[1] * 0.95, label,
                            fontsize=7, ha="center", color="gray",
                        )

                ax.set_title(title, fontsize=9)
                ax.set_xlabel("Epoch", fontsize=8)
                ax.legend(fontsize=7, loc="best")
                ax.grid(True, alpha=0.3)

            fig.suptitle(f"Training Curves: {backbone_name}", fontsize=11)
            plt.tight_layout()

            path = self.output_dir / f"{backbone_name}_curves.tiff"
            fig.savefig(str(path), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved training curves: {path}")
            return path

        except ImportError:
            logger.warning("matplotlib not available")
            return self.output_dir / f"{backbone_name}_curves_FAILED.txt"

    def plot_all_backbones_comparison(
        self,
        all_histories: dict[str, dict[str, list[float]]],
        metric: str = "val_acc",
        name: str = "all_backbones_comparison",
    ) -> Path:
        """Compare a single metric across all backbones."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            cmap20 = plt.get_cmap("tab20")
            colors = [cmap20(i / max(len(all_histories), 1)) for i in range(len(all_histories))]

            for idx, (bk_name, history) in enumerate(sorted(all_histories.items())):
                if metric in history:
                    ax.plot(
                        history[metric], label=bk_name.replace("Custom", ""),
                        color=colors[idx], linewidth=1.0,
                    )

            ax.set_title(f"{metric} Across All Backbones", fontsize=11)
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel(metric, fontsize=9)
            ax.legend(fontsize=6, loc="best", ncol=2)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            path = self.output_dir / f"{name}.tiff"
            fig.savefig(str(path), dpi=DPI, format="tiff", bbox_inches="tight")
            plt.close(fig)
            return path

        except ImportError:
            return self.output_dir / f"{name}_FAILED.txt"

    def plot_from_json(self, json_path: str | Path) -> Path:
        """Load history from JSON and plot."""
        json_path = Path(json_path)
        with open(json_path) as f:
            data = json.load(f)

        backbone_name = data.get("backbone_name", json_path.stem)
        history = data.get("history", data)
        phases = data.get("phase_boundaries", None)
        return self.plot_backbone_curves(backbone_name, history, phases)
