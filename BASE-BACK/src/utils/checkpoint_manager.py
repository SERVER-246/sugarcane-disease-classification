"""
Checkpoint Manager - Recovery System for Interrupted Training
Handles saving/resuming training progress when system shuts down
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


# Handle both package imports and direct sys.path imports
try:
    from ..config.settings import BACKBONES, CKPT_DIR
except ImportError:
    from config.settings import BACKBONES, CKPT_DIR


class CheckpointManager:
    """
    Manages training checkpoints and recovery from interruptions.

    Saves progress before training each backbone so training can resume
    if system unexpectedly shuts down.
    """

    def __init__(self, checkpoint_dir: Path | None = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir or CKPT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.recovery_file = self.checkpoint_dir / '.recovery_state.json'
        self.backbone_progress_file = self.checkpoint_dir / '.backbone_progress.json'
        self.kfold_progress_file = self.checkpoint_dir / '.kfold_progress.json'

    def save_backbone_checkpoint(
        self,
        backbone_name: str,
        model_state_dict: dict[str, Any],
        epoch: int,
        metrics: dict[str, float],
        history: dict[str, list[float]],
        optimizer_state: dict | None = None,
        scheduler_state: dict | None = None,
        stage: str = 'head'  # 'head' or 'finetune'
    ) -> str:
        """
        Save checkpoint for a specific backbone at current epoch.

        Args:
            backbone_name: Name of the backbone architecture
            model_state_dict: Model weights
            epoch: Current epoch number
            metrics: Current metrics (acc, loss, etc.)
            history: Training history so far
            optimizer_state: Optimizer state (for resuming)
            scheduler_state: Scheduler state (for resuming)
            stage: Training stage ('head' or 'finetune')

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'backbone_name': backbone_name,
            'model_state_dict': model_state_dict,
            'epoch': epoch,
            'stage': stage,
            'metrics': metrics,
            'history': history,
            'optimizer_state': optimizer_state,
            'scheduler_state': scheduler_state,
            'timestamp': datetime.now().isoformat(),
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        }

        # Save with backbone name and stage
        checkpoint_path = self.checkpoint_dir / f'{backbone_name}_{stage}_epoch{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        return str(checkpoint_path)

    def save_pipeline_state(self, state: dict[str, Any]) -> None:
        """
        Save overall pipeline state for recovery.

        Args:
            state: Pipeline state dictionary containing:
                - completed_backbones: List of completed backbone names
                - current_backbone: Name of backbone being trained
                - current_stage: 'head' or 'finetune' or 'complete'
                - current_epoch: Current epoch
                - results: Accumulated results so far
                - timestamp: When this was saved
        """
        state['timestamp'] = datetime.now().isoformat()
        with open(self.recovery_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_pipeline_state(self) -> dict[str, Any] | None:
        """
        Load pipeline state from last checkpoint.

        Returns:
            Pipeline state dictionary, or None if no checkpoint exists
        """
        if not self.recovery_file.exists():
            return None

        try:
            with open(self.recovery_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def save_kfold_progress(self, backbone_name: str, fold_results: dict[str, Any]) -> None:
        """
        Save K-fold cross-validation progress.

        Args:
            backbone_name: Name of backbone being validated
            fold_results: Results for each fold
        """
        progress = {
            'backbone_name': backbone_name,
            'fold_results': fold_results,
            'timestamp': datetime.now().isoformat()
        }

        kfold_file = self.checkpoint_dir / f'{backbone_name}_kfold_progress.json'
        with open(kfold_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def load_kfold_progress(self, backbone_name: str) -> dict[str, Any] | None:
        """
        Load K-fold progress for a specific backbone.

        Args:
            backbone_name: Name of backbone

        Returns:
            K-fold results, or None if none exist
        """
        kfold_file = self.checkpoint_dir / f'{backbone_name}_kfold_progress.json'
        if not kfold_file.exists():
            return None

        try:
            with open(kfold_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def get_latest_backbone_checkpoint(self, backbone_name: str) -> tuple[str, dict[str, Any]] | None:
        """
        Get the most recent checkpoint for a backbone.

        Args:
            backbone_name: Name of backbone

        Returns:
            Tuple of (checkpoint_path, checkpoint_data), or None if not found
        """
        # Look for checkpoint files matching pattern: {backbone_name}_*.pth
        pattern = f'{backbone_name}_*.pth'
        checkpoints = sorted(self.checkpoint_dir.glob(pattern), reverse=True)

        if not checkpoints:
            return None

        latest = checkpoints[0]
        try:
            data = torch.load(latest, map_location='cpu')
            return (str(latest), data)
        except Exception as e:
            print(f"Error loading checkpoint {latest}: {e}")
            return None

    def get_completed_backbones(self) -> list[str]:
        """
        Get list of backbones that have completed training.

        Returns:
            List of completed backbone names
        """
        state = self.load_pipeline_state()
        if state and 'completed_backbones' in state:
            return state['completed_backbones']
        return []

    def get_next_backbone_to_train(self) -> str | None:
        """
        Get the next backbone that needs training.

        Returns:
            Backbone name, or None if all are complete
        """
        completed = self.get_completed_backbones()

        for backbone_name in BACKBONES:
            if backbone_name not in completed:
                return backbone_name

        return None

    def mark_backbone_complete(self, backbone_name: str, results: dict[str, Any]) -> None:
        """
        Mark a backbone as completed.

        Args:
            backbone_name: Name of completed backbone
            results: Final results for this backbone
        """
        state = self.load_pipeline_state() or {
            'completed_backbones': [],
            'results': {}
        }

        if backbone_name not in state['completed_backbones']:
            state['completed_backbones'].append(backbone_name)

        state['results'][backbone_name] = results
        state['timestamp'] = datetime.now().isoformat()

        self.save_pipeline_state(state)

    def cleanup_old_checkpoints(self, backbone_name: str, keep_latest: int = 3) -> None:
        """
        Remove old checkpoint files, keeping only the latest.

        Args:
            backbone_name: Backbone to clean up
            keep_latest: Number of latest checkpoints to keep
        """
        pattern = f'{backbone_name}_*.pth'
        checkpoints = sorted(self.checkpoint_dir.glob(pattern), reverse=True)

        for checkpoint in checkpoints[keep_latest:]:
            try:
                checkpoint.unlink()
            except Exception as e:
                print(f"Error removing {checkpoint}: {e}")

    def get_recovery_status(self) -> dict[str, Any]:
        """
        Get current recovery status.

        Returns:
            Dictionary with recovery information
        """
        state = self.load_pipeline_state()
        completed = self.get_completed_backbones()
        next_backbone = self.get_next_backbone_to_train()

        return {
            'total_backbones': len(BACKBONES),
            'completed': len(completed),
            'remaining': len(BACKBONES) - len(completed),
            'completed_backbones': completed,
            'next_backbone': next_backbone,
            'current_state': state,
            'recovery_available': state is not None
        }

    def reset_recovery(self) -> None:
        """Reset all recovery checkpoint files."""
        try:
            self.recovery_file.unlink(missing_ok=True)
            self.backbone_progress_file.unlink(missing_ok=True)
            self.kfold_progress_file.unlink(missing_ok=True)
            print("Recovery checkpoint cleared")
        except Exception as e:
            print(f"Error resetting recovery: {e}")

    def export_recovery_summary(self, output_file: Path | None = None) -> str:
        """
        Export recovery status to a human-readable file.

        Args:
            output_file: Where to save the summary

        Returns:
            Path to summary file
        """
        if output_file is None:
            output_file = self.checkpoint_dir / 'recovery_status.txt'

        status = self.get_recovery_status()
        completed = status['completed_backbones']
        remaining = [b for b in BACKBONES if b not in completed]

        summary = f"""
=== TRAINING RECOVERY STATUS ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROGRESS:
  Total Backbones: {status['total_backbones']}
  Completed:      {status['completed']}
  Remaining:      {status['remaining']}

COMPLETED BACKBONES ({len(completed)}):
"""
        for i, backbone in enumerate(completed, 1):
            summary += f"  {i}. {backbone}\n"

        summary += f"\nREMAINING BACKBONES ({len(remaining)}):\n"
        for i, backbone in enumerate(remaining, 1):
            summary += f"  {i}. {backbone}\n"

        if status['next_backbone']:
            summary += f"\nNEXT TO TRAIN: {status['next_backbone']}\n"

        summary += f"\nRECOVERY AVAILABLE: {status['recovery_available']}\n"

        with open(output_file, 'w') as f:
            f.write(summary)

        return str(output_file)


# Convenience function for quick access
def get_checkpoint_manager(checkpoint_dir: Path | None = None) -> CheckpointManager:
    """Get a checkpoint manager instance."""
    return CheckpointManager(checkpoint_dir)
