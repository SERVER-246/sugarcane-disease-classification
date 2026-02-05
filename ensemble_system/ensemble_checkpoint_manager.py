"""
Ensemble Checkpoint Manager
Handles recovery from interrupted ensemble training
Mirrors BASE-BACK/src/utils/checkpoint_manager.py structure
"""

import sys
from pathlib import Path


# Add BASE-BACK to path BEFORE other imports
BASE_BACK_PATH = Path(__file__).parent.parent / 'BASE-BACK' / 'src'
if str(BASE_BACK_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_BACK_PATH))

import json
from datetime import datetime
from typing import Any

import torch
from config.settings import CKPT_DIR


class EnsembleCheckpointManager:
    """
    Manages ensemble training checkpoints and recovery.
    Similar to CheckpointManager but for ensemble-specific operations.
    """

    def __init__(self, checkpoint_dir: Path | None = None):
        """Initialize ensemble checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir or (CKPT_DIR / 'ensembles'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.recovery_file = self.checkpoint_dir / '.ensemble_recovery_state.json'
        self.strategy_progress_file = self.checkpoint_dir / '.strategy_progress.json'

    def save_ensemble_checkpoint(
        self,
        strategy_name: str,
        ensemble_state: dict[str, Any],
        metrics: dict[str, float]
    ) -> str:
        """
        Save checkpoint for a specific ensemble strategy.
        
        Args:
            strategy_name: Name of ensemble strategy
            ensemble_state: Ensemble model state
            metrics: Current metrics
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'strategy_name': strategy_name,
            'ensemble_state': ensemble_state,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        checkpoint_path = self.checkpoint_dir / f'{strategy_name}_ensemble.pth'
        torch.save(checkpoint, checkpoint_path)

        return str(checkpoint_path)

    def save_pipeline_state(self, state: dict[str, Any]) -> None:
        """Save overall ensemble pipeline state for recovery."""
        state['timestamp'] = datetime.now().isoformat()
        with open(self.recovery_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_pipeline_state(self) -> dict[str, Any] | None:
        """Load pipeline state from last checkpoint."""
        if not self.recovery_file.exists():
            return None

        try:
            with open(self.recovery_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def mark_strategy_complete(self, strategy_name: str, results: dict[str, Any]) -> None:
        """Mark an ensemble strategy as completed."""
        state = self.load_pipeline_state() or {
            'completed_strategies': [],
            'completed_stages': [],
            'results': {},
            'stage_results': {}
        }

        if strategy_name not in state['completed_strategies']:
            state['completed_strategies'].append(strategy_name)

        state['results'][strategy_name] = results
        state['timestamp'] = datetime.now().isoformat()

        self.save_pipeline_state(state)

    def mark_stage_complete(self, stage_name: str, results: dict[str, Any]) -> None:
        """Mark a pipeline stage as completed"""
        state = self.load_pipeline_state() or {
            'completed_strategies': [],
            'completed_stages': [],
            'results': {},
            'stage_results': {}
        }

        if stage_name not in state['completed_stages']:
            state['completed_stages'].append(stage_name)

        state['stage_results'][stage_name] = results
        state['timestamp'] = datetime.now().isoformat()

        self.save_pipeline_state(state)

    def is_stage_complete(self, stage_name: str) -> bool:
        """Check if a stage is completed"""
        state = self.load_pipeline_state()
        if state and 'completed_stages' in state:
            return stage_name in state['completed_stages']
        return False

    def get_stage_results(self, stage_name: str) -> dict[str, Any]:
        """Get results for a specific stage"""
        state = self.load_pipeline_state()
        if state and 'stage_results' in state:
            return state['stage_results'].get(stage_name, {})
        return {}

    def get_completed_strategies(self) -> list[str]:
        """Get list of completed ensemble strategies."""
        state = self.load_pipeline_state()
        if state and 'completed_strategies' in state:
            return state['completed_strategies']
        return []

    def get_recovery_status(self) -> dict[str, Any]:
        """Get current recovery status."""
        state = self.load_pipeline_state()
        completed_strategies = self.get_completed_strategies()
        completed_stages = state.get('completed_stages', []) if state else []

        all_strategies = ['voting', 'weighted', 'stacking', 'averaging']
        remaining_strategies = [s for s in all_strategies if s not in completed_strategies]

        all_stages = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6', 'stage7']
        remaining_stages = [s for s in all_stages if s not in completed_stages]

        return {
            'total_strategies': len(all_strategies),
            'completed': len(completed_strategies),
            'remaining': len(remaining_strategies),
            'completed_strategies': completed_strategies,
            'remaining_strategies': remaining_strategies,
            'completed_stages': completed_stages,
            'remaining_stages': remaining_stages,
            'current_state': state,
            'recovery_available': state is not None
        }

    def reset_recovery(self) -> None:
        """Reset all recovery checkpoint files."""
        try:
            self.recovery_file.unlink(missing_ok=True)
            self.strategy_progress_file.unlink(missing_ok=True)
            print("Ensemble recovery checkpoint cleared")
        except Exception as e:
            print(f"Error resetting recovery: {e}")

    def export_recovery_summary(self, output_file: Path | None = None) -> str:
        """Export recovery status to a human-readable file."""
        if output_file is None:
            output_file = self.checkpoint_dir / 'ensemble_recovery_status.txt'

        status = self.get_recovery_status()
        completed = status['completed_strategies']
        remaining = status['remaining_strategies']

        summary = f"""
=== ENSEMBLE TRAINING RECOVERY STATUS ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROGRESS:
  Total Strategies: {status['total_strategies']}
  Completed:        {status['completed']}
  Remaining:        {status['remaining']}
  
COMPLETED STRATEGIES ({len(completed)}):
"""
        for i, strategy in enumerate(completed, 1):
            summary += f"  {i}. {strategy}\n"

        summary += f"\nREMAINING STRATEGIES ({len(remaining)}):\n"
        for i, strategy in enumerate(remaining, 1):
            summary += f"  {i}. {strategy}\n"

        summary += f"\nRECOVERY AVAILABLE: {status['recovery_available']}\n"

        with open(output_file, 'w') as f:
            f.write(summary)

        return str(output_file)


def get_ensemble_checkpoint_manager(checkpoint_dir: Path | None = None) -> EnsembleCheckpointManager:
    """Get an ensemble checkpoint manager instance."""
    return EnsembleCheckpointManager(checkpoint_dir)
