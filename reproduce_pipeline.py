#!/usr/bin/env python3
"""
================================================================================
SUGARCANE DISEASE CLASSIFICATION - COMPLETE REPRODUCIBILITY SCRIPT
================================================================================

This script reproduces the entire pipeline from scratch:
  Phase 1: Train 15 custom backbone models (via run_pipeline.py -> BASE-BACK/src/main.py)
  Phase 2: Run 7-stage 15-COIN ensemble pipeline (via ensemble_system/run_15coin_pipeline.py)

Architecture:
  - run_pipeline.py: Entry point for backbone training
  - BASE-BACK/src/main.py: Modular training orchestrator
  - BASE-BACK/src/models/: 15 backbone architecture definitions
  - ensemble_system/: 7-stage ensemble pipeline

Author: DBT Project Team
Version: 1.0.1
Last Updated: December 2025

USAGE:
------
    # Full pipeline (WARNING: ~55 hours on RTX 4500 Ada)
    python reproduce_pipeline.py --mode full
    
    # Quick test (reduced epochs, single backbone)
    python reproduce_pipeline.py --mode quick_test
    
    # Phase 1 only (backbone training)
    python reproduce_pipeline.py --mode backbones_only
    
    # Phase 2 only (ensemble, requires trained backbones)
    python reproduce_pipeline.py --mode ensemble_only
    
    # Interactive mode (prompts for all options)
    python reproduce_pipeline.py --mode interactive

REQUIREMENTS:
-------------
    - Python 3.8+
    - PyTorch 2.0+ with CUDA
    - NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
    - 50GB+ disk space
    - Dataset: Images organized in folders by class name

================================================================================
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Paths (will be set during setup)
    'base_dir': None,
    'data_dir': None,
    'split_dir': None,

    # Training hyperparameters
    'img_size': 224,
    'batch_size': 32,
    'epochs_head': 40,
    'epochs_finetune': 25,
    'patience_head': 5,
    'patience_ft': 5,

    # Quick test settings
    'quick_epochs_head': 5,
    'quick_epochs_finetune': 3,
    'quick_batch_size': 16,

    # Dataset split ratios
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,

    # Ensemble settings
    'ensemble_epochs': 100,
    'distillation_epochs': 100,

    # Backbones
    'backbones': [
        'CustomConvNeXt', 'CustomEfficientNetV4', 'CustomGhostNetV2',
        'CustomResNetMish', 'CustomCSPDarkNet', 'CustomInceptionV4',
        'CustomViTHybrid', 'CustomSwinTransformer', 'CustomCoAtNet',
        'CustomRegNet', 'CustomDenseNetHybrid', 'CustomDeiTStyle',
        'CustomMaxViT', 'CustomMobileOne', 'CustomDynamicConvNet'
    ],
}

REQUIRED_PACKAGES = [
    'torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn',
    'xgboost', 'matplotlib', 'seaborn', 'tqdm', 'pillow'
]

OPTIONAL_PACKAGES = [
    'onnx', 'onnxruntime', 'tensorrt', 'coremltools'
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")


def print_step(step: int, total: int, text: str):
    """Print step indicator."""
    print(f"{Colors.CYAN}[{step}/{total}]{Colors.ENDC} {text}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}Cool {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}Hot {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}x {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}i{text}{Colors.ENDC}")


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no answer."""
    default_str = "Y/n" if default else "y/N"
    while True:
        response = input(f"{question} [{default_str}]: ").strip().lower()
        if response == '':
            return default
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'")


def prompt_path(prompt: str, must_exist: bool = True, default: str = None) -> Path:
    """Prompt user for a path."""
    while True:
        default_str = f" [{default}]" if default else ""
        response = input(f"{prompt}{default_str}: ").strip()

        if response == '' and default:
            response = default

        if not response:
            print("Please enter a path")
            continue

        path = Path(response)

        if must_exist and not path.exists():
            print(f"Path does not exist: {path}")
            if prompt_yes_no("Create it?", default=False):
                path.mkdir(parents=True, exist_ok=True)
                return path
            continue

        return path


def estimate_time(mode: str, num_backbones: int = 15) -> str:
    """Estimate execution time based on mode."""
    times = {
        'full': f"~{3.5 * num_backbones + 5:.1f} hours",
        'backbones_only': f"~{3.5 * num_backbones:.1f} hours",
        'ensemble_only': "~5 hours",
        'quick_test': "~30 minutes",
    }
    return times.get(mode, "Unknown")


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def check_python_version() -> bool:
    """Check Python version is 3.8+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_cuda() -> dict[str, Any]:
    """Check CUDA availability and GPU info."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            cuda_version = torch.version.cuda

            print_success(f"CUDA available: {cuda_version}")
            print_success(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")

            return {
                'available': True,
                'version': cuda_version,
                'gpu_count': gpu_count,
                'gpu_name': gpu_name,
                'gpu_memory_gb': gpu_memory
            }
        else:
            print_warning("CUDA not available - training will be VERY slow on CPU")
            return {'available': False}

    except ImportError:
        print_error("PyTorch not installed")
        return {'available': False, 'error': 'PyTorch not installed'}


def check_packages() -> dict[str, bool]:
    """Check required and optional packages."""
    results = {}

    print_info("Checking required packages...")
    missing_required = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
            results[package] = True
        except ImportError:
            results[package] = False
            missing_required.append(package)

    if missing_required:
        print_error(f"Missing required packages: {', '.join(missing_required)}")
    else:
        print_success("All required packages installed")

    print_info("Checking optional packages...")
    for package in OPTIONAL_PACKAGES:
        try:
            __import__(package)
            results[package] = True
            print_success(f"  {package}: installed")
        except ImportError:
            results[package] = False
            print_warning(f"  {package}: not installed (optional)")

    return results


def install_packages(packages: list[str]) -> bool:
    """Install missing packages using pip."""
    print_info(f"Installing: {', '.join(packages)}")

    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', *packages
        ])
        print_success("Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install packages: {e}")
        return False


def setup_environment(config: dict) -> bool:
    """Set up the complete environment."""
    print_header("ENVIRONMENT SETUP")

    # Check Python
    print_step(1, 4, "Checking Python version...")
    if not check_python_version():
        return False

    # Check packages
    print_step(2, 4, "Checking packages...")
    packages = check_packages()

    missing = [p for p, installed in packages.items()
               if not installed and p in REQUIRED_PACKAGES]

    if missing:
        if prompt_yes_no(f"Install missing packages ({', '.join(missing)})?"):
            if not install_packages(missing):
                return False
        else:
            print_error("Cannot proceed without required packages")
            return False

    # Check CUDA
    print_step(3, 4, "Checking CUDA/GPU...")
    cuda_info = check_cuda()

    if not cuda_info.get('available', False):
        if not prompt_yes_no("Continue without GPU? (Training will be very slow)"):
            return False

    # Check disk space
    print_step(4, 4, "Checking disk space...")
    base_dir = Path(config.get('base_dir', '.'))

    try:
        # Get free space
        if sys.platform == 'win32':
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(str(base_dir)), None, None, ctypes.pointer(free_bytes)
            )
            free_gb = free_bytes.value / 1e9
        else:
            import os
            statvfs = os.statvfs(base_dir)
            free_gb = (statvfs.f_frsize * statvfs.f_bavail) / 1e9

        if free_gb < 50:
            print_warning(f"Low disk space: {free_gb:.1f} GB (50+ GB recommended)")
        else:
            print_success(f"Disk space: {free_gb:.1f} GB available")

    except Exception as e:
        print_warning(f"Could not check disk space: {e}")

    print_success("Environment setup complete!")
    return True


# =============================================================================
# DATASET SETUP
# =============================================================================

def validate_dataset(data_dir: Path) -> dict[str, Any]:
    """Validate dataset structure and count images."""
    if not data_dir.exists():
        return {'valid': False, 'error': 'Directory does not exist'}

    classes = []
    total_images = 0
    class_counts = {}

    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            # Count images
            images = list(class_dir.glob('*.jpg')) + \
                     list(class_dir.glob('*.jpeg')) + \
                     list(class_dir.glob('*.png')) + \
                     list(class_dir.glob('*.bmp'))

            if images:
                classes.append(class_dir.name)
                class_counts[class_dir.name] = len(images)
                total_images += len(images)

    if not classes:
        return {'valid': False, 'error': 'No valid class folders with images found'}

    return {
        'valid': True,
        'num_classes': len(classes),
        'classes': classes,
        'total_images': total_images,
        'class_counts': class_counts
    }


def split_dataset(data_dir: Path, split_dir: Path, config: dict) -> bool:
    """Split dataset into train/val/test sets."""
    import random
    from collections import defaultdict

    print_info(f"Splitting dataset from {data_dir} to {split_dir}")

    # Validate source
    validation = validate_dataset(data_dir)
    if not validation['valid']:
        print_error(f"Invalid dataset: {validation['error']}")
        return False

    print_info(f"Found {validation['num_classes']} classes, {validation['total_images']} images")

    # Create split directories
    train_dir = split_dir / 'train'
    val_dir = split_dir / 'val'
    test_dir = split_dir / 'test'

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Split each class
    train_ratio = config.get('train_ratio', 0.8)
    val_ratio = config.get('val_ratio', 0.1)

    split_counts = defaultdict(int)

    for class_name in validation['classes']:
        source_class_dir = data_dir / class_name

        # Get all images
        images = list(source_class_dir.glob('*.jpg')) + \
                 list(source_class_dir.glob('*.jpeg')) + \
                 list(source_class_dir.glob('*.png')) + \
                 list(source_class_dir.glob('*.bmp'))

        # Shuffle
        random.seed(42)  # Reproducibility
        random.shuffle(images)

        # Calculate split points
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        # Copy files
        for split_name, split_images in splits.items():
            dest_class_dir = split_dir / split_name / class_name
            dest_class_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                dest_path = dest_class_dir / img_path.name
                if not dest_path.exists():
                    shutil.copy2(img_path, dest_path)
                split_counts[split_name] += 1

    print_success("Dataset split complete:")
    print_info(f"  Train: {split_counts['train']} images")
    print_info(f"  Val: {split_counts['val']} images")
    print_info(f"  Test: {split_counts['test']} images")

    return True


def setup_dataset(config: dict) -> bool:
    """Interactive dataset setup."""
    print_header("DATASET SETUP")

    # Get data directory
    print_info("The dataset should be organized as:")
    print_info("  data_dir/")
    print_info("    class1/")
    print_info("      image1.jpg")
    print_info("      image2.jpg")
    print_info("    class2/")
    print_info("      ...")
    print()

    data_dir = prompt_path(
        "Enter path to raw image dataset",
        must_exist=True,
        default=str(config.get('data_dir', './Data'))
    )

    # Validate dataset
    print_info("Validating dataset...")
    validation = validate_dataset(data_dir)

    if not validation['valid']:
        print_error(f"Dataset validation failed: {validation['error']}")
        print_info("Please ensure your dataset is organized with one folder per class")
        return False

    print_success(f"Found {validation['num_classes']} classes:")
    for cls, count in sorted(validation['class_counts'].items()):
        print_info(f"  {cls}: {count} images")

    # Check if split exists
    split_dir = Path(config.get('split_dir', './split_dataset'))

    if split_dir.exists() and (split_dir / 'train').exists():
        print_warning(f"Split dataset already exists at {split_dir}")
        if not prompt_yes_no("Re-split dataset? (This will overwrite existing split)"):
            config['data_dir'] = data_dir
            config['split_dir'] = split_dir
            return True

    # Split dataset
    split_dir = prompt_path(
        "Enter path for split dataset output",
        must_exist=False,
        default=str(split_dir)
    )

    if not split_dataset(data_dir, split_dir, config):
        return False

    config['data_dir'] = data_dir
    config['split_dir'] = split_dir

    return True


# =============================================================================
# PHASE 1: BACKBONE TRAINING
# =============================================================================

def prompt_backbone_selection(backbones: list[str]) -> str:
    """Prompt user to select a backbone from the list."""
    print_info("Available backbones:")
    for i, backbone in enumerate(backbones, 1):
        print(f"  {i:2d}. {backbone}")
    print()

    while True:
        response = input(f"Select backbone [1-{len(backbones)}]: ").strip()
        try:
            idx = int(response)
            if 1 <= idx <= len(backbones):
                return backbones[idx - 1]
            print(f"Please enter a number between 1 and {len(backbones)}")
        except ValueError:
            # Check if they typed the backbone name
            if response in backbones:
                return response
            print("Please enter a valid number or backbone name")


def prompt_epochs(prompt_text: str, default: int, min_val: int = 1, max_val: int = 100) -> int:
    """Prompt user for number of epochs."""
    while True:
        response = input(f"{prompt_text} [{default}]: ").strip()
        if response == '':
            return default
        try:
            val = int(response)
            if min_val <= val <= max_val:
                return val
            print(f"Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid integer")


def train_backbones(config: dict, quick_mode: bool = False) -> bool:
    """Train all 15 backbone models."""
    print_header("PHASE 1: BACKBONE TRAINING")

    backbones = config['backbones']

    if quick_mode:
        print_warning("QUICK MODE: Training a single backbone with custom epochs")
        print()

        # Let user choose which backbone
        selected_backbone = prompt_backbone_selection(backbones)
        print_success(f"Selected: {selected_backbone}")
        print()

        # Let user choose epochs
        print_info("Configure training epochs:")
        epochs_head = prompt_epochs(
            "  Head training epochs (classifier only)",
            default=config['quick_epochs_head'],
            min_val=1, max_val=50
        )
        epochs_ft = prompt_epochs(
            "  Fine-tuning epochs (full model)",
            default=config['quick_epochs_finetune'],
            min_val=1, max_val=50
        )

        # Optionally configure batch size
        print()
        if prompt_yes_no("Configure batch size?", default=False):
            batch_size = prompt_epochs(
                "  Batch size",
                default=config['quick_batch_size'],
                min_val=4, max_val=64
            )
        else:
            batch_size = config['quick_batch_size']

        backbones = [selected_backbone]

        print()
        print_info("Quick mode configuration:")
        print_info(f"  Backbone: {selected_backbone}")
        print_info(f"  Head epochs: {epochs_head}")
        print_info(f"  Fine-tune epochs: {epochs_ft}")
        print_info(f"  Batch size: {batch_size}")
    else:
        epochs_head = config['epochs_head']
        epochs_ft = config['epochs_finetune']
        batch_size = config['batch_size']

    print_info(f"Backbones to train: {len(backbones)}")
    print_info(f"Epochs (head/finetune): {epochs_head}/{epochs_ft}")
    print_info(f"Batch size: {batch_size}")
    print_info(f"Estimated time: {estimate_time('backbones_only', len(backbones))}")
    print()

    if not prompt_yes_no("Start backbone training?"):
        return False

    # Set environment variables
    os.environ['DBT_BASE_DIR'] = str(config['base_dir'])
    os.environ['DBT_RAW_DIR'] = str(config['data_dir'])
    os.environ['DBT_SPLIT_DIR'] = str(config['split_dir'])

    if quick_mode:
        os.environ['DBT_DEBUG_MODE'] = 'true'
        os.environ['DBT_DEBUG_HEAD_EPOCHS'] = str(epochs_head)
        os.environ['DBT_DEBUG_FT_EPOCHS'] = str(epochs_ft)
        os.environ['DBT_DEBUG_BATCH_SIZE'] = str(batch_size)

    # Run backbone training using the modular pipeline (run_pipeline.py -> BASE-BACK/src/main.py)
    run_pipeline_path = config['base_dir'] / 'run_pipeline.py'

    if not run_pipeline_path.exists():
        print_error(f"run_pipeline.py not found at {run_pipeline_path}")
        print_info("This script orchestrates training via BASE-BACK/src/main.py")
        return False

    # Verify BASE-BACK module exists
    base_back_main = config['base_dir'] / 'BASE-BACK' / 'src' / 'main.py'
    if not base_back_main.exists():
        print_error("BASE-BACK/src/main.py not found")
        return False

    start_time = time.time()

    try:
        if quick_mode:
            # Train single backbone in debug mode
            os.environ['DBT_DEBUG_BACKBONE'] = backbones[0]
            os.environ['DBT_DEBUG_FUNCTION'] = 'full_training'

        subprocess.run(
            [sys.executable, str(run_pipeline_path)],
            cwd=str(config['base_dir']),
            check=True
        )

        elapsed = (time.time() - start_time) / 3600
        print_success(f"Backbone training completed in {elapsed:.2f} hours")
        return True

    except subprocess.CalledProcessError as e:
        print_error(f"Backbone training failed: {e}")
        return False
    except KeyboardInterrupt:
        print_warning("Training interrupted by user")
        return False


# =============================================================================
# PHASE 2: ENSEMBLE PIPELINE
# =============================================================================

def run_ensemble_pipeline(config: dict, quick_mode: bool = False) -> bool:
    """Run the 7-stage 15-COIN ensemble pipeline."""
    print_header("PHASE 2: 15-COIN ENSEMBLE PIPELINE")

    print_info("This pipeline includes:")
    print_info("  Stage 1: Extract individual backbone predictions")
    print_info("  Stage 2: Score-level ensembles (voting, averaging)")
    print_info("  Stage 3: Stacking (Logistic Regression, XGBoost, MLP)")
    print_info("  Stage 4: Feature fusion (Concat, Attention, Bilinear)")
    print_info("  Stage 5: Mixture of Experts")
    print_info("  Stage 6: Meta-ensemble (combines all above)")
    print_info("  Stage 7: Knowledge distillation")
    print()
    print_info(f"Estimated time: {estimate_time('ensemble_only')}")
    print()

    if not prompt_yes_no("Start ensemble pipeline?"):
        return False

    # Check for trained backbones
    checkpoints_dir = config['base_dir'] / 'checkpoints'
    if not checkpoints_dir.exists():
        print_error("No checkpoints directory found. Train backbones first.")
        return False

    # Set environment
    os.environ['DBT_BASE_DIR'] = str(config['base_dir'])
    os.environ['DBT_SPLIT_DIR'] = str(config['split_dir'])

    # Run ensemble pipeline
    ensemble_script = config['base_dir'] / 'ensemble_system' / 'test_pipeline.py'

    if not ensemble_script.exists():
        # Fall back to run_15coin_pipeline.py
        ensemble_script = config['base_dir'] / 'ensemble_system' / 'run_15coin_pipeline.py'

    if not ensemble_script.exists():
        print_error("Ensemble pipeline script not found")
        return False

    start_time = time.time()

    try:
        subprocess.run(
            [sys.executable, str(ensemble_script)],
            cwd=str(config['base_dir'] / 'ensemble_system'),
            check=True
        )

        elapsed = (time.time() - start_time) / 3600
        print_success(f"Ensemble pipeline completed in {elapsed:.2f} hours")
        return True

    except subprocess.CalledProcessError as e:
        print_error(f"Ensemble pipeline failed: {e}")
        return False
    except KeyboardInterrupt:
        print_warning("Pipeline interrupted by user")
        return False


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def interactive_mode(config: dict) -> bool:
    """Run in interactive mode with user prompts."""
    print_header("INTERACTIVE MODE")

    # Get base directory
    print_info("Setting up project directory...")
    base_dir = prompt_path(
        "Enter base project directory",
        must_exist=True,
        default=str(Path(__file__).parent.absolute())
    )
    config['base_dir'] = base_dir

    # Environment setup
    if not setup_environment(config):
        return False

    # Dataset setup
    if not setup_dataset(config):
        return False

    # Choose what to run
    print()
    print_info("What would you like to run?")
    print("  1. Full pipeline (backbone training + ensemble) [~55 hours]")
    print("  2. Quick test (single backbone, reduced epochs) [~30 min]")
    print("  3. Backbone training only [~50 hours]")
    print("  4. Ensemble pipeline only (requires trained backbones) [~5 hours]")
    print()

    while True:
        choice = input("Enter choice [1-4]: ").strip()
        if choice in ['1', '2', '3', '4']:
            break
        print("Please enter 1, 2, 3, or 4")

    quick_mode = (choice == '2')

    if choice in ['1', '2', '3']:
        if not train_backbones(config, quick_mode=quick_mode):
            if choice == '1':
                print_warning("Backbone training failed, skipping ensemble")
                return False

    if choice in ['1', '2', '4']:
        if not run_ensemble_pipeline(config, quick_mode=quick_mode):
            return False

    print_header("PIPELINE COMPLETE")
    print_success("All stages completed successfully!")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reproduce the Sugarcane Disease Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
------
  full           Run complete pipeline (backbones + ensemble)
                 WARNING: Takes ~55 hours on RTX 4500 Ada GPU
                 
  quick_test     Quick validation with 1 backbone, reduced epochs
                 Takes ~30 minutes, useful for testing setup
                 
  backbones_only Train all 15 backbone models only
                 Takes ~50 hours, required before ensemble
                 
  ensemble_only  Run 7-stage ensemble pipeline only
                 Requires pre-trained backbones in checkpoints/
                 Takes ~5 hours
                 
  interactive    Step-by-step guided setup with prompts

EXAMPLES:
---------
  # Full pipeline with default settings
  python reproduce_pipeline.py --mode full
  
  # Quick test to verify setup
  python reproduce_pipeline.py --mode quick_test
  
  # Custom data directory
  python reproduce_pipeline.py --mode full --data-dir /path/to/images
  
  # Resume ensemble (backbones already trained)
  python reproduce_pipeline.py --mode ensemble_only

REQUIREMENTS:
-------------
  - Python 3.8+
  - PyTorch 2.0+ with CUDA
  - NVIDIA GPU with 8GB+ VRAM
  - 50GB+ disk space
  
For detailed documentation, see PROJECT_SUMMARY.md
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'quick_test', 'backbones_only', 'ensemble_only', 'interactive'],
        default='interactive',
        help='Execution mode (default: interactive)'
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default=str(Path(__file__).parent.absolute()),
        help='Base project directory'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to raw image dataset (folders organized by class)'
    )

    parser.add_argument(
        '--split-dir',
        type=str,
        default=None,
        help='Path for train/val/test split output'
    )

    parser.add_argument(
        '--skip-env-check',
        action='store_true',
        help='Skip environment validation (not recommended)'
    )

    parser.add_argument(
        '--yes',
        '-y',
        action='store_true',
        help='Answer yes to all prompts (non-interactive)'
    )

    args = parser.parse_args()

    # Initialize config
    config = DEFAULT_CONFIG.copy()
    config['base_dir'] = Path(args.base_dir)

    if args.data_dir:
        config['data_dir'] = Path(args.data_dir)
    else:
        config['data_dir'] = config['base_dir'] / 'Data'

    if args.split_dir:
        config['split_dir'] = Path(args.split_dir)
    else:
        config['split_dir'] = config['base_dir'] / 'split_dataset'

    # Print banner
    print_header("SUGARCANE DISEASE CLASSIFICATION")
    print_info(f"Mode: {args.mode}")
    print_info(f"Base directory: {config['base_dir']}")
    print_info(f"Data directory: {config['data_dir']}")
    print()

    # Handle different modes
    if args.mode == 'interactive':
        success = interactive_mode(config)
    else:
        # Non-interactive modes
        if not args.skip_env_check:
            if not setup_environment(config):
                sys.exit(1)

        # Check dataset
        if not config['split_dir'].exists() or not (config['split_dir'] / 'train').exists():
            print_info("Split dataset not found, creating...")
            if config['data_dir'].exists():
                if not split_dataset(config['data_dir'], config['split_dir'], config):
                    print_error("Dataset setup failed")
                    sys.exit(1)
            else:
                print_error(f"Data directory not found: {config['data_dir']}")
                print_info("Please specify --data-dir or run in interactive mode")
                sys.exit(1)

        quick_mode = (args.mode == 'quick_test')

        if args.mode in ['full', 'quick_test', 'backbones_only']:
            if not train_backbones(config, quick_mode=quick_mode):
                if args.mode == 'backbones_only':
                    sys.exit(1)
                print_warning("Backbone training failed")

        if args.mode in ['full', 'quick_test', 'ensemble_only']:
            if not run_ensemble_pipeline(config, quick_mode=quick_mode):
                sys.exit(1)

        success = True

    if success:
        print_header("SUCCESS")
        print_success("Pipeline completed successfully!")
        print_info("Results saved to:")
        print_info(f"  Checkpoints: {config['base_dir'] / 'checkpoints'}")
        print_info(f"  Ensembles: {config['base_dir'] / 'ensembles'}")
        print_info(f"  Metrics: {config['base_dir'] / 'metrics_output'}")
        sys.exit(0)
    else:
        print_header("FAILED")
        print_error("Pipeline did not complete successfully")
        sys.exit(1)


if __name__ == '__main__':
    main()
