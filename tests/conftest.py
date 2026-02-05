# ============================================================================
# DISEASE Classification Pipeline - Pytest Configuration
# ============================================================================
# Created: Sprint 2 - CI/CD Without Behavior Change
# Purpose: Shared fixtures and configuration for all tests
# ============================================================================

import os
import sys
from pathlib import Path

import pytest
import torch


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# CRITICAL: Set DBT_BASE_DIR before any imports of Base_backbones.py
# This prevents FileNotFoundError on CI runners where F:\ doesn't exist
if not os.environ.get('DBT_BASE_DIR'):
    # Use project root for tests, creating temp directories as needed
    os.environ['DBT_BASE_DIR'] = str(PROJECT_ROOT)
    os.environ['CI'] = 'true'  # Signal we're in CI-like environment


# =============================================================================
# Environment Configuration
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_environment():
    """Configure environment for testing."""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["DBT_DEBUG_MODE"] = "false"

    # Suppress TensorFlow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    yield


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def device():
    """Get the available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def device_name(device):
    """Get device name as string."""
    return str(device)


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def num_classes():
    """Number of disease classes."""
    return 13


@pytest.fixture(scope="session")
def img_size():
    """Standard image size for models."""
    return 224


@pytest.fixture(scope="session")
def batch_size():
    """Batch size for testing."""
    return 2


@pytest.fixture(scope="session")
def dummy_input(batch_size, img_size, device):
    """Create dummy input tensor for testing."""
    return torch.randn(batch_size, 3, img_size, img_size, device=device)


@pytest.fixture(scope="session")
def backbone_names():
    """List of all backbone names."""
    from Base_backbones import BACKBONES
    return BACKBONES


@pytest.fixture(scope="session")
def sample_backbone_names():
    """Subset of backbones for quick testing."""
    return ["CustomConvNeXt", "CustomResNetMish", "CustomCoAtNet"]


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def checkpoints_dir(project_root):
    """Checkpoints directory."""
    return project_root / "checkpoints"


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Data directory."""
    return project_root / "Data"


@pytest.fixture(scope="session")
def split_dataset_dir(project_root):
    """Split dataset directory."""
    return project_root / "split_dataset"


# =============================================================================
# Skip Conditions
# =============================================================================

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

requires_checkpoint = pytest.mark.skipif(
    not (Path(__file__).parent.parent / "checkpoints").exists(),
    reason="Checkpoints directory not found"
)

slow_test = pytest.mark.slow


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "integration: mark as integration test")
    config.addinivalue_line("markers", "smoke: mark as smoke test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip slow tests unless explicitly requested."""
    if not config.getoption("--runslow", default=False):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
