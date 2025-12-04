"""
Disease Classification Framework
Phase-1: Modular architecture with professional structure
"""

__version__ = '1.0.0'
__author__ = 'Disease Classification Team'

from .config import settings
from .utils import logger, DEVICE
from .models import create_custom_backbone_safe, BACKBONE_MAP
from .training import train_epoch_optimized, validate_epoch_optimized

__all__ = [
    'settings',
    'logger',
    'DEVICE',
    'create_custom_backbone_safe',
    'BACKBONE_MAP',
    'train_epoch_optimized',
    'validate_epoch_optimized',
]
