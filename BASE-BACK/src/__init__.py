"""
Disease Classification Framework
Phase-1: Modular architecture with professional structure
"""

__version__ = '1.0.0'
__author__ = 'Disease Classification Team'

from .config import settings
from .models import BACKBONE_MAP, create_custom_backbone_safe
from .training import train_epoch_optimized, validate_epoch_optimized
from .utils import DEVICE, logger


__all__ = [
    'settings',
    'logger',
    'DEVICE',
    'create_custom_backbone_safe',
    'BACKBONE_MAP',
    'train_epoch_optimized',
    'validate_epoch_optimized',
]
