"""Core utilities: logging, device management, reproducibility"""

import os
import sys
import logging
import random
import numpy as np
import torch
import platform
import hashlib
import time
import json
from pathlib import Path
from collections import deque

# Handle both package imports and direct sys.path imports
try:
    from ..config.settings import DEBUG_LOG_DIR
except ImportError:
    from config.settings import DEBUG_LOG_DIR

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =============================================================================
# DEVICE MANAGEMENT
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device():
    """Get current device"""
    return DEVICE

def get_optimal_workers():
    """Get optimal number of workers for DataLoader"""
    if platform.system() == 'Windows':
        cpu_count = os.cpu_count() or 4
        optimal_workers = max(2, min(16, int(cpu_count * 0.75)))
        return optimal_workers
    else:
        cpu_count = os.cpu_count() or 4
        return max(2, min(12, int(cpu_count * 0.8)))

OPTIMAL_WORKERS = get_optimal_workers()

# =============================================================================
# LOGGING
# =============================================================================

class DedupLogger:
    """Enhanced logger with message de-duplication and structured output"""
    def __init__(self, name="disease_pipeline"):
        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        
        self.message_cache = deque(maxlen=50)
        self.last_message_time = {}
        self.system_info_logged = False
        
        if not self.logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            from logging.handlers import RotatingFileHandler
            
            fh = RotatingFileHandler(
                Path.cwd() / 'training.log',
                maxBytes=10*1024*1024,
                backupCount=5
            )
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
            error_fh = RotatingFileHandler(
                Path.cwd() / 'training_errors.log',
                maxBytes=5*1024*1024,
                backupCount=3
            )
            error_fh.setLevel(logging.ERROR)
            error_fh.setFormatter(formatter)
            self.logger.addHandler(error_fh)
        
        self.logger.setLevel(logging.INFO)
    
    def _hash_message(self, msg):
        """Create hash of message for de-duplication"""
        return hashlib.md5(msg.encode()).hexdigest()[:8]
    
    def _should_log(self, msg, level, dedupe_window=5.0):
        """Check if message should be logged based on de-duplication rules"""
        msg_hash = self._hash_message(msg)
        current_time = time.time()
        
        if msg_hash in self.last_message_time:
            last_time = self.last_message_time[msg_hash]
            if current_time - last_time < dedupe_window:
                return False
        
        self.last_message_time[msg_hash] = current_time
        return True
    
    def info(self, msg, dedupe=False):
        if not dedupe or self._should_log(msg, 'INFO'):
            self.logger.info(msg)
    
    def debug(self, msg, dedupe=False):
        if not dedupe or self._should_log(msg, 'DEBUG'):
            self.logger.debug(msg)
    
    def warning(self, msg, dedupe=False):
        if not dedupe or self._should_log(msg, 'WARNING'):
            self.logger.warning(msg)
    
    def error(self, msg, dedupe=False):
        if not dedupe or self._should_log(msg, 'ERROR'):
            self.logger.error(msg)
    
    def exception(self, msg):
        self.logger.exception(msg)
    
    def log_system_info_once(self):
        """Log system information only once"""
        if not self.system_info_logged:
            self.info(f"Device: {DEVICE} | Platform: {platform.system()} | Optimal Workers: {OPTIMAL_WORKERS}")
            self.system_info_logged = True

logger = DedupLogger()

# =============================================================================
# SMOKE CHECK LOGGER
# =============================================================================

class SmokeCheckLogger:
    """Dedicated logger for smoke checks and validation"""
    def __init__(self):
        self.checks = []
        self.log_file = DEBUG_LOG_DIR / f'smoke_check_{time.strftime("%Y%m%d_%H%M%S")}.log'
        
    def log_check(self, check_name, status, details=""):
        """Log a smoke check result"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        check_result = {
            'timestamp': timestamp,
            'check_name': check_name,
            'status': status,
            'details': details
        }
        self.checks.append(check_result)
        
        status_symbol = "[PASS]" if status == "PASS" else "[FAIL]"
        log_msg = f"{status_symbol} {check_name}: {status}"
        if details:
            log_msg += f" - {details}"
        
        logger.info(log_msg)
        
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} | {status_symbol} {check_name}: {status}")
            if details:
                f.write(f" - {details}")
            f.write("\n")
    
    def save_summary(self):
        """Save smoke check summary"""
        import json
        summary_file = DEBUG_LOG_DIR / 'smoke_check_summary.json'
        summary = {
            'total_checks': len(self.checks),
            'passed': sum(1 for c in self.checks if c['status'] == 'PASS'),
            'failed': sum(1 for c in self.checks if c['status'] == 'FAIL'),
            'checks': self.checks
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Smoke check summary: {summary['passed']}/{summary['total_checks']} passed")
        return summary

smoke_checker = SmokeCheckLogger()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_device_info():
    """Get detailed device information"""
    return {
        'device': str(DEVICE),
        'platform': platform.system(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'optimal_workers': OPTIMAL_WORKERS,
    }

__all__ = [
    'set_seed',
    'DEVICE',
    'get_device',
    'get_optimal_workers',
    'OPTIMAL_WORKERS',
    'DedupLogger',
    'logger',
    'SmokeCheckLogger',
    'smoke_checker',
    'compute_file_sha256',
    'get_device_info',
]
