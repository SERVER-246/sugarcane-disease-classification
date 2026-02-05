"""Dataset loading, preparation, and transforms"""

import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Handle both package imports and direct sys.path imports
try:
    from ..config.settings import IMG_SIZE, RAW_DIR, SEED, SPLIT_DIR, TEST_DIR, TRAIN_DIR, VAL_DIR
except ImportError:
    from config.settings import IMG_SIZE, RAW_DIR, SEED, SPLIT_DIR, TEST_DIR, TRAIN_DIR, VAL_DIR

from . import OPTIMAL_WORKERS, logger


# =============================================================================
# DATASET CLASSES
# =============================================================================

class OptimizedImageDataset(Dataset):
    """Optimized dataset class for Windows multiprocessing"""
    def __init__(self, samples, class_names, transform=None):
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.classes = class_names
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)
        return image, target

class WindowsCompatibleImageFolder(Dataset):
    """Enhanced ImageFolder optimized for Windows multiprocessing"""
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = self.root / cls
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)
        return image, target

class OptimizedTempDataset(Dataset):
    """Temporary dataset for K-fold cross-validation"""
    def __init__(self, samples, class_names, transform=None):
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.classes = class_names
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)
        return image, target

# =============================================================================
# TRANSFORMS
# =============================================================================

def create_optimized_transforms(size=IMG_SIZE, is_training=True):
    """Create optimized transforms with augmentation"""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.2)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(size * 1.14)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# =============================================================================
# DATA LOADERS
# =============================================================================

def worker_init_fn(worker_id):
    """Initialize workers properly for Windows multiprocessing"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

def create_optimized_dataloader(dataset, batch_size, shuffle=True, num_workers=None):
    """Create optimized DataLoader with Windows support"""
    if num_workers is None:
        num_workers = OPTIMAL_WORKERS

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2 if num_workers > 0 else None,
        'worker_init_fn': worker_init_fn if num_workers > 0 else None,
        'drop_last': False,
        'timeout': 30 if num_workers > 0 else 0
    }

    if num_workers == 0:
        loader_kwargs.pop('persistent_workers', None)
        loader_kwargs.pop('worker_init_fn', None)
        loader_kwargs.pop('timeout', None)
        loader_kwargs.pop('prefetch_factor', None)
        loader_kwargs['pin_memory'] = False

    try:
        return DataLoader(dataset, **loader_kwargs)
    except Exception as e:
        logger.warning(f"DataLoader creation failed: {e}")
        loader_kwargs.update({'num_workers': 0, 'pin_memory': False})
        loader_kwargs.pop('persistent_workers', None)
        loader_kwargs.pop('worker_init_fn', None)
        loader_kwargs.pop('timeout', None)
        return DataLoader(dataset, **loader_kwargs)

# =============================================================================
# DATASET PREPARATION
# =============================================================================

def prepare_optimized_datasets(raw_dir=RAW_DIR, split_dir=SPLIT_DIR,
                              train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=SEED):
    """Optimized dataset preparation"""
    raw_dir = Path(raw_dir)
    split_dir = Path(split_dir)

    if split_dir.exists() and any(split_dir.iterdir()):
        logger.info(f"Split dataset already exists at {split_dir}")
        return

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory {raw_dir} not found.")

    logger.info(f"Creating split dataset from {raw_dir} -> {split_dir}")

    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    all_samples = []

    for cls in classes:
        cls_dir = raw_dir / cls
        for img_path in cls_dir.glob("*"):
            if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                all_samples.append((str(img_path), cls))

    filepaths = np.array([s[0] for s in all_samples])
    labels = np.array([s[1] for s in all_samples])

    if len(np.unique(labels)) < 2:
        raise ValueError("Need at least two disease classes to split.")

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_idx, test_idx = next(sss1.split(filepaths, labels))

    X_train, X_test = filepaths[train_idx], filepaths[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    val_size = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss2.split(X_train, y_train))

    X_train_final, X_val = X_train[train_idx], X_train[val_idx]
    y_train_final, y_val = y_train[train_idx], y_train[val_idx]

    def copy_files(paths_labels, target_dir):
        for path, label in paths_labels:
            dest_dir = Path(target_dir) / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / Path(path).name
            if not dest_path.exists():
                shutil.copy2(path, str(dest_path))

    copy_files(zip(X_train_final, y_train_final), TRAIN_DIR)
    copy_files(zip(X_val, y_val), VAL_DIR)
    copy_files(zip(X_test, y_test), TEST_DIR)

    logger.info(f"Split created: train={len(X_train_final)}, val={len(X_val)}, test={len(X_test)}")

def verify_dataset_split(split_dir=SPLIT_DIR):
    """Verify dataset split integrity after creation"""
    logger.info("Verifying dataset split integrity...")
    issues = []

    train_path = split_dir / 'train'
    if not train_path.exists():
        logger.error(f"Train directory not found at {train_path}")
        return False

    expected_classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    logger.info(f"  Expected classes: {expected_classes}")

    split_stats = {}

    for split_name in ['train', 'val', 'test']:
        split_path = split_dir / split_name

        if not split_path.exists():
            issues.append(f"Missing {split_name} directory")
            continue

        found_classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])

        if set(found_classes) != set(expected_classes):
            issues.append(
                f"{split_name}: class mismatch. "
                f"Expected {len(expected_classes)} classes, found {len(found_classes)}"
            )

        total_images = 0
        for cls in found_classes:
            cls_path = split_path / cls
            images = list(cls_path.glob("*"))
            images = [img for img in images if img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
            total_images += len(images)

        split_stats[split_name] = {
            'classes': len(found_classes),
            'images': total_images
        }

        if total_images == 0:
            issues.append(f"{split_name}: no images found")
        else:
            logger.info(f"  {split_name}: {total_images} images across {len(found_classes)} classes")

    if issues:
        logger.error("Dataset split validation FAILED:")
        for issue in issues:
            logger.error(f"  [FAIL] {issue}")
        return False

    logger.info("Dataset split validation PASSED")
    return True

def prepare_datasets_for_backbone(backbone_name, size=IMG_SIZE):
    """Prepare datasets for specific backbone"""
    train_tf = create_optimized_transforms(size, is_training=True)
    val_tf = create_optimized_transforms(size, is_training=False)

    if TRAIN_DIR.exists() and VAL_DIR.exists():
        train_ds = WindowsCompatibleImageFolder(str(TRAIN_DIR), transform=train_tf)
        val_ds = WindowsCompatibleImageFolder(str(VAL_DIR), transform=val_tf)
    elif RAW_DIR.exists():
        raise FileNotFoundError("Please run dataset splitting first.")
    else:
        raise FileNotFoundError("No dataset found.")

    logger.info(f"Prepared datasets for {backbone_name}: train={len(train_ds)}, val={len(val_ds)}, classes={len(train_ds.classes)}")
    return train_ds, val_ds

__all__ = [
    'OptimizedImageDataset',
    'WindowsCompatibleImageFolder',
    'OptimizedTempDataset',
    'create_optimized_transforms',
    'worker_init_fn',
    'create_optimized_dataloader',
    'prepare_optimized_datasets',
    'verify_dataset_split',
    'prepare_datasets_for_backbone',
]
