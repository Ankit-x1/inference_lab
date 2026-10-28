"""CIFAR-10 data pipeline with reproducible train/validation/test splits.

Rules enforced here:
- Training split (45,000 images) gets random crop + horizontal flip only.
- Validation (5,000 images from the official training set) and test
  (10,000 official test images) use deterministic preprocessing only.
- Test data never influences checkpoint selection or quantization calibration.
- The split seed and preprocessing settings live in the experiment config.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


class DataConfig:
    """Experiment configuration for the data pipeline."""

    def __init__(
        self,
        seed: int = 42,
        train_size: int = 45_000,
        val_size: int = 5_000,
        test_size: int = 10_000,
        batch_size: int = 128,
        num_workers: int = 2,
        data_dir: str = "data",
    ) -> None:
        self.seed = seed
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir


class TransformedSubset(Dataset):
    """A subset of a dataset with its own transform applied on read."""

    def __init__(self, dataset: Dataset, indices: list[int], transform=None) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __getitem__(self, idx: int):
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self) -> int:
        return len(self.indices)


def train_transform() -> transforms.Compose:
    """Augmentation applied to training images only."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    """Deterministic preprocessing for validation, calibration, and testing."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def load_cifar10(cfg: DataConfig):
    """Load CIFAR-10 and build train/validation/test loaders.

    Returns (train_loader, val_loader, test_loader). The official test set is
    kept entirely separate until final evaluation.
    """
    from pathlib import Path

    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Base training dataset without a transform; we apply transforms per split.
    base_train = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=None)
    test_set = datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=eval_transform()
    )

    generator = torch.Generator().manual_seed(cfg.seed)
    train_split, val_split = random_split(
        range(len(base_train)), [cfg.train_size, cfg.val_size], generator=generator
    )
    train_indices = list(train_split.indices)
    val_indices = list(val_split.indices)

    train_ds = TransformedSubset(base_train, train_indices, train_transform())
    val_ds = TransformedSubset(base_train, val_indices, eval_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_loader, val_loader, test_loader


def collect_numpy(test_loader, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Collect up to n_samples normalized inputs and labels as numpy arrays."""
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    collected = 0
    for images_batch, labels_batch in test_loader:
        images.append(images_batch.numpy())
        labels.append(labels_batch.numpy())
        collected += images_batch.shape[0]
        if collected >= n_samples:
            break
    images_arr = np.concatenate(images, axis=0)[:n_samples].astype(np.float32)
    labels_arr = np.concatenate(labels, axis=0)[:n_samples]
    return images_arr, labels_arr


def calibration_samples(train_loader, n_samples: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Fixed calibration sample from the training split only (never test data)."""
    return collect_numpy(train_loader, n_samples)
