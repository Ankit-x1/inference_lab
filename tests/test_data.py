"""Tests for the CIFAR-10 data pipeline.

Run on the machine where torch/torchvision are installed and the dataset can
be downloaded. Test data usage is kept separate: test set is never used for
splits or calibration.
"""

from __future__ import annotations

import torch

from edge_vision.data import DataConfig, TransformedSubset, eval_transform, load_cifar10


def test_class_count_and_shape():
    cfg = DataConfig(train_size=49950, val_size=50, batch_size=10, num_workers=0)
    train_loader, val_loader, test_loader = load_cifar10(cfg)

    images, labels = next(iter(test_loader))
    assert images.shape[1:] == (3, 32, 32), f"unexpected image shape {images.shape}"
    assert labels.min() >= 0 and labels.max() <= 9, f"labels out of range: {labels}"
    assert len(test_loader.dataset) == 10_000


def test_split_sizes():
    cfg = DataConfig(train_size=45_000, val_size=5_000, batch_size=64, num_workers=0)
    train_loader, val_loader, test_loader = load_cifar10(cfg)
    assert len(train_loader.dataset) == 45_000
    assert len(val_loader.dataset) == 5_000
    assert len(test_loader.dataset) == 10_000


def test_split_reproducible():
    cfg1 = DataConfig(seed=42, train_size=49000, val_size=1000, batch_size=32, num_workers=0)
    cfg2 = DataConfig(seed=42, train_size=49000, val_size=1000, batch_size=32, num_workers=0)
    _, v1, _ = load_cifar10(cfg1)
    _, v2, _ = load_cifar10(cfg2)
    idx1 = v1.dataset.indices
    idx2 = v2.dataset.indices
    assert idx1 == idx2, "validation split must be reproducible with fixed seed"


def test_validation_deterministic():
    cfg = DataConfig(train_size=49950, val_size=50, batch_size=10, num_workers=0)
    _, val_loader, _ = load_cifar10(cfg)
    dataset = val_loader.dataset
    first = dataset[0][0]
    second = dataset[0][0]
    assert torch.equal(first, second), "validation preprocessing must be deterministic"


def test_transformed_subset_len():
    base = list(range(20))
    subset = TransformedSubset(None, base[:7], transform=None)
    assert len(subset) == 7
