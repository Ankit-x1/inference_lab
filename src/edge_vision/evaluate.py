"""Evaluation: top-1 and per-class accuracy on a held-out set."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Dict:
    """Compute top-1 and per-class accuracy on a loader."""
    model.eval()
    correct = 0
    total = 0
    num_classes = 10
    class_correct = np.zeros(num_classes, dtype=int)
    class_total = np.zeros(num_classes, dtype=int)
    predictions: list = []
    labels_list: list = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        for label, pred in zip(labels.tolist(), predicted.tolist()):
            class_total[label] += 1
            class_correct[label] += int(label == pred)
        predictions.extend(predicted.tolist())
        labels_list.extend(labels.tolist())

    per_class = {
        i: (class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0)
        for i in range(num_classes)
    }
    return {
        "top1_accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "per_class_accuracy": per_class,
        "predictions": predictions,
        "labels": labels_list,
    }
