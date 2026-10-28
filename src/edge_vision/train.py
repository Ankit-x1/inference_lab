"""Supervised training loop: cross-entropy + AdamW, best validation checkpoint saved."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .model import create_model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (average loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate loss and accuracy on a loader. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train(
    cfg: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    checkpoint_dir: str = "artifacts/checkpoints",
) -> Tuple[nn.Module, float, list]:
    """Train the model and save the best validation checkpoint.

    Returns (model, best_val_accuracy, history).
    """
    torch.manual_seed(cfg.get("seed", 42))
    model = create_model(num_classes=cfg.get("num_classes", 10)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 0.001),
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_path / "best.pt"
    history_path = checkpoint_path / "history.json"

    best_val_acc = 0.0
    history = []

    epochs = cfg.get("epochs", 20)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                },
                best_path,
            )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, best_val_acc, history
