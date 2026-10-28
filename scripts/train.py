"""Train the baseline FP32 classifier."""

import json
from pathlib import Path
import torch
from edge_vision.data import DataConfig, load_cifar10
from edge_vision.train import train

with open("configs/experiment.json") as f:
    cfg = json.load(f)

data_cfg = DataConfig(
    seed=cfg["seed"],
    train_size=cfg["splits"]["train"],
    val_size=cfg["splits"]["val"],
    batch_size=cfg["train"]["batch_size"],
    num_workers=0,
)
train_loader, val_loader, _ = load_cifar10(data_cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, best_val_acc, history = train(
    cfg["train"], train_loader, val_loader, device,
    checkpoint_dir="artifacts/checkpoints",
)
print(f"Best validation accuracy: {best_val_acc:.4f}")
