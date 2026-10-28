"""Export and validate ONNX model."""

import json
from pathlib import Path
import torch
from edge_vision.data import DataConfig, load_cifar10
from edge_vision.model import create_model
from edge_vision.export_onnx import export_model, load_onnx_session, compare_pytorch_onnx

with open("configs/experiment.json") as f:
    cfg = json.load(f)

ckpt = "artifacts/checkpoints/best.pt"
onnx_path = "artifacts/models/model_fp32.onnx"
Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

manifest = export_model(
    ckpt,
    onnx_path,
    input_shape=tuple(cfg["input_shape"]),
    opset=cfg["export"]["opset"],
)
print(f"Exported: {onnx_path}")

data_cfg = DataConfig(
    seed=cfg["seed"],
    batch_size=cfg["benchmark"]["test_samples"],
    num_workers=0,
)
_, _, test_loader = load_cifar10(data_cfg)
device = torch.device("cpu")
model = create_model()
ckpt_data = torch.load(ckpt, map_location="cpu")
model.load_state_dict(ckpt_data["model_state_dict"])
model.to(device).eval()

session = load_onnx_session(onnx_path)
result = compare_pytorch_onnx(
    model, session, test_loader, device, n_samples=64, tolerance=1e-4
)
print(f"Max abs diff: {result['max_abs_diff']:.2e}")
print(f"Within tolerance: {result['within_tolerance']}")
print(f"Class agreement: {result['class_agreement']:.4f}")
