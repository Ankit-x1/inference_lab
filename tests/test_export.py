"""Tests for ONNX export correctness (shape, finite logits, class agreement)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import onnxruntime as ort
from edge_vision.export_onnx import compare_pytorch_onnx, load_onnx_session, run_onnx
from edge_vision.model import create_model


def test_onnx_output_shape_and_finite(tmp_path):
    model = create_model()
    model.eval()
    onnx_path = tmp_path / "model.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32),
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,  # legacy TorchScript exporter; avoids onnx_ir incompatibility
    )
    session = load_onnx_session(str(onnx_path))
    outputs = run_onnx(session, np.random.randn(2, 3, 32, 32).astype(np.float32))
    assert outputs.shape == (2, 10), f"unexpected ONNX output shape {outputs.shape}"
    assert np.isfinite(outputs).all(), "ONNX logits must be finite"


def test_pytorch_onnx_class_agreement():
    model = create_model()
    model.eval()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.onnx")
        torch.onnx.export(
            model,
            torch.randn(1, 3, 32, 32),
            path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=13,
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            dynamo=False,
        )
        session = load_onnx_session(path)
    images = np.random.randn(8, 3, 32, 32).astype(np.float32)
    loader = _fake_loader(images, labels=[0] * 8)
    result = compare_pytorch_onnx(
        model, session, loader, torch.device("cpu"), n_samples=8, tolerance=1e-4
    )
    assert result["within_tolerance"] is True
    assert result["class_agreement"] == 1.0


def _fake_loader(images, labels):
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.from_numpy(images), torch.tensor(labels))
    return DataLoader(ds, batch_size=4, shuffle=False)
