"""PyTorch-to-ONNX export and correctness verification.

Exports the best checkpoint to ONNX with a fixed single-image input shape,
then verifies that PyTorch and ONNX Runtime return consistent logits within a
documented numerical tolerance.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from .data import DataConfig, collect_numpy, load_cifar10
from .model import create_model


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def export_model(
    checkpoint_path: str,
    onnx_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    opset: int = 13,
) -> Dict:
    """Export the trained checkpoint to ONNX and write a manifest."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    manifest = {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_accuracy": checkpoint.get("val_accuracy"),
        "input_shape": list(input_shape),
        "opset": opset,
        "onnx_file": onnx_path,
        "onnx_sha256": file_sha256(onnx_path),
        "python_version": platform.python_version(),
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
    }
    manifest_path = str(Path(onnx_path).with_suffix(".manifest.json"))
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_onnx_session(onnx_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def run_onnx(session: ort.InferenceSession, images_np: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: images_np})[0]


def compare_pytorch_onnx(
    model: torch.nn.Module,
    session: ort.InferenceSession,
    test_loader: DataLoader,
    device: torch.device,
    n_samples: int = 64,
    tolerance: float = 1e-4,
) -> Dict:
    """Compare PyTorch and ONNX Runtime logits on fixed samples.

    Returns a dict with max absolute difference, pass/fail, and class agreement.
    """
    model.eval()
    images_np, labels_np = collect_numpy(test_loader, n_samples)

    # PyTorch logits
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(images_np).to(device)).cpu().numpy()
    onnx_logits = run_onnx(session, images_np)

    abs_diff = np.abs(torch_logits - onnx_logits)
    max_diff = float(np.max(abs_diff))

    torch_preds = torch_logits.argmax(axis=1)
    onnx_preds = onnx_logits.argmax(axis=1)
    agreement = float(np.mean(torch_preds == onnx_preds))

    return {
        "n_samples": int(n_samples),
        "tolerance": tolerance,
        "max_abs_diff": max_diff,
        "within_tolerance": bool(max_diff <= tolerance),
        "class_agreement": agreement,
        "torch_predictions": torch_preds.tolist(),
        "onnx_predictions": onnx_preds.tolist(),
        "labels": labels_np.tolist(),
    }


def verify_on_pi(onnx_path: str, images_np: np.ndarray) -> np.ndarray:
    """Placeholder entry point for the Pi correctness test (run on the Pi).

    Loads the exported ONNX model with ONNX Runtime CPU and runs it on the
    provided normalized inputs, returning logits.
    """
    session = load_onnx_session(onnx_path)
    return run_onnx(session, images_np)
