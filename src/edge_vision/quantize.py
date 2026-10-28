"""Post-training static INT8 quantization via ONNX Runtime.

Loads the already-exported FP32 ONNX model, calibrates on a fixed sample of
training images (never test data), and writes a separate INT8 ONNX model.
The FP32 model is never overwritten.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from .data import DataConfig, collect_numpy, load_cifar10


class CalibrationReader:
    """Provides calibration batches in the exact input tensor format expected
    by the model. Implements get_next() / reset() consumed by quantize_static."""

    def __init__(self, images: np.ndarray, input_name: str) -> None:
        self.images = images
        self.input_name = input_name
        self.idx = 0

    def reset(self) -> None:
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.images):
            return None
        batch = self.images[self.idx : self.idx + 1].astype(np.float32)
        self.idx += 1
        return {self.input_name: batch}


def quantize_model(
    fp32_onnx: str,
    int8_onnx: str,
    calibration_images: np.ndarray,
    input_name: str = "input",
    calibrate_method: str = "MinMax",
) -> Tuple[str, dict]:
    """Run static INT8 quantization and return (int8_path, config_record).

    Config record documents activation/weight types, quantization format,
    calibration sample count, and calibration method.
    """
    session = ort.InferenceSession(fp32_onnx, providers=["CPUExecutionProvider"])
    actual_input_name = session.get_inputs()[0].name
    if input_name != actual_input_name:
        input_name = actual_input_name

    reader = CalibrationReader(calibration_images, input_name)

    quantize_static(
        model_input=fp32_onnx,
        model_output=int8_onnx,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QUInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=calibrate_method,
    )

    config_record = {
        "fp32_model": fp32_onnx,
        "int8_model": int8_onnx,
        "quantization_method": "post_training_static",
        "calibration_method": calibrate_method,
        "calibration_samples": int(len(calibration_images)),
        "calibration_source": "train_split",
        "quant_format": "QOperator",
        "weight_type": "QUInt8",
        "activation_type": "QUInt8",
        "input_name": input_name,
        "providers": ["CPUExecutionProvider"],
    }
    return int8_onnx, config_record


def load_calibration_images(cfg: DataConfig, n_samples: int = 512) -> np.ndarray:
    """Build a fixed calibration subset from the training split."""
    train_loader, _, _ = load_cifar10(cfg)
    images, _ = collect_numpy(train_loader, n_samples)
    return images
