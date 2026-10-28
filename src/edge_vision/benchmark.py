"""Batch-one inference benchmark.

Measures model-only latency (tensor handed to runtime -> result), top-1
accuracy, p50/p95/p99 latency, model file size, and peak process memory.
The same inputs, preprocessing, runtime, and timing method are used for every
model variant so FP32 and INT8 comparisons are fair.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - onnxruntime required only at benchmark time
    ort = None

try:
    import tracemalloc  # standard library; approximate peak Python memory
except Exception:  # pragma: no cover - tracemalloc is always available in stdlib
    tracemalloc = None


def _require_onnxruntime() -> None:
    if ort is None:
        raise RuntimeError(
            "onnxruntime is required to run benchmarks. Install the CPU package "
            "(onnxruntime) and re-run."
        )


@dataclass
class BenchmarkConfig:
    model_path: str
    input_shape: tuple = (1, 3, 32, 32)
    warmup: int = 10
    repetitions: int = 100
    test_samples: int = 1000
    threads: int = 1
    providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])


def load_inputs_labels(test_loader, n_samples: int):
    images = []
    labels = []
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


def file_size_mb(path: str) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_session(model_path: str, threads: int, providers: List[str]) -> ort.InferenceSession:
    _require_onnxruntime()
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def benchmark(
    session: ort.InferenceSession,
    images: np.ndarray,
    labels: np.ndarray,
    cfg: BenchmarkConfig,
) -> Dict[str, Any]:
    """Run model-only latency and accuracy benchmark on prepared inputs.

    Returns a dict of summary metrics plus per-sample records.
    """
    _require_onnxruntime()
    input_name = session.get_inputs()[0].name
    n = min(len(images), cfg.repetitions) if cfg.repetitions else len(images)
    n = min(n, len(images))

    # Warm-up: not timed.
    for i in range(min(cfg.warmup, n)):
        session.run(None, {input_name: images[i : i + 1]})

    if tracemalloc is not None:
        tracemalloc.start()

    latencies: List[float] = []
    predictions: List[int] = []
    correct = 0

    for i in range(n):
        start = time.perf_counter()
        outputs = session.run(None, {input_name: images[i : i + 1]})[0]
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # milliseconds
        pred = int(np.argmax(outputs[0]))
        predictions.append(pred)
        if pred == int(labels[i]):
            correct += 1

    peak_memory_mb = 0.0
    if tracemalloc is not None:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak / (1024 * 1024)

    latencies_arr = np.array(latencies)
    per_sample = [
        {
            "sample_id": int(i),
            "latency_ms": float(latencies_arr[i]),
            "prediction": int(predictions[i]),
            "label": int(labels[i]),
            "correct": bool(predictions[i] == labels[i]),
        }
        for i in range(n)
    ]

    return {
        "top1_accuracy": correct / n if n > 0 else 0.0,
        "correct": int(correct),
        "total": int(n),
        "latency_p50_ms": float(np.percentile(latencies_arr, 50)),
        "latency_p95_ms": float(np.percentile(latencies_arr, 95)),
        "latency_p99_ms": float(np.percentile(latencies_arr, 99)),
        "latency_mean_ms": float(np.mean(latencies_arr)),
        "latency_std_ms": float(np.std(latencies_arr)),
        "peak_memory_mb": peak_memory_mb,
        "model_file_size_mb": file_size_mb(cfg.model_path),
        "model_sha256": file_sha256(cfg.model_path),
        "threads": cfg.threads,
        "providers": cfg.providers,
        "per_sample": per_sample,
    }


def environment_record() -> Dict[str, Any]:
    _require_onnxruntime()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "onnxruntime_version": ort.__version__,
    }


def save_results(
    results: Dict[str, Any],
    env: Dict[str, Any],
    run_id: str,
    output_dir: str = "artifacts/metrics",
) -> Tuple[str, str]:
    """Save a JSON manifest and a CSV of per-sample records. Returns paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "environment": env,
        "settings": {
            "model_path": results.get("model_path", ""),
            "threads": results.get("threads"),
            "providers": results.get("providers"),
            "test_samples": results.get("total"),
        },
        "metrics": {
            k: results[k]
            for k in [
                "top1_accuracy",
                "correct",
                "total",
                "latency_p50_ms",
                "latency_p95_ms",
                "latency_p99_ms",
                "latency_mean_ms",
                "latency_std_ms",
                "peak_memory_mb",
                "model_file_size_mb",
                "model_sha256",
            ]
            if k in results
        },
    }
    manifest_path = out / f"{run_id}.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    csv_path = out / f"{run_id}.per_sample.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "sample_id",
                "latency_ms",
                "prediction",
                "label",
                "correct",
            ],
        )
        writer.writeheader()
        for row in results.get("per_sample", []):
            writer.writerow(
                {
                    "run_id": run_id,
                    "sample_id": row["sample_id"],
                    "latency_ms": row["latency_ms"],
                    "prediction": row["prediction"],
                    "label": row["label"],
                    "correct": row["correct"],
                }
            )
    return str(manifest_path), str(csv_path)
