#!/usr/bin/env bash
# Run FP32 / INT8 inference benchmark on the Raspberry Pi 5.
# Usage: ./scripts/benchmark_pi.sh <model.onnx> <run_id> [threads]
# Example: ./scripts/benchmark_pi.sh artifacts/models/model_fp32.onnx fp32_run_1 1
# Requires: python environment with numpy, onnxruntime (CPU) installed.

set -euo pipefail

MODEL="${1:?usage: benchmark_pi.sh <model.onnx> <run_id> [threads]}"
RUN_ID="${2:?usage: benchmark_pi.sh <model.onnx> <run_id> [threads]}"
THREADS="${3:-1}"

if [ ! -f "$MODEL" ]; then
  echo "Model file not found: $MODEL" >&2
  exit 1
fi

echo "Benchmarking $MODEL (run $RUN_ID, threads=$THREADS)"

python - <<PY
import json
import time
from pathlib import Path
from edge_vision.data import DataConfig, load_cifar10
from edge_vision.benchmark import (
    BenchmarkConfig, benchmark, create_session, environment_record, save_results,
)

with open("configs/experiment.json") as f:
    cfg = json.load(f)

data_cfg = DataConfig(
    seed=cfg["seed"],
    batch_size=cfg["benchmark"]["test_samples"],
    num_workers=0,
)
_, _, test_loader = load_cifar10(data_cfg)
images, labels = __import__("edge_vision.data", fromlist=["collect_numpy"]).collect_numpy(
    test_loader, cfg["benchmark"]["test_samples"]
)

session = create_session("$MODEL", int("$THREADS"), ["CPUExecutionProvider"])
bcfg = BenchmarkConfig(
    model_path="$MODEL",
    threads=int("$THREADS"),
    warmup=cfg["benchmark"]["warmup"],
    repetitions=cfg["benchmark"]["repetitions"],
    test_samples=cfg["benchmark"]["test_samples"],
)
results = benchmark(session, images, labels, bcfg)
results["model_path"] = "$MODEL"
env = environment_record()
manifest_path, csv_path = save_results(results, env, "$RUN_ID", output_dir="artifacts/metrics")

print(json.dumps(results["metrics"] if "metrics" in results else {
    k: results[k] for k in ["top1_accuracy","latency_p50_ms","latency_p95_ms","latency_p99_ms","peak_memory_mb","model_file_size_mb"] if k in results
}, indent=2))
print(f"Manifest: {manifest_path}")
print(f"Per-sample CSV: {csv_path}")
PY

echo "Benchmark complete for run $RUN_ID."
