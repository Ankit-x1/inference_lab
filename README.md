# Edge Vision Inference Lab

Accuracy, latency, and model-size trade-offs on Raspberry Pi 5.

## Research question

Does converting a trained image classifier from FP32 to INT8 improve inference
performance on a Raspberry Pi 5, while keeping classification accuracy within
an acceptable limit?

This project applies a real optimization technique (post-training static INT8
quantization), designs a controlled experiment, and produces trustworthy
evidence on actual hardware. It does not claim to invent quantization or any
novel algorithm.

## Final output

A working classifier, a repeatable benchmark, a tested quantized model, raw
measurement files, a results report, and a repository another developer can
reproduce.

## Technology stack (version one — nothing else)

| Part                       | Choice                                 | Why                                                                   |
| -------------------------- | -------------------------------------- | --------------------------------------------------------------------- |
| Language                   | Python                                 | One language for training, evaluation, export, benchmarking.          |
| Environment / dependencies | uv                                     | Repeatable environments; records versions in `uv.lock`.             |
| Training framework         | PyTorch                                | Model defined and trained on the development machine.                 |
| Dataset                    | CIFAR-10                               | Small, established; downloads via torchvision.                        |
| Model                      | Small MobileNet-style CNN              | Depthwise-separable convolutions; lightweight, practical.             |
| Model export               | ONNX                                   | Usable by a separate inference runtime.                               |
| Inference runtime          | ONNX Runtime CPU                       | Same runtime on PC and Pi; Linux ARM64 CPU support. CPU package only. |
| Quantization               | ONNX Runtime post-training static INT8 | Converts FP32 using a calibration sample; no retraining.              |
| Testing                    | pytest                                 | Catches incorrect preprocessing, splits, export problems.             |
| Results                    | CSV + JSON + Markdown                  | Measurements and an easy-to-inspect report.                           |
| Git                        | Git + GitHub                           | Track each working milestone.                                         |

Not in version one: TensorFlow, TFLite, TensorRT, CUDA, Docker, Kubernetes,
web dashboard, reinforcement learning, early exit, pruning, distillation,
camera integration, cloud deployment.

Both FP32 and INT8 run through ONNX Runtime on the Pi, so the quantization
comparison is not confounded by a change in software engine.

## Repository structure

```
edge-vision-inference/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── configs/
│   └── experiment.json          # seeds, splits, training, quantization, benchmark settings
├── src/
│   └── edge_vision/
│       ├── __init__.py
│       ├── data.py              # CIFAR-10 loading, train/val/test splits, transforms
│       ├── model.py             # Small MobileNet-style CNN
│       ├── train.py             # Supervised training loop (AdamW, cross-entropy, best checkpoint)
│       ├── evaluate.py          # Top-1 and per-class accuracy
│       ├── export_onnx.py       # PyTorch -> ONNX export + correctness check
│       ├── quantize.py          # Post-training static INT8 quantization
│       ├── benchmark.py         # Batch-one latency/accuracy/memory benchmark
│       └── report.py            # Comparison table and plots from saved measurements
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_export.py
│   └── test_benchmark.py
├── scripts/
│   ├── train.ps1                # Windows: train the baseline classifier
│   ├── export.ps1               # Windows: export to ONNX + verify
│   └── benchmark_pi.sh          # Raspberry Pi: run FP32 / INT8 benchmark
├── docs/
│   ├── experiment_design.md
│   ├── raspberry_pi_setup.md
│   └── results.md               # measured results (template until Pi runs done)
├── artifacts/                   # generated, not committed: checkpoints, models, metrics, plots
└── data/                        # downloaded dataset, not committed
```

Downloaded datasets, trained checkpoints, exported models, and large raw
measurement files are kept out of Git. The repository contains the code needed
to recreate them, plus small example summaries.

## Setup

### Prerequisites (development machine, Windows)

- Python 3.11 (recommended initial target)
- `uv`

```bash
mkdir edge-vision-inference
cd edge-vision-inference
git init
uv init --package
uv python pin 3.11
uv add torch torchvision numpy onnx onnxruntime
uv add --dev pytest ruff
```

Create the empty GitHub repository (for example `edge-vision-inference`), then
connect it using the remote URL GitHub provides. Do not merge this into an
existing `rl_vision` repository; keep it separate.

### `.gitignore`

Excludes `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `data/`,
`artifacts/`, `*.pyc`, `.env`, and temporary outputs.

## Implementation milestones (run in order)

### Milestone 0 — Repository foundation

```bash
uv sync --dev
uv run pytest
uv run ruff check src tests
```

Complete when: environment installs cleanly, imports work, tests pass, minimal
README with setup instructions exists.

### Milestone 1 — Raspberry Pi runtime smoke test

On the Pi:

```bash
uname -m
cat /etc/os-release
python3 --version
python -c "import numpy; import onnxruntime; print(onnxruntime.__version__)"
```

Then export a tiny ONNX model on Windows (`scripts/export.ps1`), copy it to
the Pi, load it with ONNX Runtime CPU, and confirm valid numerical output.

Complete when: the Pi can load an ONNX model and return valid output.

### Milestone 2 — CIFAR-10 data pipeline

```bash
uv run pytest tests/test_data.py -v
```

Complete when: a batch loads with shape `[B, 3, 32, 32]`; labels are 0–9;
split sizes are 45,000 / 5,000 / 10,000; the split is reproducible with seed
42; validation reads are deterministic.

Splits come from `configs/experiment.json`. Training gets random crop +
horizontal flip only; validation/calibration/testing use deterministic
preprocessing. Test data is never used for calibration or checkpoint choice.

### Milestone 3 — Train the baseline model

```bash
.\scripts\train.ps1
uv run pytest tests/test_model.py -v
```

Complete when: a best validation checkpoint is saved
(`artifacts/checkpoints/best.pt`), test accuracy is recorded, and tests prove
model output has shape `[B, 10]` and consists of finite values.

### Milestone 4 — Export and validate ONNX

```bash
.\scripts\export.ps1
uv run pytest tests/test_export.py -v
```

Complete when: PyTorch, ONNX Runtime on Windows, and (after copying to the Pi)
ONNX Runtime on the Pi return consistent results on the same test images
within the documented tolerance (`1e-4` max absolute logit difference).

### Milestone 5 — FP32 inference benchmark

```bash
./scripts/benchmark_pi.sh artifacts/models/model_fp32.onnx fp32_run_1 1
./scripts/benchmark_pi.sh artifacts/models/model_fp32.onnx fp32_run_2 1
uv run pytest tests/test_benchmark.py -v
```

Complete when: the benchmark runs the FP32 model and writes one machine-readable
record per measured inference plus a summary with accuracy and latency
percentiles (`artifacts/metrics/`). This is the baseline.

Metrics recorded: top-1 accuracy, median (p50) latency, p95, p99, model file
size, peak process memory. Batch size 1; warm-up before timing; model-only
latency (tensor handed to runtime → result).

### Milestone 6 — Quantize to INT8

```bash
uv run python -m edge_vision.quantize ...
# (driver script wires load_cifar10 -> collect_numpy calibration -> quantize_static)
```

The sequence: load the exported FP32 ONNX model; calibrate on a fixed sample
of ~512 training images (never test data); produce a separate INT8 ONNX model;
validate it loads and produces finite outputs; evaluate on the same held-out
test set; benchmark on the Pi with the same procedure.

Quantization config recorded: MinMax calibration, QOperator format, QUInt8
weights and activations, 512 calibration samples from the train split.

### Milestone 7 — Controlled systems experiment (extension)

Repeat benchmarks with 1 thread and 4 inference threads for FP32 and INT8,
keeping all other settings fixed. Record whether thread count alters the
FP32 vs INT8 comparison.
