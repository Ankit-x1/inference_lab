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

## What was actually measured (on real Raspberry Pi 5 hardware)

| Metric                            | FP32 (measured) | INT8                                |
| --------------------------------- | --------------- | ----------------------------------- |
| Top-1 accuracy                    | 0.71 (71%)      | Could not measure                   |
| Accuracy drop (percentage points) | Reference       | N/A                                 |
| Median latency p50 (ms)           | ~0.226          | Could not measure                   |
| p95 latency (ms)                  | ~0.332          | Could not measure                   |
| p99 latency (ms)                  | ~0.399          | Could not measure                   |
| Model file size (MB)              | ~0.056          | Model file exists but crashes on Pi |
| Peak process memory (MB)          | ~0.020          | Could not measure                   |

**INT8 limitation:** The quantized INT8 model was generated successfully on
Windows using ONNX Runtime post-training static quantization, but ONNX Runtime
CPU on this Raspberry Pi 5 segfaults on import and cannot execute the
quantized model. FP32 is the only runnable variant on this device. This is
reported honestly as a hardware/runtime limitation, not suppressed or worked
around.

## Technology stack (version one — nothing else)

| Part                       | Choice                                 | Why                                                                   |
| -------------------------- | -------------------------------------- | --------------------------------------------------------------------- |
| Language                   | Python                                 | One language for training, evaluation, export, benchmarking.          |
| Environment / dependencies | uv                                     | Repeatable environments; records versions in `uv.lock`.             |
| Training framework         | PyTorch                                | Model defined and trained on the development machine (Windows).       |
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
├── .python-version
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
│   ├── train.py                 # Windows: train the baseline classifier
│   ├── export.py                # Windows: export to ONNX + verify
│   └── benchmark_pi.sh          # Raspberry Pi: run FP32 / INT8 benchmark
├── docs/
│   ├── experiment_design.md
│   ├── raspberry_pi_setup.md
│   └── results.md               # measured results
├── artifacts/                   # generated, not committed: checkpoints, models, metrics, plots
└── data/                        # downloaded dataset, not committed
```

Downloaded datasets, trained checkpoints, exported models, and large raw
measurement files are kept out of Git. The repository contains the code needed
to recreate them.

## Setup

### Prerequisites (development machine, Windows)

- Python 3.11 (recommended)
- `uv`

```powershell
mkdir edge-vision-inference
cd edge-vision-inference
git init
uv init --package
uv python pin 3.11
uv add torch torchvision numpy onnx onnxruntime onnxscript
uv add --dev pytest ruff
```

Create the empty GitHub repository, then connect it:

```powershell
git remote add origin https://github.com/Ankit-x1/inference_lab.git
```

### `.gitignore`

Excludes `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `data/`,
`artifacts/`, `*.pyc`, `.env`, and temporary outputs. `configs/experiment.json`
is committed; generated `artifacts/*.json` and `*.onnx` are ignored.
