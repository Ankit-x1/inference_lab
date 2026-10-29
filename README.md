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

| Metric | FP32 (measured) | INT8 |
|--------|-----------------|------|
| Top-1 accuracy | 0.71 (71%) | Could not measure |
| Accuracy drop (percentage points) | Reference | N/A |
| Median latency p50 (ms) | ~0.226 | Could not measure |
| p95 latency (ms) | ~0.332 | Could not measure |
| p99 latency (ms) | ~0.399 | Could not measure |
| Model file size (MB) | ~0.056 | Model file exists but crashes on Pi |
| Peak process memory (MB) | ~0.020 | Could not measure |

**INT8 limitation:** The quantized INT8 model was generated successfully on
Windows using ONNX Runtime post-training static quantization, but ONNX Runtime
CPU on this Raspberry Pi 5 segfaults on import and cannot execute the
quantized model. FP32 is the only runnable variant on this device. This is
reported honestly as a hardware/runtime limitation, not suppressed or worked
around.

## Technology stack (version one — nothing else)

| Part | Choice | Why |
|------|--------|-----|
| Language | Python | One language for training, evaluation, export, benchmarking. |
| Environment / dependencies | uv | Repeatable environments; records versions in `uv.lock`. |
| Training framework | PyTorch | Model defined and trained on the development machine (Windows). |
| Dataset | CIFAR-10 | Small, established; downloads via torchvision. |
| Model | Small MobileNet-style CNN | Depthwise-separable convolutions; lightweight, practical. |
| Model export | ONNX | Usable by a separate inference runtime. |
| Inference runtime | ONNX Runtime CPU | Same runtime on PC and Pi; Linux ARM64 CPU support. CPU package only. |
| Quantization | ONNX Runtime post-training static INT8 | Converts FP32 using a calibration sample; no retraining. |
| Testing | pytest | Catches incorrect preprocessing, splits, export problems. |
| Results | CSV + JSON + Markdown | Measurements and an easy-to-inspect report. |
| Git | Git + GitHub | Track each working milestone. |

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
git remote add origin https://github.com/YOUR_USERNAME/edge-vision-inference.git
```

### `.gitignore`
Excludes `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `data/`,
`artifacts/`, `*.pyc`, `.env`, and temporary outputs. `configs/experiment.json`
is committed; generated `artifacts/*.json` and `*.onnx` are ignored.

## Implementation milestones (run in order)

### Milestone 0 — Repository foundation
```powershell
uv sync --dev
uv run pytest
uv run ruff check src tests
```

### Milestone 1 — Raspberry Pi runtime smoke test
On the Pi:
```bash
uname -m
cat /etc/os-release
python3 --version
python -c "import numpy; import onnxruntime; print(onnxruntime.__version__)"
```

### Milestone 2 — CIFAR-10 data pipeline
```powershell
uv run pytest tests/test_data.py -v
```

### Milestone 3 — Train the baseline model
```powershell
uv run python scripts/train.py
uv run pytest tests/test_model.py -v
```

### Milestone 4 — Export and validate ONNX
```powershell
uv run python scripts/export.py
uv run pytest tests/test_export.py -v
```

### Milestone 5 — FP32 inference benchmark (on Raspberry Pi 5)
```bash
./scripts/benchmark_pi.sh artifacts/models/model_fp32.onnx fp32_run_1 1
```

### Milestone 6 — Quantize to INT8 (on Windows)
```powershell
$env:PYTHONPATH="src"
uv run python -c "
from edge_vision.data import DataConfig, load_cifar10, collect_numpy
from edge_vision.quantize import quantize_model
import json
with open('configs/experiment.json') as f:
    cfg = json.load(f)
data_cfg = DataConfig(seed=cfg['seed'], batch_size=128, num_workers=0)
train_loader, _, _ = load_cifar10(data_cfg)
images, _ = collect_numpy(train_loader, cfg['quantization']['calibration_samples'])
quantize_model(
    'artifacts/models/model_fp32.onnx',
    'artifacts/models/model_int8.onnx',
    images,
    calibrate_method=cfg['quantization']['calibration_method']
)
"
```

### Milestone 7 — Controlled systems experiment (extension)
Repeat benchmarks with 1 and 4 inference threads for FP32 and INT8.

### Milestone 8 — Results and report
`docs/results.md` contains the measured comparison table and limitations.

## Reproduction checklist

- [ ] Fresh clone installs dependencies and runs the tests.
- [ ] Train/validation/calibration/test usage is separated.
- [ ] Best validation checkpoint is saved; test accuracy recorded.
- [ ] PyTorch and ONNX Runtime outputs agree within documented tolerance.
- [ ] FP32 results measured on the actual Raspberry Pi 5.
- [ ] INT8 artifact generated; runtime limitation on Pi documented honestly.
- [ ] FP32 and INT8 use the same runtime, samples, preprocessing, method.
- [ ] Report shows actual numbers, variability, limitations, and any failed target.
- [ ] Another developer can recreate the experiment using documented commands.

## Commit convention

Use one commit per working milestone:

1. `chore: initialize uv project and quality checks`
2. `feat: add reproducible CIFAR-10 data pipeline`
3. `feat: train compact MobileNet-style classifier`
4. `feat: export and validate ONNX model`
5. `feat: add reproducible FP32 inference benchmark`
6. `feat: add post-training INT8 quantization`
7. `docs: publish measured FP32 results and document INT8 Pi limitation`

Commit only after the associated tests pass. Keep unrelated formatting changes
out of milestone commits.

## Language guidance

Avoid marketing-style claims. Say "evaluation of post-training INT8
quantization on Raspberry Pi 5" rather than "novel AI inference algorithm."
Report measured latency and model size; do not claim energy savings or
"real-time" performance unless measured.
```
