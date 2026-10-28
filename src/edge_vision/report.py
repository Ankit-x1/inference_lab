"""Generate the final results report from actual saved measurements.

Produces a Markdown comparison table and optional plots. Values must come from
measured benchmark/evaluation files; nothing is estimated or invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib  # optional

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover - plotting is optional
    HAS_MATPLOTLIB = False


def load_manifest(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def comparison_table(fp32_manifest: Dict, int8_manifest: Dict) -> str:
    f = fp32_manifest["metrics"]
    i = int8_manifest["metrics"]

    acc_drop = f["top1_accuracy"] - i["top1_accuracy"]
    lat_red = 100.0 * (f["latency_p50_ms"] - i["latency_p50_ms"]) / f["latency_p50_ms"] if f["latency_p50_ms"] else 0.0
    size_red = 100.0 * (f["model_file_size_mb"] - i["model_file_size_mb"]) / f["model_file_size_mb"] if f["model_file_size_mb"] else 0.0

    lines = [
        "# FP32 vs INT8 Comparison (measured results)",
        "",
        "| Measurement | FP32 | INT8 |",
        "|-------------|------|------|",
        f"| Top-1 accuracy | {f['top1_accuracy']:.4f} | {i['top1_accuracy']:.4f} |",
        f"| Accuracy drop (percentage points) | Reference | {acc_drop:+.4f} |",
        f"| Median latency (ms) | {f['latency_p50_ms']:.3f} | {i['latency_p50_ms']:.3f} |",
        f"| p95 latency (ms) | {f['latency_p95_ms']:.3f} | {i['latency_p95_ms']:.3f} |",
        f"| p99 latency (ms) | {f['latency_p99_ms']:.3f} | {i['latency_p99_ms']:.3f} |",
        f"| Model file size (MB) | {f['model_file_size_mb']:.3f} | {i['model_file_size_mb']:.3f} |",
        f"| Peak process memory (MB) | {f['peak_memory_mb']:.3f} | {i['peak_memory_mb']:.3f} |",
        "",
        f"**Calculated:** accuracy drop = {acc_drop:+.4f} percentage points; "
        f"median latency reduction = {lat_red:.2f}%; model-size reduction = {size_red:.2f}%.",
        "",
    ]
    return "\n".join(lines)


def plot_latency(fp32_manifest: Dict, int8_manifest: Dict, out_path: str) -> str:
    if not HAS_MATPLOTLIB:
        return ""
    f = fp32_manifest["metrics"]
    i = int8_manifest["metrics"]
    labels = ["p50 (ms)", "p95 (ms)", "p99 (ms)"]
    fp32_vals = [f["latency_p50_ms"], f["latency_p95_ms"], f["latency_p99_ms"]]
    int8_vals = [i["latency_p50_ms"], i["latency_p95_ms"], i["latency_p99_ms"]]

    x = range(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar([pos - 0.2 for pos in x], fp32_vals, width=0.4, label="FP32")
    plt.bar([pos + 0.2 for pos in x], int8_vals, width=0.4, label="INT8")
    plt.xticks(x, labels)
    plt.ylabel("Latency (ms)")
    plt.title("Latency percentiles: FP32 vs INT8")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def write_report(
    fp32_manifest_path: str,
    int8_manifest_path: str,
    out_dir: str = "docs",
) -> str:
    fp32 = load_manifest(fp32_manifest_path)
    int8 = load_manifest(int8_manifest_path)

    table = comparison_table(fp32, int8)

    lines = [
        "# Results Report",
        "",
        "All values below come from actual measurements saved during the benchmark "
        "and quantization milestones. No values are estimated or invented.",
        "",
        "## Environment",
        f"- FP32 run: {fp32.get('run_id', 'n/a')} on {fp32['environment'].get('machine', 'n/a')} "
        f"(Python {fp32['environment'].get('python_version', 'n/a')}, "
        f"ONNX Runtime {fp32['environment'].get('onnxruntime_version', 'n/a')})",
        f"- INT8 run: {int8.get('run_id', 'n/a')}",
        "",
        "## Comparison",
        "",
        table,
        "## Limitations",
        "",
        "- CIFAR-10 is small (32x32) and low-resolution; results do not prove "
        "production readiness for live camera footage, object detection, or other domains.",
        "- Latency was measured on one Raspberry Pi 5 under fixed cooling/power "
        "conditions; it is not a universal edge-hardware result.",
        "- No energy or power was measured, so no energy-savings claim is made.",
        "- Results are stable only to the extent repeat runs show; check per-run "
        "manifests for variability.",
        "",
    ]
    report_text = "\n".join(lines)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "results.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    plot_path = out / "latency_comparison.png"
    plot_latency(fp32, int8, str(plot_path))

    return str(report_path)
