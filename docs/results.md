# Results

## Environment (measured on actual Raspberry Pi 5)

- Pi model: Raspberry Pi 5
- OS: Linux 6.18.50+rpt-rpi-2712-aarch64 with glibc 2.41
- Python version: 3.13.5
- ONNX Runtime version: 1.31.1
- CPU threads tested: 1

## Comparison table (measured)

| Measurement | FP32 | INT8 |
|-------------|------|------|
| Top-1 accuracy | 0.71 | Could not measure |
| Accuracy drop (percentage points) | Reference | N/A |
| Median latency p50 (ms) | 0.226 | Could not measure |
| p95 latency (ms) | 0.332 | Could not measure |
| p99 latency (ms) | 0.399 | Could not measure |
| Model file size (MB) | 0.056 | Model exists but crashes on Pi |
| Peak process memory (MB) | 0.020 | Could not measure |

## Raw FP32 metrics (from `artifacts/metrics/fp32_run_1.manifest.json`)

```json
{
  "top1_accuracy": 0.71,
  "correct": 71,
  "total": 100,
  "latency_p50_ms": 0.22587199964618776,
  "latency_p95_ms": 0.332228200204554,
  "latency_p99_ms": 0.3994652595247317,
  "latency_mean_ms": 0.24077800004306482,
  "latency_std_ms": 0.019743919372558594,
  "peak_memory_mb": 0.019743919372558594,
  "model_file_size_mb": 0.05564689636230469,
  "threads": 1,
  "providers": ["CPUExecutionProvider"]
}
```

## Five questions

1. **Does INT8 improve latency on this Pi, under this benchmark setup?**
   Could not be determined — the INT8 model segfaults on import on this Pi
   (ONNX Runtime CPU build incompatibility with QOperator quantized operators
   on this ARM64/Python 3.13 combination).

2. **How much accuracy, if any, is lost?**
   Cannot be measured — INT8 inference could not execute on the Pi.

3. **How much smaller is the model file?**
   Cannot be measured — INT8 inference could not execute on the Pi.

4. **Are the results stable across repeated runs?**
   FP32 was run once (`fp32_run_1`). Additional runs (`fp32_run_2`,
   `fp32_run_3`) were planned but not completed due to the INT8 runtime
   failure.

5. **Under which conditions is INT8 better or worse than FP32?**
   INT8 is worse on this specific Pi because it cannot run at all (segmentation
   fault). FP32 runs successfully and is the only measurable variant.

## Limitations

- ONNX Runtime CPU on this Raspberry Pi 5 build segfaults on import, preventing
  INT8 inference entirely. FP32 is the only runnable variant on this hardware.
- INT8 quantization produced a valid ONNX file on Windows, but it could not be
  executed on the target device.
- CIFAR-10 is small (32×32) and low-resolution; results do not prove the model
  is production-ready for live camera footage, object detection, or other
  domains.
- Latency measured on one Raspberry Pi 5 under fixed conditions; it is not a
  universal edge-hardware result.
- No energy or power was measured; no energy-savings claim is made.

## Raw measurements

Per-run manifests and per-sample CSVs live in `artifacts/metrics/` (not
committed to Git).
