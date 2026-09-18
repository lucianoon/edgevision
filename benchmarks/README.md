# Benchmark log

All runs: yolo26n, imgsz 640, conf 0.5, `videos/sample.mp4` (bus.jpg repeated),
100 measured frames. Machine: Windows 11, CPU only (no CUDA), torch 2.14.0+cpu,
onnxruntime 1.30.0 CPUExecutionProvider, ultralytics 8.4.155. Latencies in ms.

## 2026-09-18 - Sprint 2: PyTorch vs ONNX Runtime (CPU)

| backend | warmup | stage       | mean | p50  | p95   | max   |
|---------|--------|-------------|------|------|-------|-------|
| pytorch | 10     | inference   | 71.7 | 48.7 | 137.7 | 579.0 |
| pytorch | 10     | end_to_end  | 72.1 | 49.1 | 138.0 | 579.5 |
| pytorch | 20     | inference   | 47.8 | 47.3 | 52.3  | 54.5  |
| pytorch | 20     | end_to_end  | 48.2 | 47.7 | 52.7  | 54.8  |
| onnx    | 10     | preprocess  | 5.2  | 5.1  | 6.6   | 7.3   |
| onnx    | 10     | inference   | 36.9 | 36.9 | 43.6  | 45.2  |
| onnx    | 10     | postprocess | 2.5  | 2.5  | 3.3   | 4.6   |
| onnx    | 10     | end_to_end  | 45.0 | 45.1 | 52.6  | 53.4  |

FPS (from end_to_end mean): pytorch 20.7, onnx 22.2. Decode ~2.7 ms in all runs.

Observations:

- The Sprint 1 figure (~155 ms/frame) was dominated by warm-up; with a proper
  warm-up both backends settle below 50 ms on this CPU.
- 10 warm-up frames were not enough for PyTorch (one 579 ms outlier, p95 138 ms);
  20 were. ONNX Runtime was stable from the first measured frame.
- ONNX Runtime CPU is ~7% faster end-to-end than PyTorch CPU here. The real gain
  expected from this path is on NVIDIA hardware (TensorRT), not on CPU.
- With our own pipeline, pre + post-processing cost ~7.7 ms (17% of the frame):
  a target for later optimization (e.g. GPU preprocessing) once inference shrinks.

Raw reports: `results/*.json` (include environment details).
