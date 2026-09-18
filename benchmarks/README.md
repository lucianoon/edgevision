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

## 2026-09-18 - Sprint 3: GPU (EC2 g4dn.xlarge, Tesla T4 16 GB)

Driver 595.91 / CUDA 13.2, container `nvcr.io/nvidia/tensorrt:26.04-py3`
(TensorRT 10.16.1), torch 2.14.0+cu130, onnxruntime-gpu 1.30.0 (CUDAExecutionProvider).
Same clip, 300 measured frames after 30 warm-up frames. Latencies in ms (mean).

| backend            | preprocess | inference | postprocess | end_to_end | p95 e2e | FPS   |
|--------------------|-----------:|----------:|------------:|-----------:|--------:|------:|
| pytorch (CUDA)     | (inside)   | 11.9      | (inside)    | 12.4       | 13.0    | 80.6  |
| onnx (CUDA EP)     | 3.1        | 6.1       | 2.1         | 11.3       | 11.5    | 88.6  |
| tensorrt FP32      | 3.0        | 5.5       | 2.1         | 10.6       | 10.9    | 94.6  |
| tensorrt FP16      | 3.0        | 4.1       | 2.1         | 9.3        | 9.8     | 108.1 |

Decode of the 810x1080 clip: ~1.0 ms in every run.

trtexec's own measurement of the same engines (batch 1, 15 s runs, `--separateProfileRun`):

| engine | GPU compute | H2D | D2H | latency (mean) | throughput | build time | size   |
|--------|------------:|----:|----:|---------------:|-----------:|-----------:|-------:|
| FP32   | 3.38 ms     | 0.80| 0.43| 4.62 ms        | 295 qps    | 83 s       | 12.1 MB|
| FP16   | 1.68 ms     | 0.83| 0.44| 2.95 ms        | 595 qps    | 283 s      | 7.1 MB |

Observations:

- **FP16 halves GPU compute** (3.38 -> 1.68 ms, trtexec) with no loss on the parity
  test (every FP16 box within IoU > 0.9 and 0.05 confidence of the ONNX reference).
- **The bottleneck moved off the GPU.** In our Python pipeline the model is 4.1 ms of a
  9.3 ms frame; letterbox + NMS on the CPU cost 5.1 ms (55%), and H2D/D2H copies plus
  Python dispatch add ~1.2 ms over trtexec's compute time. Speeding the model up
  further buys little until pre/post-processing moves to the GPU or to C++.
- TensorRT FP32 vs ONNX Runtime CUDA differ by only 0.6 ms of inference on the T4;
  the ONNX Runtime CUDA EP is already a strong baseline for a small model like yolo26n.
- PyTorch/Ultralytics on CUDA (11.9 ms, own pre/post inside) is the slowest path, as in
  the CPU runs, but the gap to TensorRT FP16 is 1.3x end-to-end, not the 3-5x often quoted:
  the quoted numbers compare *model* time, not *frame* time.
- CPU (Sprint 2) -> T4 FP16: end-to-end 45 ms -> 9.3 ms (4.8x); inference 37 -> 4.1 ms (9x).

Raw reports: `results/tensorrt_*`, `results/onnx_cuda_*`, `results/pytorch_cuda_*`,
`results/trtexec_*` and the run log `results/gpu_sprint3_*.log`.
