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

## 2026-09-18 - Sprint 4 prep: real 1080p clip and NMS inside the graph

Clip: `videos/pedestrian_area_1080p25.webm` (1920x1080, CC0; ~49 candidates >= 0.5 per
frame reduced to ~6 detections by NMS, almost all `person`). Two graphs of the same
yolo26n weights: **raw** (`1x84x8400`, our letterbox + NMS in Python) and **nmsgraph**
(Ultralytics `export(nms=True, conf=0.5, iou=0.45)`, ONNX `NonMaxSuppression` op,
output `1x300x6`; decoded by `postprocess.decode_end2end`). 300 frames, 30 warm-up. ms, mean.

### Laptop CPU (i7-1185G7, onnxruntime CPU)

| graph    | decode | preprocess | inference | postprocess | end_to_end | FPS  |
|----------|-------:|-----------:|----------:|------------:|-----------:|-----:|
| raw      | 5.7    | 6.0        | 39.7      | 2.7         | 48.9       | 20.5 |
| nmsgraph | 5.6    | 5.7        | 40.6      | 0.2         | 46.9       | 21.3 |

### Tesla T4 (g4dn.xlarge, TensorRT 10.16 / CUDA 13.2)

| backend / graph          | decode | preprocess | inference | postprocess | end_to_end | p95  | FPS   |
|--------------------------|-------:|-----------:|----------:|------------:|-----------:|-----:|------:|
| pytorch CUDA             | 1.7    | (inside)   | 11.8      | (inside)    | 12.4       | 15.0 | 81.0  |
| onnx CUDA / raw          | 4.5*   | 6.0        | 6.3       | 2.2         | 14.4       | 18.2 | 69.3  |
| onnx CUDA / nmsgraph     | 2.2    | 5.7        | 6.1       | 0.2         | 12.0       | 15.1 | 83.1  |
| tensorrt FP16 / raw      | 2.0    | 3.1        | 4.2       | 2.1         | 9.5        | 11.8 | 105.7 |
| tensorrt FP16 / nmsgraph | 2.0    | 3.2        | 4.2       | 0.2         | **7.6**    | 9.6  | **130.9** |

\* one 89 ms decode outlier in the first run (cold VP8 decoder); p50 was 2.3 ms.

Observations:

- **NMS in the graph is free on the GPU and removes the Python post-processing**
  (2.1 -> 0.2 ms). TensorRT inference did not change (4.2 ms) with NMS inside, and the
  ONNX Runtime CUDA path lost only 0.2 ms. Decision for the C++ runtime: ship the
  engine with NMS built in; no NMS code to write or maintain in C++.
- **Pre-processing is now the largest CPU cost**: 3.1-3.2 ms on the T4 box (5.7-6.0 ms
  under ONNX Runtime, whose CPU is busier) for a 1080p letterbox + BGR->RGB + NCHW +
  /255 in OpenCV/NumPy. That is 42% of the 7.6 ms frame. Next target: resize/normalize
  on the GPU (CUDA kernel or NPP) fed by a single H2D copy of the raw frame.
- **Decode of 1080p VP8 costs ~2 ms of CPU on the box, 5.7 ms on the laptop.** Real
  cameras (H.264/H.265 RTSP) will need hardware decode (NVDEC via GStreamer) to keep
  this off the CPU - the GStreamer step of the roadmap is also a performance step.
- Per-frame accuracy of the nmsgraph path matches ours: same classes, IoU > 0.95,
  confidence within 0.02 on the sample image (`tests/test_end2end_output.py`).
- Sprint 3 figures on the synthetic clip (9.3 ms) and these on the real clip (9.5 ms) agree;
  the still-image clip did not distort the raw-path numbers, but it hid the decode cost.

Raw reports: `results/*_raw_*`, `results/*_nmsgraph_*`, `results/gpu_nms_compare_*.log`.

## 2026-09-19 - Sprint 4: C++ TensorRT runtime with CUDA pre-processing

Same T4 box, same engine (`yolo26n_nms_fp16.engine`, NMS in the graph) and the same
1080p clip for every row. C++ = `cpp/build/edgevision_trt` (OpenCV decode -> pinned
host frame -> H2D -> CUDA letterbox kernel writing the engine input -> enqueueV3 ->
D2H of 300x6 -> rescale). 300 frames, 30 warm-up, ms mean.

| runtime                               | decode | preprocess | inference | postprocess | end_to_end | p95 e2e | FPS   |
|---------------------------------------|-------:|-----------:|----------:|------------:|-----------:|--------:|------:|
| Python TensorRT (OpenCV/NumPy pre)    | 2.3    | 3.5        | 4.7       | 0.2         | 8.4        | 10.5    | 118.8 |
| C++, pageable host frame              | 2.5    | 1.6        | 2.7       | 0.0         | 4.3        | 5.2     | 233.9 |
| C++, pinned host frame                | 3.0    | 1.1        | 2.6       | 0.0         | 3.7        | 4.5     | 269.5 |
| C++, pinned, no per-stage syncs       | 2.9    | -          | -         | -           | **3.0**    | 3.5     | **332.0** |

Decode is measured outside end_to_end in every runtime (it is the same OpenCV/FFmpeg
VP8 software decoder, 2-3 ms of CPU per 1080p frame).

Observations:

- **Goal met: < 5 ms per frame.** From 8.4 ms (Python, same engine) to 3.0 ms (C++,
  no measurement syncs): 2.8x, and 6.9x the Sprint 3 Python figure with our NMS.
- **Pre-processing fell from 3.5 ms to 1.1 ms** and most of that 1.1 ms is the 6 MB
  H2D copy (pinned) - the letterbox kernel itself is ~0.1 ms at 640x640. Pageable
  memory costs +0.5 ms (staged copy); pinned frames are worth it.
- **"inference" 2.6 ms vs trtexec's 1.7 ms GPU compute** is launch overhead + the
  synchronisation we add to time the stage. Without per-stage syncs the whole frame
  takes 3.0 ms, i.e. the pipeline overlaps as TensorRT intends.
- **Kernel accuracy:** CTest compares the CUDA letterbox with the OpenCV pipeline the
  Python path uses: mean diff < 0.5/255, max <= 3/255 on four geometries; the parity
  test against the Python ONNX detector on bus.jpg passes (same classes, IoU > 0.9).
- **Decode is now the largest CPU cost (2.5-3 ms) and it caps a single stream at
  ~330 FPS regardless of the GPU.** Next: hardware decode (NVDEC through GStreamer or
  OpenCV cudacodec) so the frame never touches host memory, and batching several
  streams per enqueue.

Raw reports: `results/cpp_tensorrt_*`, `results/tensorrt_fp16_nmsgraph_20260919*`,
`results/gpu_sprint4_*.log`.
