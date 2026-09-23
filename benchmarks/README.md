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

## 2026-09-19 - Sprint 5: hardware decode (NVDEC) - the frame never touches the CPU

Same T4, same engine (`yolo26n_nms_fp16`, NMS in the graph), same C++ runtime. New
`--decoder nvdec`: FFmpeg libavformat/libavcodec with the CUDA hwaccel; frames arrive
as NV12 in device memory and a second kernel (`letterbox_nv12_cuda`) does colour
conversion + letterbox + normalisation straight into the engine input. The H.264 clip
is the VP8 clip re-encoded with libx264 (`videos/pedestrian_area_1080p25_h264.mp4`).
300 frames, 30 warm-up, ms mean. `decode` is outside `end_to_end`; **sustained** =
1000 / (decode + end_to_end) is what one stream can actually reach.

| codec | decoder        | decode | preprocess | inference | end_to_end | e2e no-stage-sync | sustained FPS |
|-------|----------------|-------:|-----------:|----------:|-----------:|------------------:|--------------:|
| VP8   | opencv (CPU)   | 2.7    | 1.0        | 2.4       | 3.5        | -                 | ~160          |
| VP8   | nvdec          | 1.5    | 0.3        | 1.9       | 2.2        | 2.0               | ~285          |
| H.264 | opencv (CPU)   | 2.9    | 1.1        | 2.6       | 3.7        | -                 | ~150          |
| H.264 | nvdec          | 0.4    | 0.3        | 1.8       | 2.1        | 2.0               | **~415**      |
| H.264 | nvdec, RTSP 25 fps live | 36.6* | 0.3 | 2.0      | 2.3        | -                 | (paced by camera) |

\* waiting for the next packet from the camera simulator (MediaMTX + ffmpeg on the host,
RTSP over TCP); the pipeline needs 2.3 ms of the 40 ms frame period, ~6% of one T4.

Detections with nvdec vs opencv decode on the same H.264 frame: same 4 classes, IoU
0.985-0.999 (the two paths differ only in YUV->RGB rounding and chroma interpolation).
CTest `nv12`: kernel vs OpenCV's NV12->BGR path mean < 2/255. 30 Python tests passed.

Observations:

- **Pre-processing went from 1.0 to 0.3 ms**: the 6 MB host->device copy is gone; what
  is left is the kernel itself. Inference also dropped 2.4 -> 1.9 ms because the CPU no
  longer competes with decode and there is less to synchronise on the stream.
- **Decode: 2.9 ms of CPU -> 0.4 ms waiting for NVDEC** for H.264. VP8 on NVDEC is slower
  (1.5 ms); real cameras are H.264/H.265 anyway.
- **One 1080p H.264 stream now costs ~2.4 ms of wall time per frame** (~415 FPS if the
  source were unlimited) vs 6.6 ms with CPU decode and 12 ms in Python with TensorRT
  (Sprint 3). From the Sprint 1 CPU baseline (48 ms) that is 20x.
- At 25 fps a stream uses ~6% of the GPU's time budget: the headroom is for **many
  streams per GPU** (T4 NVDEC handles several 1080p30 H.264 streams), the next thing to
  measure (N decoders feeding one engine, batched).
- Real cameras confirmed through RTSP/TCP via the same libavformat path; the
  `h264 ... co located POCs unavailable` warnings are the decoder joining mid-GOP.

Raw reports: `results/cpp_tensorrt_{opencv,nvdec}_*`, `results/gpu_sprint5_*.log`.

## 2026-09-19 - Sprint 6 (phase A): how many 1080p streams per T4?

`edgevision_trt --streams N`: N independent pipelines, each with its own NVDEC decoder
(libav CUDA hwaccel) and its own TensorRT execution context (batch 1) on its own thread,
all sharing the T4. Same `yolo26n_nms_fp16` engine. GPU and NVDEC utilisation sampled
with `nvidia-smi` every 500 ms during each run (middle half of the samples).

### File source, unpaced (each stream as fast as decode + inference allow), no per-stage syncs

| streams | aggregate FPS | per-stream FPS | e2e mean (ms) | e2e p95 | GPU util | NVDEC util | GPU mem |
|--------:|--------------:|---------------:|--------------:|--------:|---------:|-----------:|--------:|
| 1  | ~500 | 500  | 2.0  | 2.0  | -    | 8%  | 336 MiB  |
| 2  | 480  | 240  | 3.1  | 3.5  | 26%  | 12% | 514 MiB  |
| 4  | 547  | 137  | 5.8  | 6.8  | 78%  | 42% | 876 MiB  |
| 6  | 539  | 90   | 9.2  | 11.5 | 69%  | 38% | 1269 MiB |
| 8  | 538  | 67   | 12.5 | 16.3 | 84%  | 47% | 1629 MiB |
| 12 | 542  | 45   | 19.3 | 27.0 | 90%  | 49% | 2384 MiB |

Stage breakdown at 4 streams (per-stage syncs on): decode 0.7, preprocess 0.2,
**inference 5.7**, postprocess 0.0, e2e 5.9 ms. Single stream inference was 1.8 ms:
four batch-1 contexts serialise on the GPU, so each waits for the others.

### Live RTSP cameras (MediaMTX + ffmpeg simulator on the host, 25 fps each, TCP)

| cameras | per-camera FPS | e2e mean (ms) | e2e p95 | GPU util | NVDEC util |
|--------:|---------------:|--------------:|--------:|---------:|-----------:|
| 1  | 25 (paced) | 2.4  | 2.5  | 5%  | 2%  |
| 4  | 24.3       | 5.3  | 6.1  | 14% | 8%  |
| 8  | 23.6       | 9.9  | 12.4 | 28% | 21% |
| 12 | 24.6       | 14.7 | 19.4 | 48% | 34% |

Observations:

- **The T4 saturates at ~540 frames/s aggregate with batch-1 contexts**, reached already
  at 4 streams; adding streams only divides that budget (per-stream FPS = 540 / N) and
  stretches per-frame latency linearly (~1.6 ms per extra stream). NVDEC is not the
  limit: ~50% at 12 streams.
- **12 live 1080p cameras at 25 fps run with no dropped frames at 48% GPU**, 14.7 ms per
  frame against a 40 ms budget. Extrapolated ceiling with this design: ~20-22 cameras
  per T4 (540 / 25), latency then ~35 ms.
- **The ceiling is inference concurrency, not compute.** yolo26n at batch 1 uses a
  fraction of the T4's SMs (trtexec: 1.7 ms GPU time, i.e. 590 inferences/s single
  stream), and N contexts contend for launches instead of filling the GPU. Batching the
  N letterboxed frames into one `Nx3x640x640` inference (phase B: dynamic-batch engine,
  one enqueue per round, ping-pong input buffers) is the lever to raise the ceiling.
- GPU memory: ~170 MiB per stream (decoder surfaces + engine context); 12 streams = 2.4
  GB of the T4's 15 GB, so memory is not the limit either.

Raw reports: `results/cpp_tensorrt_nvdec_s*_h264_file_*`, `results/cpp_tensorrt_nvdec_s*_rtsp_live_*`,
`results/gpu_sprint6_*.log`, `results/gpu_rtsp_streams_*.log`.

## 2026-09-19 - Sprint 6 (phase B): batched inference across streams

`edgevision_trt --streams N --batched`: N decoder workers letterbox into their slot of
a shared `Nx3x640x640` input (two ping-pong buffers) and a coordinator runs ONE
`enqueueV3` per round on a static-batch engine. Same clip, same T4, FP16.

**Export pitfall (corrected 2026-09-21).** `export(nms=True, dynamic=True)` with the
default `batch=1` produces a graph whose NMS only fills image 0 of a batch (ORT:
`[bus, zidane] -> [5, 0]`). This is not a bug: the exporter unrolls the NMS loop for
`batch` images at trace time and warns `'dynamic=True' export requires a maximum batch
size, e.g. 'batch=16'` (a warning we had filtered out). `export(nms=True, dynamic=True,
batch=4)` is correct for any runtime batch <= 4 (`[5, 3]`, `[5, 3, 5, 3]`, `[3]`), see
`scripts/check_dynamic_nms_export.py`. Phase B used static `batch=N` engines
(`yolo26n_nms_b{4,8,12}_fp16.engine`), which are equally correct; `TrtEngine` supports both.

| N  | design      | aggregate FPS | round / e2e mean (ms) | p95  | max  | trtexec batch-N qps x N |
|---:|-------------|--------------:|----------------------:|-----:|-----:|------------------------:|
| 4  | independent | 529           | 5.9                   | 7.2  | 8.6  | -                       |
| 4  | batched     | 552           | 6.2                   | 6.5  | 6.6  | 131 x 4 = 525           |
| 8  | independent | 536           | 12.7                  | 16.3 | 21.0 | -                       |
| 8  | batched     | 578           | 11.8                  | 12.4 | 12.9 | 68.4 x 8 = 547          |
| 12 | independent | 539           | 19.5                  | 27.0 | 42.0 | -                       |
| 12 | batched     | 562           | 18.4                  | 19.2 | 23.1 | 43.8 x 12 = 526         |

Batched worker means (decode / preprocess): N=4 0.8 / 0.3 ms, N=8 1.5 / 0.5, N=12 1.9 / 1.1;
`gather` (coordinator waiting for the slowest worker) ~0.0 ms: workers always finish
before the previous round's inference does. Detections of the batched path match the
single-stream path (same classes, IoU 0.998).

Observations:

- **Batching buys 4-8% throughput, not a new ceiling.** trtexec itself says why: batch 4,
  8 and 12 all deliver ~525-550 images/s of GPU compute, the same as batch 1 (~590/s
  with no other work). yolo26n FP16 on a T4 costs ~1.8 ms of GPU per 1080p->640 image
  whatever the batch; the phase A result (~540 fps) was already the compute ceiling,
  not a concurrency artefact.
- **What batching does buy is predictability**: p95/max latency drops from 16.3/21.0 to
  12.4/12.9 ms at 8 streams and from 27/42 to 19/23 ms at 12, because one context
  serialises the work instead of N contexts fighting for the GPU.
- **Design decision**: keep the simple independent-context design (phase A) as the
  default; use `--batched` where tail latency matters. To go beyond ~550 1080p frames/s
  per T4 the model, not the pipeline, has to get cheaper: INT8 (T4 tensor cores, needs
  calibration), a smaller input (e.g. 512 or 416, ~1.5-2.3x fewer pixels) or a bigger
  GPU (L4/A10G). That is the next measurable hypothesis.

Raw reports: `results/cpp_tensorrt_nvdec_b*_h264_file_*` (batched),
`results/cpp_tensorrt_nvdec_s*_h264_file_*` (independent), `results/trtexec_b*_fp16_*.log`,
`results/gpu_sprint6b_*.log`.

## 2026-09-21 - Sprint 7: a cheaper model, with the quality measured

Hypotheses from Sprint 6: to go beyond ~550 frames/s per T4 the *model* must get cheaper.
Two levers, each measured for accuracy (mAP on COCO val2017) and speed (our C++ runtime,
NVDEC, T4): **smaller input** (640 -> 512 -> 416) and **INT8** (Ultralytics export,
calibrated on 500 val2017 images disjoint from the 4500 used for mAP; `configs/coco_*.yaml`,
`scripts/get_coco_val.sh`, `scripts/export_engines_ultralytics.py`, `scripts/eval_map.py`).

| imgsz | precision (mAP engine)     | mAP50-95 | mAP50 | mAP75 | 1-stream e2e (ms) | 1-stream FPS | 12-stream aggregate FPS |
|------:|----------------------------|---------:|------:|------:|------------------:|-------------:|------------------------:|
| 640   | FP32 PyTorch (reference)   | **0.404** | 0.565 | 0.439 | 2.0 (FP16 engine) | 492          | 533                     |
| 512   | FP32 PyTorch               | 0.378    | 0.532 | 0.403 | 1.8 (FP16 engine) | 558          | **675**                 |
| 416   | FP32 PyTorch               | 0.350    | 0.497 | 0.376 | 1.7 (FP16 engine) | 594          | 706                     |
| 640   | INT8 (PTQ, raw head)       | 0.367    | 0.523 | 0.399 | -                 | -            | -                       |
| 512   | INT8                       | 0.345    | 0.494 | 0.373 | -                 | -            | -                       |
| 416   | INT8                       | 0.315    | 0.459 | 0.338 | -                 | -            | -                       |

Speed rows use the FP16 NMS-in-graph engines of the same size. mAP on 4500 images, conf 0.001,
iou 0.7, Ultralytics `val`. **Correction (Sprint 9):** this section used to say FP16 costs no
measurable mAP; that was never measured here. Measured in Sprint 9, the FP16 engines lose
0.6-0.7 points (0.399 @ 640, 0.372 @ 512), see below.

Observations:

- **Resolution is the cheap lever.** 640 -> 512 gives +27% aggregate throughput (533 -> 675
  fps, i.e. ~27 cameras at 25 fps instead of ~21) for -2.6 mAP points; 416 gives +32% for
  -5.4 points, with diminishing returns: the pipeline is launch-bound at batch 1 (1.7 ms
  whatever the size), so single-stream latency barely moves.
- **INT8 post-training quantisation costs ~3.5 mAP points at every size** on this nano model
  (0.404 -> 0.367 at 640), more than dropping to 512 costs, and its speed gain could not be
  realised in our pipeline: TensorRT cannot build INT8 for the NMS-in-graph export
  (`Could not find any implementation for node .../cv3.0/.../Conv`, also with 8 GiB
  workspace), so INT8 would need NMS back in the runtime plus QAT to recover accuracy.
  Not worth it here; on a Jetson (INT8-heavy) the calculus changes.
- **Decision:** production default becomes **FP16 at 512** (`yolo26n_nms_512_fp16`):
  0.378 mAP50-95, ~675 frames/s aggregate on a T4. Keep 640 when small objects matter.
- Validation of the harness: the PyTorch 640 reference (0.404) matches Ultralytics' published
  yolo26n COCO mAP (~0.40), so the split, labels and protocol are right.

Ops lessons that cost ~US$ 1.5 of idle GPU: (1) `curl` without `-f` "downloaded" an HTML
error page as val2017.zip; (2) Ultralytics needs a `train:` key even to calibrate; (3) the
PyTorch DataLoader deadlocks in Docker's 64 MB `/dev/shm`: run with `--ipc=host`; (4) never
put `$VAR` in an inline `run.sh` command - use a script in the repo.

Raw: `results/sprint7_map.json`, `results/sprint7_exports*.json`, `results/cpp_tensorrt_nvdec*_s7_*`,
`results/sprint7b_eval.out`.

## 2026-09-21 - Sprint 8: ByteTrack in the C++ runtime

`edgevision_trt --track` runs a ByteTrack tracker per stream (`cpp/src/tracker.cpp`: Kalman
filter on cx/cy/aspect/h + velocities, Hungarian IoU association in two passes - high-score
then low-score detections - unconfirmed-track gate, 30-frame lost buffer, class-aware).
Because the production graph bakes conf 0.5, tracking uses an engine exported with
**conf 0.1** (`yolo26n_nms_512_fp16_c10`) so the weak detections that give ByteTrack its
robustness exist. Unit tests (`cpp/tests/test_tracker.cpp`) cover the Hungarian solver,
Kalman extrapolation, stable ids on two movers, a 10-frame occlusion keeping its id,
loss beyond the buffer getting a new id, low-score detections sustaining but never creating
tracks, and class-aware ids.

![tracking on the pedestrian clip](../docs/tracking_frame.jpg)

### Cost of tracking (T4, NVDEC, 512 FP16, 1080p clip)

| run | tracking stage | end_to_end | 1-stream FPS | 12-stream aggregate |
|---|---:|---:|---:|---:|
| detection only | - | 1.9 ms | 517 | 675 (Sprint 7) |
| detection + ByteTrack | 0.0 ms (tens of us) | 2.0 ms | 508 | 677 |

Tracking is free at this scale (<= ~10 objects per frame): the Hungarian solve on a
10x10 cost matrix and a handful of 8-state Kalman updates cost microseconds.

### Behaviour vs the Ultralytics ByteTrack reference (same clip, 300 frames, imgsz 512, conf 0.1)

| tracker | unique ids | mean track length (frames) | tracks / frame | tracks >= 2 s |
|---|---:|---:|---:|---:|
| Ultralytics ByteTrack (`bytetrack.yaml`: high 0.25, new 0.25) | 85 | 25.6 | 7.25 | 15 |
| edgevision C++ (ByteTrack paper defaults: high 0.5, new 0.6) | 45 | 39.4 | 5.91 | 13 |

Observations:

- **Fewer, longer tracks than the Ultralytics default**, by design of the thresholds: the
  paper's 0.5/0.6 start fewer tracks on marginal detections than Ultralytics' 0.25. Both
  find the same ~13-15 persistent pedestrians; the difference is in short-lived tracks on
  weak detections. `--track-thresh` exposes the knob.
- **Visual check** (`tracks_render_512.mp4`, frames at n=20/120/250): boxes tight, ids stable
  across seconds (#4 keeps its number 100 frames later), bicycle tracked as its own class.
- **Known ByteTrack limitation seen once**: an id (#21) handed over from a pedestrian who
  left to another who walked through the predicted box of the lost track. IoU-only
  association cannot tell them apart; appearance embeddings (BoT-SORT/DeepSORT style)
  would - a possible later step if id purity matters for the use case.
- `--render` (OpenCV decoder, host frames) writes an annotated mp4 at ~50 fps, bound by
  the CPU H.264 encode, not by the pipeline.

Raw: `results/cpp_tensorrt_*_s8_*`, `results/tracks_cpp_512.jsonl`, `results/track_reference.json`.

**Note (Sprint 9):** the C++ side of this table ran with the `--confidence 0.5` default still
applied under `--track`, so detections between 0.1 and 0.5 never reached the tracker. Part of the
"fewer tracks" difference above was that bug, not the thresholds. Fixed in Sprint 9.

## 2026-09-22 - Sprint 9: accuracy of what actually ships

Two claims in this file had no artefact behind them: the mAP of the FP16 engines that run in
production, and any tracking accuracy at all (Sprint 8 only counted ids). One T4 session
(~1 h 45 min, about US$ 1) measured both. Scripts: `scripts/gpu_sprint9*.sh`,
`scripts/get_mot17.sh`, `scripts/mot17_eval.py`.

### mAP of the TensorRT FP16 engines (COCO val2017, same 4500 images and protocol as Sprint 7)

| engine | mAP50-95 | mAP50 | vs FP32 reference |
|---|---:|---:|---:|
| raw head, FP16 @ 640 | 0.399 | 0.554 | -0.6 (0.404) |
| raw head, FP16 @ 512 | **0.372** | 0.522 | -0.7 (0.378) |
| NMS in graph, FP16 @ 512, conf 0.5 (detection default) | 0.257 | 0.326 | operating point |
| NMS in graph, FP16 @ 512, conf 0.1 (tracking engine) | 0.339 | 0.471 | operating point |

FP16 costs 0.6-0.7 points, not zero. The two NMS rows are not comparable with the protocol rows:
the confidence baked into the graph (0.5 / 0.1) truncates the precision-recall curve that mAP
integrates at conf 0.001. They are there to show what the operating point gives up.

### Tracking accuracy: MOT17 train, TrackEval (HOTA / CLEAR / Identity), pedestrians

The 7 MOT17 train sequences (5316 frames; FRCNN copies, the frames are identical in the three),
encoded to near-lossless H.264 for the runtime. Same engine (512, FP16, NMS in graph, conf 0.1).

| tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| edgevision C++, before the fix (track_thresh 0.5) | 26.9 | 16.6 | 43.9 | 18.6 | 27.4 | 136 |
| Ultralytics ByteTrack, `yolo26n.pt` FP32 | 31.2 | 23.3 | 42.1 | 25.2 | 34.0 | 387 |
| Ultralytics ByteTrack on the same TensorRT engine | 31.8 | 24.1 | 42.2 | 26.6 | 35.2 | 358 |
| edgevision C++, fixed, track_thresh 0.5 (paper) | 29.4 | 18.5 | **47.2** | 20.5 | 30.3 | **137** |
| edgevision C++, fixed, track_thresh 0.25 | **32.7** | **24.6** | 44.1 | **27.1** | **37.2** | 275 |

How the gap was found, in order (each step is a versioned log):

1. **track_thresh sweep** (0.25-0.6, `sprint9_sweep.log`): HOTA moved by less than half a point,
   and 0.25 and 0.35 gave *identical* results, which is only possible if no detection scored
   between 0.25 and 0.5.
2. **NMS IoU 0.45 -> 0.7** (Ultralytics' tracking default, `sprint9_iou.log`): no change.
3. **Isolation** (`sprint9_isolate.log`): Ultralytics' ByteTrack on the very engine the C++
   runtime uses scored HOTA 31.8, so the detector and the engine were fine and the gap was in the
   runtime.
4. **Root cause:** `edgevision_trt` kept its `--confidence 0.5` default under `--track`, dropping
   every detection below 0.5 before ByteTrack; the second association (low-score detections) was
   always empty. `--track` now defaults `--confidence` to the tracker's low threshold (0.1), an
   explicit `--confidence` still wins and warns when it starves the second association
   (`cpp/src/cli.cpp`, CTest `test_cli`).

Observations:

- With the fix, the C++ tracker beats Ultralytics' ByteTrack on the same detections on every
  combined metric, with fewer identity switches (275 vs 358).
- The paper's 0.5 keeps the best association and the fewest switches (AssA 47.2, 137 IDSW) at the
  cost of detection recall; 0.25 matches Ultralytics' default. It was chosen on the evaluation
  split itself, with no held-out set: read the table as a sensitivity analysis.
- Absolute numbers are modest because the detector is a COCO nano model at 512 never trained on
  MOT17's crowded, small pedestrians (DetA ~24). MOT17-05 (640x480, closer people) reaches
  HOTA 42.7 / IDF1 57.3.

Raw: `results/sprint9_map.json`, `results/mot17_eval.json` (per sequence), `results/sprint9*.log`.
