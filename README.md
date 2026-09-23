# EdgeVision

[![ci](https://github.com/lucianoon/edgevision/actions/workflows/ci.yml/badge.svg)](https://github.com/lucianoon/edgevision/actions/workflows/ci.yml)
![license](https://img.shields.io/badge/license-MIT-green)
![tensorrt](https://img.shields.io/badge/TensorRT-10.16-76b900)
![cuda](https://img.shields.io/badge/CUDA-13.2-76b900)

**Real-time multi-camera object detection and tracking, taken from a PyTorch notebook-style
baseline to a C++/CUDA runtime, one measured step at a time.** Every stage of the journey has
a benchmark, a test and a written conclusion, so each optimisation is justified by the
bottleneck the previous measurement exposed.

![ByteTrack on the pedestrian clip](docs/tracking_demo.gif)

<sub>6 s of the 1080p pedestrian clip (CC0) tracked by the C++ runtime on a Tesla T4; the full-resolution frame is in `docs/tracking_frame.jpg`.</sub>

## Highlights (Tesla T4, 1080p H.264, YOLO26n)

| what | result |
|---|---|
| End-to-end latency per frame (decode on NVDEC, CUDA pre-processing, TensorRT FP16, NMS in graph, tracking) | **2.0 ms + 0.4 ms decode wait** |
| Aggregate throughput, 12 independent streams | **675 frames/s** (compute ceiling of the model on this GPU) |
| Live RTSP cameras at 25 fps with zero dropped frames | **12 tested**, ~27 extrapolated, GPU at 48% |
| Accuracy on COCO val2017 (mAP50-95) | FP32 reference 0.404 @ 640 (matches the published figure); **TensorRT FP16 @ 512: 0.372** under the standard protocol (FP16 costs 0.6-0.7 points). The deployed engine bakes NMS into the graph at conf 0.5, where it scores 0.257: that operating point truncates the curve mAP integrates |
| Tracking accuracy, MOT17 train (TrackEval, pedestrians) | **HOTA 32.7, IDF1 37.2, MOTA 27.1** with `--track-thresh 0.25`: above Ultralytics' ByteTrack on the same engine (31.8 / 35.2 / 26.6) |
| Speed-up vs the Python/PyTorch baseline on the same GPU | 12.4 ms -> 2.0 ms per frame (**6x**); vs the CPU baseline 48 ms (**24x**) |
| Tracking cost (ByteTrack, C++, dependency-free) | tens of microseconds per frame |
| Total cloud spend for all GPU measurements | about **US$ 9** (g4dn.xlarge, auto-stopped, zero-cost standby) |

## Architecture

```
 RTSP / file ──► NVDEC (FFmpeg CUDA hwaccel) ──► NV12 frame in GPU memory
                                                      │
                             CUDA kernel: letterbox + BT.601 -> RGB + /255 + NCHW
                                                      │  (writes straight into the engine input)
                             TensorRT 10 engine, FP16 @ 512, NMS inside the graph
                                                      │  1 x 300 x 6  (x1 y1 x2 y2 score class)
                             D2H 7 KB ──► rescale ──► ByteTrack (Kalman + Hungarian, 2 passes)
                                                      │
                                              tracks: id, box, class, score  (JSONL / video)

 N streams: one decoder + one execution context per thread (default), or --batched:
 workers fill slots of a ping-pong N x 3 x S x S buffer and one enqueueV3 serves all.
```

Python (`python/edgevision/`) holds the reference implementation and the benchmark harness;
C++ (`cpp/`) holds the production runtime. Both share the same `Detection` contract and are
checked against each other by parity tests.

## The journey, in measurements

| sprint | step | key number on the T4 | what it taught |
|---|---|---|---|
| 1 | PyTorch/Ultralytics baseline, webcam, FPS counter | 48 ms/frame on a laptop CPU | Warm-up matters: the first figure (155 ms) was noise |
| 2 | Own letterbox + decode + NMS, ONNX Runtime, per-stage metrics | 45 ms (CPU) | Without stage timing you cannot tell where the time goes |
| 3 | TensorRT FP32/FP16 via `trtexec`, Python runtime, GPU box as code | 9.3 ms, 108 fps | FP16 halves GPU time but the frame barely moves: CPU pre/post-processing is 55% |
| 4 prep | NMS inside the graph, real 1080p clip | 7.6 ms, 131 fps | NMS on the GPU is free; pre-processing is now 42% |
| 4 | C++ runtime with a CUDA letterbox kernel, pinned frames | 3.0 ms, 332 fps | Pre-processing 3.5 -> 1.1 ms; CPU video decode is next |
| 5 | NVDEC through libav, RTSP camera simulator | 2.0 ms + 0.4 ms decode, ~415 fps | The frame never touches the CPU; one 25 fps camera uses ~6% of the GPU |
| 6A | N independent streams, live RTSP sweep | 540 fps aggregate ceiling; 12 cameras at 48% GPU | Latency grows 1.6 ms per extra stream; NVDEC is not the limit |
| 6B | Batched inference (static-batch engines, ping-pong buffers) | 552-578 fps (+4-8%) | Batching only tightens tail latency: the model's compute is the ceiling |
| 7 | Accuracy vs speed: input size and INT8, mAP on COCO | 512: +27% throughput for -2.6 mAP; INT8: -3.5 mAP, unbuildable with NMS | Resolution is the cheap lever; INT8 PTQ hurts a nano model |
| 8 | ByteTrack in C++ | tracking ~0 ms; 45 ids / 300 frames | Cheap on the GPU budget; its accuracy was not measured yet (and was hurt by a bug, see 9) |
| 9 | Accuracy of what ships: FP16 engine mAP, MOT17 HOTA/IDF1 | FP16 -0.7 mAP; HOTA 26.9 -> 32.7 after a fix | Measuring found a real bug: `--track` kept the 0.5 score filter, so ByteTrack's second association never saw a detection |

![latency per frame across the sprints](docs/latency_by_sprint.png)

Full tables, raw JSON reports and observations: [`benchmarks/README.md`](benchmarks/README.md).
Every GPU run wrote its report to `benchmarks/results/`, versioned.

## From tracks to events

Detections and tracks are not what a business buys; counts, occupancy and alerts are. The
`edgevision-events` command applies a YAML rule file (zones as polygons, counting lines with a
direction, dwell thresholds, per-class filters) to the track stream the runtime emits and writes
one JSON event per line, optionally POSTing each to a webhook:

```bash
edgevision-events benchmarks/results/tracks_cpp_512.jsonl --rules configs/rules.example.yaml --fps 25 --out events.jsonl
```

On the 12-second pedestrian clip tracked by the C++ runtime (T4, Sprint 8) the example rules
give: gate crossings 6 east / 8 west, 44 entries into the platform zone, 3 people staying
longer than 5 s, 105 events in total. Events: `zone_enter`, `zone_exit` (with dwell and reason:
left / lost / end_of_stream), `dwell_exceeded`, `line_cross` (with direction label). The layer
is pure Python, dependency-free, fully unit-tested with synthetic trajectories and regression-
tested on the real dump (`tests/data/`), and reads stdin so the runtime can be piped in live.

## Observability

Both commands serve Prometheus metrics while they run (`--metrics-port`): fps, per-stage latency
of the current window (mean/p50/p95/max), frames and detections totals from the app; line
crossings, zone entries and occupancy, dwell alerts, active tracks and events per type from the
events layer. `observability/` has a Prometheus + Grafana stack with a provisioned dashboard:

```bash
edgevision --backend onnx --source <video> --no-display --metrics-port 9108
docker compose -f observability/docker-compose.yml up      # Grafana on :3000, dashboard "EdgeVision"
```

The exporter is dependency-free (text exposition over `http.server`, daemon thread, never on the
frame path); `/healthz` answers while the process runs. Details in `observability/README.md`.

## Repository layout

```
python/edgevision/   pip-installable package (`edgevision`, `edgevision-benchmark`, `edgevision-events`):
                     detector.py (Ultralytics), onnx_detector.py, tensorrt_detector.py,
                     events/ (rules.py, geometry.py, engine.py, sinks.py, cli.py), observability.py,
                     preprocess.py, postprocess.py (decode + NMS), factory.py, metrics.py,
                     benchmark.py, main.py
cpp/                 CMake project: letterbox.cu, trt_engine.cpp, video_decoder.cpp (NVDEC),
                     detector.cpp, batch_pipeline.cpp, stream_runner.cpp, tracker.cpp, cli.cpp,
                     report.cpp, main.cpp (wiring only); CTests. -DEDGEVISION_CPU_ONLY=ON builds
                     the GPU-free library (tracker, metrics, names, CLI, report) and its tests
scripts/             ONNX/engine exports, COCO mAP evaluation, tracking reference,
                     gpu_sprint*.sh (one reproducible measurement per sprint)
scripts/aws/         deploy / resume / standby / run / sync for the GPU box
infra/               CloudFormation for the GPU box, scoped IAM policy, cost notes
docker/              TensorRT container (NGC 26.04 + OpenCV, libav, CMake)
benchmarks/          README with every result, results/*.json reports and logs
tests/               pytest: pre/post-processing, parity between backends, C++ runtime
configs/             app.yaml (backend, engine paths), rules.example.yaml (events), COCO yaml
observability/       Prometheus + Grafana compose stack, scrape config, provisioned dashboard
```

## Reproduce

**CPU only (what the CI runs, ~1 min):**

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.lock -e .  # pinned deps + the package
python scripts/export_onnx.py && python scripts/export_onnx.py --nms
pytest -q
edgevision-benchmark --backend onnx --source <video>   # or: edgevision --backend onnx --source 0
```

**GPU (any machine with an NVIDIA GPU and Docker; the repo used an EC2 g4dn.xlarge):**

```bash
docker build -f docker/Dockerfile.tensorrt -t edgevision:trt .
docker run --rm --gpus all --ipc=host -v $PWD:/workspace/edgevision edgevision:trt scripts/gpu_sprint8.sh
```

`infra/README.md` documents the AWS workflow used here: CloudFormation stack, SSM-only
access, and a zero-cost standby that removes the instance and its disk between sessions.

## Engineering practices worth noting

- **Measure before optimising**: each sprint states a hypothesis, the bottleneck it targets and
  the number it moved. Two hypotheses were rejected by their own measurements (batching, INT8).
- **Tests at every layer**: unit tests for the CUDA kernels (against an OpenCV reference), the
  tracker (synthetic scenes with known identities) and the Python pipeline; parity tests
  between PyTorch, ONNX Runtime, TensorRT and the C++ runtime. CI on every push: ruff, mypy
  and shellcheck; the Python suite against pinned dependencies with a coverage floor; the
  GPU-free C++ library built with strict warnings, unit-tested, clang-format and clang-tidy.
- **Infrastructure as code with cost guards**: the GPU box is a CloudFormation template with
  an uptime cap and a standby script; total spend for the whole project was about US$ 9.
- **Findings written down, including the wrong ones**: TensorRT cannot build INT8 for the
  NMS-in-graph export; and a "bug" in Ultralytics' dynamic-batch NMS export turned out to be
  a documented requirement (`batch=<max>`) once the warning was read - corrected in the log.

## Limitations and next steps

- Detects the 80 COCO classes; no faces, plates or behaviour. Rules over tracks (zones,
  counting, dwell, alerts) exist as a post-process on the track dump; feeding them live from the
  C++ runtime (IPC instead of JSONL) and adding MQTT/Kafka sinks is the next step.
- Identity handovers can happen when tracks cross (IoU-only association); appearance
  embeddings (BoT-SORT) would fix that where id purity matters.
- The MOT17 numbers are a sanity check, not a leaderboard entry: train split (test ground truth is
  private), a COCO nano detector never trained on MOT17, and `track_thresh` 0.25 (Ultralytics'
  default; the paper's 0.5 gives HOTA 29.4) picked on that same split, with no held-out set.
- Not yet on a Jetson itself. The aarch64 build is validated on an EC2 g5g (Graviton2 + T4G, the
  same ISA and TensorRT stack as JetPack): full runtime, five CTests, NVDEC, multi-stream, batched
  and tracking pass (`scripts/gpu_check_runtime.sh`, log in `benchmarks/results/`). There the
  Python references (onnxruntime-gpu, torch) run on the CPU: their aarch64 wheels ship no sm_75
  kernels, so `EDGEVISION_ORT_PROVIDERS` / `EDGEVISION_TORCH_DEVICE` force the CPU.

## Contributing

`CONTRIBUTING.md` has the setup, the checks the CI runs (and how to run them locally with
pre-commit), the GPU box workflow and the conventions; `CHANGELOG.md` tracks releases.

## License

MIT. Benchmark clips are CC0 / CC BY (see `videos/README.md`) and are not part of the repository.
