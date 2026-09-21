# EdgeVision

[![ci](https://github.com/lucianoon/edgevision/actions/workflows/ci.yml/badge.svg)](https://github.com/lucianoon/edgevision/actions/workflows/ci.yml) MIT License

Real-time object detection pipeline, evolved in stages:
PyTorch baseline -> ONNX Runtime -> TensorRT (FP32/FP16) -> C++ runtime -> NVDEC/RTSP -> multi-stream -> accuracy vs speed (INT8, input size) -> Jetson.

## Setup

Python 3.12 (pinned in `.python-version`). Python 3.11.0rc1 and 3.13+ break
torch / TLS on this machine.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Run

```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "python"
python -m edgevision.main                                  # webcam 0, backend from configs/app.yaml
python -m edgevision.main --backend onnx --source video.mp4
python -m edgevision.main --no-display --max-frames 100
pytest
```

Weights (`yolo26n.pt`) are downloaded by Ultralytics on first run into `models/pytorch/`.

## Backends

| backend   | inference          | pre/post-processing            |
|-----------|--------------------|--------------------------------|
| `pytorch` | Ultralytics `predict()` | inside Ultralytics (opaque) |
| `onnx`    | ONNX Runtime       | ours: `preprocess.py` (letterbox), `postprocess.py` (decode + NMS) |
| `tensorrt`| TensorRT engine (`tensorrt_detector.py`, TensorRT 10 API + cuda-python) | same as `onnx` |
| C++ `cpp/` | TensorRT 10 C++ API (`edgevision_trt`) | CUDA letterbox kernel, NMS in the graph |

Export the ONNX graph (static `1x3x640x640`, opset 17, output `1x84x8400`):

```powershell
python scripts/export_onnx.py
```

## Benchmark

```powershell
python -m edgevision.benchmark --backend pytorch --frames 100
python -m edgevision.benchmark --backend onnx --frames 100
```

Prints mean / p50 / p95 / max per stage (decode, preprocess, inference,
postprocess, end_to_end) and writes a JSON report with environment details to
`benchmarks/results/`. Results in `benchmarks/README.md`.

## TensorRT (Sprint 3)

Needs an NVIDIA GPU. This laptop has none, so engines are built and measured on an
EC2 GPU box defined in `infra/gpu-dev.yaml`; see `infra/README.md` for the workflow.
`scripts/build_engine.sh` builds FP32 and FP16 engines with `trtexec` from the ONNX
export and keeps trtexec's own timing reports in `benchmarks/results/`;
`scripts/gpu_sprint3.sh` runs the whole measurement inside the container.

First results on a Tesla T4 (end-to-end per frame, mean): PyTorch CUDA 12.4 ms,
ONNX Runtime CUDA 11.3 ms, TensorRT FP32 10.6 ms, TensorRT FP16 9.3 ms (108 FPS).
GPU compute for FP16 is 1.7 ms; CPU-side letterbox + NMS now dominate the frame.
Details in `benchmarks/README.md`.

### NMS inside the graph (Sprint 4 prep)

`python scripts/export_onnx.py` exports the raw head; adding NMS to the graph
(`YOLO(...).export(format="onnx", nms=True, conf=0.5, iou=0.45)`) gives a `1x300x6`
output that `postprocess.py` recognises by shape and decodes without running NMS.
On the T4 with a real 1080p clip this took TensorRT FP16 from 9.5 to 7.6 ms per frame
(131 FPS); `configs/app.yaml` now points at the NMS graphs. Benchmark clips and their
licenses: `videos/README.md`.

## C++ runtime (Sprint 4)

`cpp/` holds a CMake project (C++17 + CUDA, TensorRT 10, OpenCV) that reproduces the
detector with a CUDA letterbox kernel writing straight into the engine input and an
engine with NMS in the graph, so there is no CPU post-processing. It builds inside the
TensorRT container on the GPU box (`scripts/gpu_sprint4.sh`), has a CTest for the kernel
and a pytest parity check against the Python ONNX path.

```bash
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=75   # T4; Orin = 87
cmake --build cpp/build && (cd cpp/build && ctest)
cpp/build/edgevision_trt --engine models/tensorrt/yolo26n_nms_fp16.engine     --source videos/pedestrian_area_1080p25.webm --frames 300 --warmup 30
```

Tesla T4, 1080p clip: 3.0 ms per frame end-to-end (332 FPS) vs 8.4 ms for the Python
TensorRT path on the same engine. Details in `benchmarks/README.md`.

## Hardware decode and RTSP (Sprint 5)

`edgevision_trt --decoder nvdec` decodes with NVDEC through FFmpeg's CUDA hwaccel; the
NV12 frame stays on the GPU and `letterbox_nv12_cuda` feeds the engine directly. Files
and RTSP/RTMP/HTTP URLs work alike (`--source rtsp://...`). The container must expose
NVDEC (`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`, set in the Dockerfile).
`scripts/rtsp_sim.sh` runs a MediaMTX + ffmpeg camera simulator on the host.

Tesla T4, 1080p H.264: 2.0 ms end-to-end + 0.4 ms decode wait per frame (~415 FPS
sustained for one stream) vs 3.7 + 2.9 ms with CPU decode. Details in `benchmarks/README.md`.

## Multiple streams per GPU (Sprint 6, phase A)

`edgevision_trt --streams N [--source ...]...` runs N independent decoder + TensorRT
context pipelines on threads and reports per-stream and aggregate throughput.
`scripts/gpu_sprint6.sh` sweeps N on a file source with GPU/NVDEC utilisation sampling;
`scripts/gpu_rtsp_streams.sh` (host side) does the same against N live RTSP cameras
from `scripts/rtsp_sim.sh`.

`--batched` (phase B) shares one static-batch engine across the streams (one
`enqueueV3` per round, ping-pong input buffers); the ONNX must be exported with
`batch=N` because Ultralytics' dynamic-batch NMS export only fills image 0.

Tesla T4: ~540-580 1080p frames/s aggregate whatever the design (that is the GPU compute
ceiling of yolo26n FP16, ~1.8 ms/image); batching mainly tightens tail latency (p95
16 -> 12 ms at 8 streams). 12 live 1080p cameras at 25 fps run with no drops at 48%
GPU. Details in `benchmarks/README.md`.

## Accuracy vs speed (Sprint 7)

`scripts/get_coco_val.sh` + `scripts/eval_map.py` measure COCO mAP; `scripts/export_engines_ultralytics.py`
exports FP16/INT8 engines at several input sizes (INT8 calibrated on a disjoint split). On the
T4: 640 -> 512 buys +27% throughput for -2.6 mAP50-95 points; INT8 PTQ costs ~3.5 points at any
size and cannot be built with NMS in the graph. Production default: **FP16 at 512**. Details in
`benchmarks/README.md`.
