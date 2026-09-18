# EdgeVision

Real-time object detection pipeline, evolved in stages:
PyTorch baseline -> ONNX Runtime -> TensorRT (FP32/FP16) -> C++ runtime -> Jetson.

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
