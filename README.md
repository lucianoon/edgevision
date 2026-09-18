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

## Sprint 1: PyTorch baseline

```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "python"
python -m edgevision.main                      # webcam 0, config from configs/app.yaml
python -m edgevision.main --source video.mp4   # file or RTSP URL
python -m edgevision.main --no-display --max-frames 100
pytest
```

Weights (`yolo26n.pt`) are downloaded by Ultralytics on first run into `models/pytorch/`.
