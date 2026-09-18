#!/usr/bin/env bash
# Sprint 4 prep, runs INSIDE the TensorRT container on the GPU box:
#   docker run --rm --gpus all -v /opt/edgevision/repo:/workspace/edgevision edgevision:trt scripts/gpu_nms_compare.sh
# Compares the raw-head graph (our NMS in Python) with the graph that has NMS built in,
# for ONNX Runtime CUDA and TensorRT FP16, on a real 1080p clip.
set -euo pipefail
cd /workspace/edgevision
SOURCE=${SOURCE:-videos/pedestrian_area_1080p25.webm}
FRAMES=${FRAMES:-300}
WARMUP=${WARMUP:-30}
LOG=benchmarks/results/gpu_nms_compare_$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p benchmarks/results
exec > >(tee "$LOG") 2>&1

echo "== environment"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python3 -c "import tensorrt, onnxruntime as ort; print('tensorrt', tensorrt.__version__, '| ort', ort.__version__)"
python3 -c "import cv2; c=cv2.VideoCapture('$SOURCE'); print('clip', '$SOURCE', int(c.get(3)), 'x', int(c.get(4)), int(c.get(7)), 'frames')"

echo "== engine for the NMS graph"
[ -f models/tensorrt/yolo26n_nms_fp16.engine ] || scripts/build_engine.sh models/onnx/yolo26n_nms.onnx models/tensorrt
[ -f models/tensorrt/yolo26n_fp16.engine ] || scripts/build_engine.sh models/onnx/yolo26n.onnx models/tensorrt

B="python3 -m edgevision.benchmark --source $SOURCE --frames $FRAMES --warmup $WARMUP"
echo "== benchmarks on $SOURCE"
$B --backend onnx     --model models/onnx/yolo26n.onnx                 --label cuda_raw
$B --backend onnx     --model models/onnx/yolo26n_nms.onnx             --label cuda_nmsgraph
$B --backend tensorrt --model models/tensorrt/yolo26n_fp16.engine      --label fp16_raw
$B --backend tensorrt --model models/tensorrt/yolo26n_nms_fp16.engine  --label fp16_nmsgraph
$B --backend pytorch                                                   --label cuda

echo "== tests"
python3 -m pytest -q
echo "== done: $LOG"
