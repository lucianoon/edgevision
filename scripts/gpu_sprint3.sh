#!/usr/bin/env bash
# Sprint 3 measurement run. Executes INSIDE the TensorRT container on the GPU box:
#   docker run --rm --gpus all -v $PWD:/workspace/edgevision edgevision:trt scripts/gpu_sprint3.sh
# Builds FP32/FP16 engines and benchmarks four backends on the same clip.
set -euo pipefail
cd /workspace/edgevision
FRAMES=${FRAMES:-300}
WARMUP=${WARMUP:-30}
LOG=benchmarks/results/gpu_sprint3_$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p benchmarks/results
exec > >(tee "$LOG") 2>&1

echo "== environment"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c "import tensorrt, torch, onnxruntime as ort; print('tensorrt', tensorrt.__version__, '| torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| ort', ort.__version__, ort.get_available_providers())"

echo "== engines"
scripts/build_engine.sh models/onnx/yolo26n.onnx models/tensorrt

echo "== benchmarks ($FRAMES frames, $WARMUP warmup)"
python3 -m edgevision.benchmark --backend tensorrt --model models/tensorrt/yolo26n_fp32.engine --label fp32 --frames "$FRAMES" --warmup "$WARMUP"
python3 -m edgevision.benchmark --backend tensorrt --model models/tensorrt/yolo26n_fp16.engine --label fp16 --frames "$FRAMES" --warmup "$WARMUP"
python3 -m edgevision.benchmark --backend onnx --label cuda --frames "$FRAMES" --warmup "$WARMUP"
python3 -m edgevision.benchmark --backend pytorch --label cuda --frames "$FRAMES" --warmup "$WARMUP"

echo "== tests"
python3 -m pytest -q

echo "== done: $LOG"
