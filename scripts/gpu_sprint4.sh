#!/usr/bin/env bash
# Sprint 4, runs INSIDE the TensorRT container on the GPU box:
#   docker run --rm --gpus all -v /opt/edgevision/repo:/workspace/edgevision edgevision:trt scripts/gpu_sprint4.sh
# Builds the C++ runtime, runs its unit test, benchmarks it against the Python TensorRT
# path on the same engine and clip, and runs the parity tests.
set -euo pipefail
cd /workspace/edgevision
SOURCE=${SOURCE:-videos/pedestrian_area_1080p25.webm}
FRAMES=${FRAMES:-300}
WARMUP=${WARMUP:-30}
ENGINE=models/tensorrt/yolo26n_nms_fp16.engine
LOG=benchmarks/results/gpu_sprint4_$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p benchmarks/results
exec > >(tee "$LOG") 2>&1

echo "== environment"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cmake --version | head -1; nvcc --version | tail -1
pkg-config --modversion opencv4 2>/dev/null || echo "opencv4 pkg-config missing"

echo "== engine"
[ -f "$ENGINE" ] || scripts/build_engine.sh models/onnx/yolo26n_nms.onnx models/tensorrt

echo "== build C++ runtime"
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-75}"
cmake --build cpp/build --parallel
(cd cpp/build && ctest --output-on-failure)

echo "== C++ runtime benchmarks on $SOURCE"
cpp/build/edgevision_trt --engine "$ENGINE" --source "$SOURCE" --frames "$FRAMES" --warmup "$WARMUP" --label fp16_nmsgraph
cpp/build/edgevision_trt --engine "$ENGINE" --source "$SOURCE" --frames "$FRAMES" --warmup "$WARMUP" --label fp16_nmsgraph_nostage --no-stage-timing
cpp/build/edgevision_trt --engine "$ENGINE" --source "$SOURCE" --frames "$FRAMES" --warmup "$WARMUP" --label fp16_nmsgraph_pageable --no-pinned

echo "== Python TensorRT path, same engine and clip"
python3 -m edgevision.benchmark --backend tensorrt --model "$ENGINE" --label fp16_nmsgraph --source "$SOURCE" --frames "$FRAMES" --warmup "$WARMUP"

echo "== tests"
python3 -m pytest -q
echo "== done: $LOG"
