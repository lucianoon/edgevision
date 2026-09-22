#!/usr/bin/env bash
# Build the four FP16 engines that scripts/gpu_check_runtime.sh uses, with trtexec only (no torch),
# so it works on any GPU box, including aarch64 ones (g5g, Jetson) where no CUDA torch wheel exists.
# Runs INSIDE the TensorRT container:
#   docker run --rm --gpus all -v /opt/edgevision/repo:/workspace/edgevision edgevision:trt scripts/gpu_build_engines_trtexec.sh
# Inputs (models/onnx, exported on any machine with scripts/export_onnx.py --nms):
#   yolo26n_nms.onnx          640, conf 0.5          -> yolo26n_nms_fp16.engine
#   yolo26n_nms_512.onnx      --imgsz 512            -> yolo26n_nms_512_fp16.engine
#   yolo26n_nms_512_c10.onnx  --imgsz 512 --conf 0.1 -> yolo26n_nms_512_fp16_c10.engine (tracking)
#   yolo26n_nms_b4.onnx       --batch 4              -> yolo26n_nms_b4_fp16.engine (batched)
# Class-name sidecars (<engine>.names.json) are versioned in models/tensorrt already.
# Existing engines are kept unless FORCE=1; they are GPU- and TensorRT-version specific.
set -uo pipefail
cd /workspace/edgevision || exit 1
OUT=models/tensorrt
mkdir -p "$OUT" benchmarks/results
failures=0

build() {  # build <onnx> <engine>
    local onnx=models/onnx/$1 engine=$OUT/$2
    if [ ! -f "$onnx" ]; then echo "SKIP $2 (missing $onnx)"; return; fi
    if [ -f "$engine" ] && [ -z "${FORCE:-}" ]; then echo "KEEP $engine"; return; fi
    echo "== $onnx -> $engine ($(date -u +%H:%M:%S))"
    if trtexec --onnx="$onnx" --saveEngine="$engine" --fp16 --warmUp=200 --duration=3 \
        > "benchmarks/results/trtexec_build_${2%.engine}_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1; then
        ls -la "$engine"
    else
        echo "FAIL $engine"; failures=$((failures + 1))
    fi
}

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
uname -m
build yolo26n_nms.onnx yolo26n_nms_fp16.engine
build yolo26n_nms_512.onnx yolo26n_nms_512_fp16.engine
build yolo26n_nms_512_c10.onnx yolo26n_nms_512_fp16_c10.engine
build yolo26n_nms_b4.onnx yolo26n_nms_b4_fp16.engine
echo "== engines built with $failures failure(s)"
exit $((failures > 0))
