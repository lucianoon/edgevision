#!/usr/bin/env bash
# Build-and-smoke check of the C++ runtime, runs INSIDE the TensorRT container on the GPU box:
#   docker run --rm --gpus all --ipc=host -v /opt/edgevision/repo:/workspace/edgevision edgevision:trt scripts/gpu_check_runtime.sh
# Not a measurement: it proves that the runtime builds, that every CTest passes (letterbox, nv12,
# tracker, cli, report), that both decoders, multi-stream, batched and tracking paths still run,
# and that the parity tests against the Python backends hold. A few minutes on a T4.
# Engines are expected in models/tensorrt (scripts/aws/sync-down.sh / S3); missing ones skip their step.
set -uo pipefail
cd /workspace/edgevision || exit 1
CLIP=videos/pedestrian_area_1080p25_h264.mp4
ENGINE=${ENGINE:-models/tensorrt/yolo26n_nms_512_fp16.engine}
ENGINE_640=models/tensorrt/yolo26n_nms_fp16.engine
ENGINE_TRACK=models/tensorrt/yolo26n_nms_512_fp16_c10.engine
ENGINE_B4=models/tensorrt/yolo26n_nms_b4_fp16.engine
BUS=$(python3 -c "from ultralytics.utils import ASSETS; print(ASSETS / 'bus.jpg')")
OUT=benchmarks/results
TMP=$(mktemp -d)
LOG=$OUT/gpu_check_runtime_$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p "$OUT"
exec > >(tee "$LOG") 2>&1
failures=0
step() { echo; echo "== $* ($(date -u +%H:%M:%S))"; }
check() {  # check <name> <command...>: runs, records pass/fail, never aborts the script
    local name=$1; shift
    if "$@"; then echo "PASS $name"; else echo "FAIL $name"; failures=$((failures + 1)); fi
}

step "environment"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cmake --version | head -1; nvcc --version | tail -1
git -C . log --oneline -1
ls models/tensorrt/*.engine 2>/dev/null || echo "no engines in models/tensorrt"
[ -f "$CLIP" ] || ffmpeg -hide_banner -loglevel error -y -i videos/pedestrian_area_1080p25.webm -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -an "$CLIP"

step "build (full runtime) with the compiler's warnings visible"
rm -rf cpp/build
check configure cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-75}"
cmake --build cpp/build --parallel 2>&1 | tee "$TMP/build.log" | grep -E "warning|error|FAILED" | head -40
check build test -x cpp/build/edgevision_trt
echo "warnings in build: $(grep -c "warning" "$TMP/build.log")"

step "CTests: letterbox, nv12, tracker, cli, report"
check ctest ctest --test-dir cpp/build --output-on-failure

step "CLI: --help and argument validation"
check help bash -c 'cpp/build/edgevision_trt --help | head -1 | grep -q "^usage:"'
check bad-arg bash -c '! cpp/build/edgevision_trt --engine x --decoder gstreamer 2>/dev/null'

if [ -f "$ENGINE_640" ]; then
    step "single stream, OpenCV decoder, still image: detections dump + report -> parity test"
    check still cpp/build/edgevision_trt --engine "$ENGINE_640" --source "$BUS" --frames 3 --warmup 1 \
        --dump-detections "$TMP/dets.json" --out "$TMP/still.json"
    python3 -c "import json; d=json.load(open('$TMP/still.json')); print('report keys:', len(d), '| stages:', list(d['stages']))"
    check parity python3 -m pytest -q tests/test_cpp_runtime.py
else
    echo "SKIP still/parity ($ENGINE_640 missing)"
fi

if [ -f "$ENGINE" ]; then
    step "NVDEC, 1 stream and 3 independent streams (short)"
    check nvdec-1 bash -c "cpp/build/edgevision_trt --engine $ENGINE --source $CLIP --decoder nvdec --frames 60 --warmup 10 --out $TMP/nvdec1.json | grep -E '^backend|^end_to_end'"
    check nvdec-3 bash -c "cpp/build/edgevision_trt --engine $ENGINE --source $CLIP --decoder nvdec --streams 3 --frames 60 --warmup 10 --no-stage-timing --out $TMP/nvdec3.json | grep -E '^streams=|aggregate'"
    python3 -c "import json; d=json.load(open('$TMP/nvdec3.json')); print('per_stream entries:', len(d['per_stream']), '| mode:', d['mode'], '| decoder:', d['decoder'])"
else
    echo "SKIP nvdec ($ENGINE missing)"
fi

if [ -f "$ENGINE_TRACK" ]; then
    step "tracking: NVDEC + ByteTrack, then OpenCV + render + JSONL dump (short)"
    check track-nvdec bash -c "cpp/build/edgevision_trt --engine $ENGINE_TRACK --source $CLIP --decoder nvdec --track --frames 60 --warmup 10 --out $TMP/track.json | grep -E '^tracking|^tracking |^end_to_end'"
    check track-render bash -c "cpp/build/edgevision_trt --engine $ENGINE_TRACK --source $CLIP --decoder opencv --track --frames 30 --warmup 5 --dump-tracks $TMP/tracks.jsonl --render $TMP/tracks.mp4 --out $TMP/track_render.json | grep -E '^tracking'"
    python3 scripts/track_stats.py "$TMP/tracks.jsonl" | head -5
    ls -la "$TMP/tracks.mp4"
else
    echo "SKIP tracking ($ENGINE_TRACK missing)"
fi

if [ -f "$ENGINE_B4" ]; then
    step "batched: 4 streams on the batch-4 engine (short)"
    check batched bash -c "cpp/build/edgevision_trt --engine $ENGINE_B4 --source $CLIP --decoder nvdec --streams 4 --batched --frames 40 --warmup 10 --out $TMP/batched.json | grep -E '^batched|aggregate'"
else
    echo "SKIP batched ($ENGINE_B4 missing)"
fi

step "Python suite on the box (TensorRT detector parity included)"
check pytest python3 -m pytest -q

echo
echo "== done ($(date -u +%H:%M:%S)): $failures failure(s); log: $LOG"
exit $((failures > 0))
