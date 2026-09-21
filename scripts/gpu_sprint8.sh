#!/usr/bin/env bash
# Sprint 8, runs INSIDE the TensorRT container on the GPU box:
#   docker run --rm --gpus all --ipc=host -v /opt/edgevision/repo:/workspace/edgevision edgevision:trt scripts/gpu_sprint8.sh
# ByteTrack in the C++ runtime: engine with conf 0.1 (weak detections for the second
# association), unit tests, annotated video + track dump, tracking cost, 12-stream check.
set -uo pipefail
cd /workspace/edgevision || exit 1
CLIP=videos/pedestrian_area_1080p25_h264.mp4
FRAMES=${FRAMES:-300}
WARMUP=${WARMUP:-30}
OUT=benchmarks/results
ENGINE=models/tensorrt/yolo26n_nms_512_fp16_c10.engine
step() { echo; echo "== $* ($(date -u +%H:%M:%S))"; }

step "environment"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
[ -f "$CLIP" ] || ffmpeg -hide_banner -loglevel error -y -i videos/pedestrian_area_1080p25.webm -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -an "$CLIP"

step "engine: NMS in graph, 512, FP16, conf 0.1 (tracking needs low-score detections)"
[ -f "$ENGINE" ] || python3 scripts/export_engines_ultralytics.py --imgsz 512 --precision fp16 --families nms --conf 0.1 --summary $OUT/sprint8_exports.json 2>&1 | grep -E "^\{|rror" | cut -c1-200
ls -la "$ENGINE"

step "build + unit tests (letterbox, nv12, tracker)"
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-75}" >/dev/null
cmake --build cpp/build --parallel 2>&1 | grep -E "error|warning: unused|FAILED|Linking CXX executable edgevision_trt" | head -20
(cd cpp/build && ctest --output-on-failure) | tail -6

step "tracking on the clip: annotated video + track dump (opencv decoder, host frames)"
cpp/build/edgevision_trt --engine "$ENGINE" --source "$CLIP" --decoder opencv --track --frames "$FRAMES" --warmup "$WARMUP" \
    --label s8_track_render --dump-tracks $OUT/tracks_cpp_512.jsonl --render $OUT/tracks_render_512.mp4 | grep -E "^tracking|^backend|^(decode|preprocess|inference|postprocess|tracking|end_to_end)|saved"
ls -la $OUT/tracks_render_512.mp4 $OUT/tracks_cpp_512.jsonl
ffmpeg -hide_banner -loglevel error -y -i $OUT/tracks_render_512.mp4 -vf "select='eq(n\,20)+eq(n\,120)+eq(n\,250)'" -vsync vfr -frames:v 3 $OUT/tracks_frame_%d.png && ls $OUT/tracks_frame_*.png

step "tracking statistics: C++ vs Ultralytics ByteTrack reference"
python3 scripts/track_stats.py $OUT/tracks_cpp_512.jsonl
[ -f $OUT/track_reference.json ] && cat $OUT/track_reference.json || echo "(no reference json synced)"

step "tracking cost: NVDEC, 1 stream, per-stage; then 12 streams aggregate with tracking"
cpp/build/edgevision_trt --engine "$ENGINE" --source "$CLIP" --decoder nvdec --track --frames "$FRAMES" --warmup "$WARMUP" --label s8_track | grep -E "^tracking|^backend|^(decode|preprocess|inference|postprocess|tracking|end_to_end)|saved"
cpp/build/edgevision_trt --engine "$ENGINE" --source "$CLIP" --decoder nvdec --frames "$FRAMES" --warmup "$WARMUP" --label s8_notrack | grep -E "^backend|^(inference|postprocess|end_to_end)|saved"
cpp/build/edgevision_trt --engine "$ENGINE" --source "$CLIP" --decoder nvdec --track --streams 12 --frames "$FRAMES" --warmup "$WARMUP" --label s8_track --no-stage-timing | grep -E "^tracking|^streams=|^end_to_end|saved"

step "tests"
python3 -m pytest -q 2>&1 | tail -2
echo "== done ($(date -u +%H:%M:%S))"
