#!/usr/bin/env bash
# Sprint 5, runs INSIDE the TensorRT container on the GPU box (needs NVDEC: container
# started with NVIDIA_DRIVER_CAPABILITIES including 'video' - set in the Dockerfile):
#   docker run --rm --gpus all -v /opt/edgevision/repo:/workspace/edgevision edgevision:trt scripts/gpu_sprint5.sh
# Hardware decode (NVDEC via libav) vs CPU decode, VP8 and H.264, same engine.
set -euo pipefail
cd /workspace/edgevision
CLIP_VP8=${SOURCE:-videos/pedestrian_area_1080p25.webm}
CLIP_H264=videos/pedestrian_area_1080p25_h264.mp4
FRAMES=${FRAMES:-300}
WARMUP=${WARMUP:-30}
ENGINE=models/tensorrt/yolo26n_nms_fp16.engine
LOG=benchmarks/results/gpu_sprint5_$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p benchmarks/results
exec > >(tee "$LOG") 2>&1

echo "== environment"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
ffmpeg -hide_banner -hwaccels | tr '\n' ' '; echo
ffmpeg -hide_banner -decoders 2>/dev/null | grep -E "cuvid|nvdec" | head -5 || echo "no cuvid decoders listed (hwaccel nvdec may still work)"
ls -la /usr/lib/x86_64-linux-gnu/libnvcuvid* 2>/dev/null || echo "libnvcuvid NOT visible: NVDEC will fail"

echo "== engine"
[ -f "$ENGINE" ] || scripts/build_engine.sh models/onnx/yolo26n_nms.onnx models/tensorrt

echo "== H.264 version of the clip (what real cameras send)"
[ -f "$CLIP_H264" ] || ffmpeg -hide_banner -loglevel error -y -i "$CLIP_VP8" -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -an "$CLIP_H264"
ls -la "$CLIP_VP8" "$CLIP_H264"

echo "== build C++ runtime"
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-75}"
cmake --build cpp/build --parallel
(cd cpp/build && ctest --output-on-failure)

B="cpp/build/edgevision_trt --engine $ENGINE --frames $FRAMES --warmup $WARMUP"
echo "== VP8 clip: CPU decode (opencv) vs NVDEC"
$B --source "$CLIP_VP8"  --decoder opencv --label vp8
$B --source "$CLIP_VP8"  --decoder nvdec  --label vp8
$B --source "$CLIP_VP8"  --decoder nvdec  --label vp8_nostage --no-stage-timing
echo "== H.264 clip: CPU decode (opencv) vs NVDEC"
$B --source "$CLIP_H264" --decoder opencv --label h264
$B --source "$CLIP_H264" --decoder nvdec  --label h264
$B --source "$CLIP_H264" --decoder nvdec  --label h264_nostage --no-stage-timing

echo "== detections: nvdec vs opencv on the first measured H.264 frame"
$B --source "$CLIP_H264" --decoder opencv --frames 1 --warmup 0 --out /tmp/r1.json --dump-detections /tmp/dets_opencv.json >/dev/null
$B --source "$CLIP_H264" --decoder nvdec  --frames 1 --warmup 0 --out /tmp/r2.json --dump-detections /tmp/dets_nvdec.json >/dev/null
python3 - <<'PY'
import json, numpy as np, sys
sys.path.insert(0, "python")
from edgevision.postprocess import box_iou
a = json.load(open("/tmp/dets_opencv.json")); b = json.load(open("/tmp/dets_nvdec.json"))
print(f"opencv: {len(a)} detections, nvdec: {len(b)} detections")
if a and b:
    ref = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in a])
    ious = [float(box_iou(np.array([d["x1"], d["y1"], d["x2"], d["y2"]]), ref).max()) for d in b]
    print("classes equal:", sorted(d["class_name"] for d in a) == sorted(d["class_name"] for d in b))
    print("best IoU per nvdec box: min %.3f mean %.3f" % (min(ious), sum(ious) / len(ious)))
PY

echo "== tests"
python3 -m pytest -q
echo "== done: $LOG"
