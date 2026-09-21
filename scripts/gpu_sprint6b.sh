#!/usr/bin/env bash
# Sprint 6 phase B, runs INSIDE the TensorRT container on the GPU box:
#   docker run --rm --gpus all -v /opt/edgevision/repo:/workspace/edgevision edgevision:trt scripts/gpu_sprint6b.sh
# Batched inference (one batch-N engine per round) vs N independent contexts, file source, unpaced.
set -euo pipefail
cd /workspace/edgevision
CLIP=videos/pedestrian_area_1080p25_h264.mp4
FRAMES=${FRAMES:-300}
WARMUP=${WARMUP:-30}
BATCHES=${BATCHES:-4 8 12}
LOG=benchmarks/results/gpu_sprint6b_$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p benchmarks/results
exec > >(tee "$LOG") 2>&1

echo "== environment"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
[ -f models/tensorrt/yolo26n_nms_fp16.engine ] || scripts/build_engine.sh models/onnx/yolo26n_nms.onnx models/tensorrt
[ -f "$CLIP" ] || ffmpeg -hide_banner -loglevel error -y -i videos/pedestrian_area_1080p25.webm -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -an "$CLIP"

echo "== static-batch engines (FP16 only)"
for N in $BATCHES; do
  E=models/tensorrt/yolo26n_nms_b${N}_fp16.engine
  if [ ! -f "$E" ]; then
    echo "-- building batch $N"
    trtexec --onnx=models/onnx/yolo26n_nms_b${N}.onnx --saveEngine="$E" --fp16 --warmUp=500 --duration=8 --avgRuns=50 \
      > "benchmarks/results/trtexec_b${N}_fp16_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
    grep -E "Throughput|GPU Compute Time:|Engine built" benchmarks/results/trtexec_b${N}_fp16_*.log | sed 's/^\[[^]]*\] \[I\] //' | cut -c1-120
    cp models/tensorrt/yolo26n_nms.names.json models/tensorrt/yolo26n_nms_b${N}_fp16.names.json 2>/dev/null || cp models/tensorrt/yolo26n_nms_fp16.names.json models/tensorrt/yolo26n_nms_b${N}_fp16.names.json
  fi
done
ls -la models/tensorrt/*.engine

echo "== build"
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-75}" >/dev/null
cmake --build cpp/build --parallel
(cd cpp/build && ctest --output-on-failure)

sample_util() {  # sample_util <tag> <command...>
  local tag=$1; shift
  nvidia-smi --query-gpu=utilization.gpu,utilization.decoder,memory.used --format=csv,noheader -lms 500 > /tmp/util_$tag.csv &
  local upid=$!
  "$@"
  kill $upid 2>/dev/null || true
  python3 - "$tag" <<'PY'
import statistics, sys
rows = [l.split(",") for l in open(f"/tmp/util_{sys.argv[1]}.csv") if l.strip()]
if rows:
    g = [int(r[0].split()[0]) for r in rows]; d = [int(r[1].split()[0]) for r in rows]; m = [int(r[2].split()[0]) for r in rows]
    mid = g[len(g)//4:-len(g)//4 or None] or g; midd = d[len(d)//4:-len(d)//4 or None] or d
    print(f"    nvidia-smi: gpu util mean {statistics.mean(mid):.0f}% max {max(g)}%, decoder util mean {statistics.mean(midd):.0f}%, mem max {max(m)} MiB ({len(rows)} samples)")
PY
}

echo "== batched vs independent, file source, unpaced"
for N in $BATCHES; do
  E=models/tensorrt/yolo26n_nms_b${N}_fp16.engine
  echo "--- batched N=$N"
  sample_util "b$N" cpp/build/edgevision_trt --engine "$E" --source "$CLIP" --decoder nvdec --streams "$N" --batched \
      --frames "$FRAMES" --warmup "$WARMUP" --label h264_file 2>&1 | grep -E "^batched|^(gather|inference|postprocess|end_to_end)|worker means|saved"
  echo "--- independent N=$N (phase A design, batch-1 engine)"
  sample_util "i$N" cpp/build/edgevision_trt --engine models/tensorrt/yolo26n_nms_fp16.engine --source "$CLIP" --decoder nvdec \
      --streams "$N" --frames "$FRAMES" --warmup "$WARMUP" --label h264_file --no-stage-timing 2>&1 | grep -E "^streams=|^end_to_end|saved"
done

echo "== detections: batched N=4 (stream 0) vs single stream on the same clip"
cpp/build/edgevision_trt --engine models/tensorrt/yolo26n_nms_b4_fp16.engine --source "$CLIP" --decoder nvdec --streams 4 --batched --frames 1 --warmup 0 --out /tmp/rb.json --dump-detections /tmp/dets_batched.json >/dev/null
cpp/build/edgevision_trt --engine models/tensorrt/yolo26n_nms_fp16.engine --source "$CLIP" --decoder nvdec --frames 1 --warmup 0 --out /tmp/rs.json --dump-detections /tmp/dets_single.json >/dev/null
python3 - <<'PY'
import json, numpy as np, sys
sys.path.insert(0, "python")
from edgevision.postprocess import box_iou
a = json.load(open("/tmp/dets_single.json")); b = json.load(open("/tmp/dets_batched.json"))
print(f"single: {len(a)} detections, batched: {len(b)} detections")
if a and b:
    ref = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in a])
    ious = [float(box_iou(np.array([d["x1"], d["y1"], d["x2"], d["y2"]]), ref).max()) for d in b]
    print("classes equal:", sorted(d["class_name"] for d in a) == sorted(d["class_name"] for d in b), "| min IoU %.3f" % min(ious))
PY

echo "== tests"
python3 -m pytest -q
echo "== done: $LOG"
