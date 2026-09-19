#!/usr/bin/env bash
# Sprint 6 phase A, runs INSIDE the TensorRT container on the GPU box:
#   docker run --rm --gpus all --network host -v /opt/edgevision/repo:/workspace/edgevision edgevision:trt scripts/gpu_sprint6.sh
# How many independent 1080p H.264 streams does one T4 sustain with one NVDEC decoder
# + one TensorRT context per stream? Also: 8 live RTSP cameras (start scripts/rtsp_sim.sh
# on the host first; skipped if the URL is unreachable).
set -euo pipefail
cd /workspace/edgevision
CLIP=videos/pedestrian_area_1080p25_h264.mp4
FRAMES=${FRAMES:-300}
WARMUP=${WARMUP:-30}
ENGINE=models/tensorrt/yolo26n_nms_fp16.engine
RTSP=${RTSP:-rtsp://127.0.0.1:8554/cam}
LOG=benchmarks/results/gpu_sprint6_$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p benchmarks/results
exec > >(tee "$LOG") 2>&1

echo "== environment"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
[ -f "$ENGINE" ] || scripts/build_engine.sh models/onnx/yolo26n_nms.onnx models/tensorrt
[ -f "$CLIP" ] || ffmpeg -hide_banner -loglevel error -y -i videos/pedestrian_area_1080p25.webm -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -an "$CLIP"

echo "== build"
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-75}" >/dev/null
cmake --build cpp/build --parallel
(cd cpp/build && ctest --output-on-failure)

B="cpp/build/edgevision_trt --engine $ENGINE --source $CLIP --decoder nvdec --frames $FRAMES --warmup $WARMUP"
echo "== file source, N independent streams (unpaced: as fast as decode+inference allow)"
for N in 1 2 4 6 8 12; do
  echo "--- streams=$N"
  nvidia-smi --query-gpu=utilization.gpu,utilization.decoder,memory.used --format=csv,noheader -lms 500 > /tmp/util_$N.csv &
  UPID=$!
  $B --streams "$N" --label "h264_file" --no-stage-timing | grep -E "streams=|stream [0-9]|end_to_end|saved"
  kill $UPID 2>/dev/null || true
  python3 - "$N" <<'PY'
import sys, statistics
rows = [l.split(",") for l in open(f"/tmp/util_{sys.argv[1]}.csv") if l.strip()]
gpu = [int(r[0].split()[0]) for r in rows]; dec = [int(r[1].split()[0]) for r in rows]; mem = [int(r[2].split()[0]) for r in rows]
mid = gpu[len(gpu)//4: -len(gpu)//4 or None]; midd = dec[len(dec)//4: -len(dec)//4 or None]
print(f"    nvidia-smi: gpu util mean {statistics.mean(mid):.0f}% max {max(gpu)}%, decoder util mean {statistics.mean(midd):.0f}% max {max(dec)}%, mem max {max(mem)} MiB ({len(rows)} samples)")
PY
done

echo "== stage breakdown at 4 streams (with per-stage syncs)"
$B --streams 4 --label "h264_file_stages" | grep -E "streams=|stream [0-9]|^(decode|preprocess|inference|postprocess|end_to_end)|saved"

echo "== live RTSP cameras (paced at 25 fps by the simulator)"
if ffprobe -v error -rtsp_transport tcp -i "$RTSP" -show_entries stream=codec_name -of csv=p=0 2>/dev/null | grep -q .; then
  for N in 1 4 8; do
    echo "--- rtsp streams=$N"
    nvidia-smi --query-gpu=utilization.gpu,utilization.decoder --format=csv,noheader -lms 500 > /tmp/util_rtsp_$N.csv &
    UPID=$!
    cpp/build/edgevision_trt --engine "$ENGINE" --source "$RTSP" --decoder nvdec --streams "$N" --frames 250 --warmup 25 --label "rtsp_live" --no-stage-timing 2>/dev/null | grep -E "streams=|stream [0-9]|end_to_end|saved"
    kill $UPID 2>/dev/null || true
    python3 - "$N" <<'PY'
import sys, statistics
rows = [l.split(",") for l in open(f"/tmp/util_rtsp_{sys.argv[1]}.csv") if l.strip()]
gpu = [int(r[0].split()[0]) for r in rows]; dec = [int(r[1].split()[0]) for r in rows]
mid = gpu[len(gpu)//4: -len(gpu)//4 or None]; midd = dec[len(dec)//4: -len(dec)//4 or None]
print(f"    nvidia-smi: gpu util mean {statistics.mean(mid):.0f}%, decoder util mean {statistics.mean(midd):.0f}% ({len(rows)} samples)")
PY
  done
else
  echo "RTSP simulator not reachable at $RTSP (start scripts/rtsp_sim.sh on the host); skipping"
fi

echo "== tests"
python3 -m pytest -q
echo "== done: $LOG"
