#!/usr/bin/env bash
# Host-side (GPU box): N live RTSP cameras from the simulator -> N NVDEC + TensorRT
# pipelines in the container, with GPU / decoder utilisation sampled by nvidia-smi.
#   scripts/gpu_rtsp_streams.sh [1 4 8 12]
set -uo pipefail
cd /opt/edgevision/repo
ENGINE=models/tensorrt/yolo26n_nms_fp16.engine
RTSP=${RTSP:-rtsp://127.0.0.1:8554/cam}
LOG=benchmarks/results/gpu_rtsp_streams_$(date -u +%Y%m%dT%H%M%SZ).log
COUNTS=${*:-1 4 8 12}

bash scripts/rtsp_sim.sh stop >/dev/null 2>&1
bash scripts/rtsp_sim.sh start | tail -1
sleep 8
echo "simulator containers up: $(docker ps --format '{{.Names}}' | grep -cE 'mediamtx|rtsp-push')/2"

for N in $COUNTS; do
    echo "--- rtsp streams=$N"
    nvidia-smi --query-gpu=utilization.gpu,utilization.decoder --format=csv,noheader -lms 500 > /tmp/util_rtsp_$N.csv &
    UPID=$!
    docker run --rm --gpus all --network host -v /opt/edgevision/repo:/workspace/edgevision edgevision:trt \
        cpp/build/edgevision_trt --engine "$ENGINE" --source "$RTSP" --decoder nvdec --streams "$N" \
        --frames 250 --warmup 25 --label rtsp_live --no-stage-timing 2>/dev/null \
        | grep -E "^streams=|^backend=|stream [0-9]+:|^end_to_end|^saved" | head -20
    kill $UPID 2>/dev/null
    python3 - "$N" <<'PY'
import statistics, sys
rows = [l.split(",") for l in open(f"/tmp/util_rtsp_{sys.argv[1]}.csv") if l.strip()]
if rows:
    g = [int(r[0].split()[0]) for r in rows]; d = [int(r[1].split()[0]) for r in rows]
    m = g[len(g)//4:-len(g)//4 or None] or g; md = d[len(d)//4:-len(d)//4 or None] or d
    print(f"    nvidia-smi: gpu util mean {statistics.mean(m):.0f}% max {max(g)}%, decoder util mean {statistics.mean(md):.0f}% ({len(rows)} samples)")
PY
done 2>&1 | tee "$LOG"

bash scripts/rtsp_sim.sh stop >/dev/null
echo "== done: $LOG"
