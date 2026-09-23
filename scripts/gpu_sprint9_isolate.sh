#!/usr/bin/env bash
# Sprint 9, third follow-up (inside the container): neither track_thresh nor the NMS IoU
# explains the gap to Ultralytics on MOT17. Isolate detector vs tracker by running
# Ultralytics' ByteTrack on the very engine the C++ runtime uses (FP16, NMS in graph, conf 0.1).
set -uo pipefail
cd /workspace/edgevision || exit 1
MOT=benchmarks/results/mot17
TAG=ultralytics_bytetrack_on_trt_engine_512_fp16
mkdir -p "$MOT/$TAG/data"
for S in datasets/MOT17/train/*-FRCNN; do
    N=$(basename "$S")
    python3 scripts/mot17_eval.py reference "$S" "$MOT/$TAG/data/$N.txt" \
        --weights models/tensorrt/yolo26n_nms_512_fp16_c10.engine 2>&1 | tail -1
done
python3 scripts/mot17_eval.py evaluate --gt datasets/MOT17/train --trackers $MOT \
    --out benchmarks/results/mot17_eval.json 2>&1 | grep -E "^(cpp|ultra)"
echo "== isolate done ($(date -u +%H:%M:%S))"
