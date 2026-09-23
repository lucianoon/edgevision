#!/usr/bin/env bash
# Sprint 9, second follow-up (inside the container). The track_thresh sweep barely moved
# DetA, so the gap to Ultralytics is on the detector side: the shipped NMS-in-graph engine
# suppresses at IoU 0.45, while Ultralytics' model.track() defaults to IoU 0.7, which keeps
# overlapping pedestrians in crowded MOT17 scenes. Export the tracking engine at IoU 0.7
# and rerun the C++ tracker with the paper's track_thresh (0.5) and with 0.25.
set -uo pipefail
cd /workspace/edgevision || exit 1
MOT=benchmarks/results/mot17
python3 scripts/export_engines_ultralytics.py --imgsz 512 --precision fp16 --families nms --conf 0.1 --iou 0.7 \
    --out-dir models/tensorrt/iou07 --summary benchmarks/results/sprint9_exports_iou07.json 2>&1 | grep -E "^\{|rror" | cut -c1-200
ENGINE=$(ls models/tensorrt/iou07/yolo26n_nms_512_fp16_c10.engine)
for TH in 0.5 0.25; do
    TAG=cpp_bytetrack_512_fp16_iou07_th${TH/./}
    mkdir -p "$MOT/$TAG/data"
    for S in datasets/MOT17/train/*-FRCNN; do
        N=$(basename "$S"); LEN=$(sed -n 's/^seqLength=//p' "$S/seqinfo.ini")
        cpp/build/edgevision_trt --engine "$ENGINE" --source "$S/$N.mp4" --decoder opencv --track \
            --track-thresh "$TH" --frames "$LEN" --warmup 0 --label "s9_${TAG}_$N" --no-stage-timing \
            --dump-tracks "/tmp/$N.jsonl" --out "/tmp/$N.report.json" >/dev/null
        python3 scripts/mot17_eval.py to-mot "/tmp/$N.jsonl" "$MOT/$TAG/data/$N.txt" >/dev/null
    done
    echo "iou 0.7, track_thresh=$TH done"
done
python3 scripts/mot17_eval.py evaluate --gt datasets/MOT17/train --trackers $MOT \
    --out benchmarks/results/mot17_eval.json 2>&1 | grep -E "^(cpp|ultra)"
echo "== iou follow-up done ($(date -u +%H:%M:%S))"
