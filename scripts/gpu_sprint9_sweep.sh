#!/usr/bin/env bash
# Sprint 9 follow-up (inside the container): the C++ ByteTrack lost to Ultralytics' ByteTrack
# on MOT17 mostly on detection (DetA), with fewer ID switches. Hypothesis: the paper's
# track_thresh 0.5 opens fewer tracks than Ultralytics' 0.25 new-track threshold. Sweep it.
# Note: the sweep runs on the same MOT17 train split used for evaluation (no held-out set),
# so the best value is reported as a sensitivity analysis, not as a tuned result.
set -uo pipefail
cd /workspace/edgevision || exit 1
MOT=benchmarks/results/mot17
ENGINE=models/tensorrt/yolo26n_nms_512_fp16_c10.engine
for TH in 0.25 0.35 0.45 0.6; do
    TAG=cpp_bytetrack_512_fp16_th${TH/./}
    mkdir -p "$MOT/$TAG/data"
    for S in datasets/MOT17/train/*-FRCNN; do
        N=$(basename "$S"); LEN=$(sed -n 's/^seqLength=//p' "$S/seqinfo.ini")
        cpp/build/edgevision_trt --engine "$ENGINE" --source "$S/$N.mp4" --decoder opencv --track \
            --track-thresh "$TH" --frames "$LEN" --warmup 0 --label "s9_${TAG}_$N" --no-stage-timing \
            --dump-tracks "/tmp/$N.jsonl" --out "/tmp/$N.report.json" >/dev/null
        python3 scripts/mot17_eval.py to-mot "/tmp/$N.jsonl" "$MOT/$TAG/data/$N.txt" >/dev/null
    done
    echo "track_thresh=$TH done"
done
python3 scripts/mot17_eval.py evaluate --gt datasets/MOT17/train --trackers $MOT \
    --out benchmarks/results/mot17_eval.json 2>&1 | grep -E "^(cpp|ultra)"
echo "== sweep done ($(date -u +%H:%M:%S))"
