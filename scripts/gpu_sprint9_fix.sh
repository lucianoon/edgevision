#!/usr/bin/env bash
# Sprint 9, final run (inside the container). Root cause of the MOT17 gap: edgevision_trt kept
# its --confidence default (0.5) with --track, so detections below 0.5 never reached ByteTrack
# and its second association was empty. Ultralytics' ByteTrack on the same engine scored
# HOTA 31.8, isolating the runtime. With the fix (--track defaults --confidence to the
# tracker's low threshold, 0.1), rerun the C++ tracker: paper track_thresh 0.5 and 0.25.
set -uo pipefail
cd /workspace/edgevision || exit 1
MOT=benchmarks/results/mot17
ENGINE=models/tensorrt/yolo26n_nms_512_fp16_c10.engine
cmake --build cpp/build --parallel >/tmp/build.log 2>&1 || { tail -30 /tmp/build.log; echo "BUILD FAILED"; exit 1; }
(cd cpp/build && ctest --output-on-failure) | tail -3
# Guard against evaluating a stale binary: the fix changes the usage text.
cpp/build/edgevision_trt --help | grep -q "defaults to 0.1" || { echo "stale binary"; exit 1; }
for TH in 0.5 0.25; do
    TAG=cpp_bytetrack_512_fp16_fixed_th${TH/./}
    rm -rf "${MOT:?}/$TAG" && mkdir -p "$MOT/$TAG/data"
    for S in datasets/MOT17/train/*-FRCNN; do
        N=$(basename "$S"); LEN=$(sed -n 's/^seqLength=//p' "$S/seqinfo.ini")
        cpp/build/edgevision_trt --engine "$ENGINE" --source "$S/$N.mp4" --decoder opencv --track \
            --track-thresh "$TH" --frames "$LEN" --warmup 0 --label "s9_${TAG}_$N" --no-stage-timing \
            --dump-tracks "/tmp/$N.jsonl" --out "$MOT/$TAG/$N.report.json" >/dev/null
        python3 scripts/mot17_eval.py to-mot "/tmp/$N.jsonl" "$MOT/$TAG/data/$N.txt" >/dev/null
    done
    echo "fixed, track_thresh=$TH done"
done
python3 scripts/mot17_eval.py evaluate --gt datasets/MOT17/train --trackers $MOT \
    --out benchmarks/results/mot17_eval.json 2>&1 | grep -E "^(cpp|ultra)"
echo "== fix run done ($(date -u +%H:%M:%S))"
