#!/usr/bin/env bash
# Sprint 7 plan B (inside the container). Each step logs to benchmarks/results/ and never
# stops the others: INT8 raw engines for mAP, one INT8+NMS attempt with more workspace,
# mAP table, FP16 speed. Expected ~45 min on a T4.
set -uo pipefail
cd /workspace/edgevision || exit 1
CLIP=videos/pedestrian_area_1080p25_h264.mp4
OUT=benchmarks/results
step() { echo; echo "== $* ($(date -u +%H:%M:%S))"; }

if [ -z "${ONLY_EVAL:-}" ]; then
step "engines: INT8 raw 512/416 (640 exists)"
python3 scripts/export_engines_ultralytics.py --imgsz 640 512 416 --precision int8 --families raw --summary $OUT/sprint7_exports_raw_int8.json 2>&1 | grep -E "^\{|exists|Error|error" | cut -c1-200

step "engine: INT8 + NMS 640 with workspace 8 GiB (single attempt)"
python3 scripts/export_engines_ultralytics.py --imgsz 640 --precision int8 --families nms --workspace 8 --summary $OUT/sprint7_exports_nms_int8.json 2>&1 | grep -E "^\{|exists|Error|error|Could not" | cut -c1-200
ls -la models/tensorrt/*int8*.engine 2>/dev/null
fi

step "mAP on 4500 val2017 images: PyTorch FP32 at 640/512/416, INT8 raw engines"
rm -f $OUT/sprint7_map.json
python3 scripts/eval_map.py models/pytorch/yolo26n.pt --imgsz 640 512 416 --batch 16 2>&1 | grep -E "^\{|Error|error" | cut -c1-200
for E in models/tensorrt/yolo26n_raw_*_int8.engine; do
  python3 scripts/eval_map.py "$E" 2>&1 | grep -E "^\{|Error|error" | cut -c1-200
done

step "speed: C++ runtime, NVDEC, NMS engines (1 and 12 streams)"
for E in models/tensorrt/yolo26n_nms_640_fp16.engine models/tensorrt/yolo26n_nms_512_fp16.engine models/tensorrt/yolo26n_nms_416_fp16.engine models/tensorrt/yolo26n_nms_640_int8.engine; do
  [ -f "$E" ] || { echo "missing $E"; continue; }
  TAG=$(basename "$E" .engine | sed 's/yolo26n_nms_/s7_/')
  echo "--- $E"
  cpp/build/edgevision_trt --engine "$E" --source "$CLIP" --decoder nvdec --frames 300 --warmup 30 --label "$TAG" --no-stage-timing | grep -E "^backend=|^end_to_end|^decode|saved"
  cpp/build/edgevision_trt --engine "$E" --source "$CLIP" --decoder nvdec --streams 12 --frames 300 --warmup 30 --label "$TAG" --no-stage-timing | grep -E "^streams=|^end_to_end|saved"
done

step "summary"
python3 - <<'PY'
import json, glob, re
maps = json.load(open("benchmarks/results/sprint7_map.json"))
speed = {}
for f in glob.glob("benchmarks/results/cpp_tensorrt_nvdec*_s7_*.json"):
    r = json.load(open(f)); m = re.search(r"s7_(\d+)_(fp16|int8)", f)
    speed[(int(m.group(1)), m.group(2), r.get("streams", 1))] = r.get("aggregate_fps") or r["fps"]
print(f"{'imgsz':>6} {'precision':>13} {'mAP50-95':>9} {'mAP50':>7} {'val ms':>7} {'1-stream fps':>13} {'12-stream agg':>14}")
for row in maps:
    s, p = row["imgsz"], row["precision"]; pp = "fp16" if p == "fp32-pytorch" else p
    f1 = speed.get((s, pp, 1)); f12 = speed.get((s, pp, 12))
    print(f"{s:>6} {p:>13} {row['map50_95']:>9.3f} {row['map50']:>7.3f} {row['val_inference_ms']:>7.2f} {'-' if f1 is None else round(f1):>13} {'-' if f12 is None else round(f12):>14}")
PY
echo "== done ($(date -u +%H:%M:%S))"
