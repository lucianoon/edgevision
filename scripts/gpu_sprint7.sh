#!/usr/bin/env bash
# Sprint 7, runs INSIDE the TensorRT container on the GPU box:
#   docker run --rm --gpus all -v /opt/edgevision/repo:/workspace/edgevision edgevision:trt scripts/gpu_sprint7.sh
# Cheaper model, quality measured: INT8 and smaller inputs. mAP on COCO val2017 (Ultralytics
# val, raw-head engines) + speed in the C++ runtime (NMS engines, 1 and 12 NVDEC streams).
set -euo pipefail
cd /workspace/edgevision
CLIP=videos/pedestrian_area_1080p25_h264.mp4
SIZES=${SIZES:-640 512 416}
FRAMES=${FRAMES:-300}
WARMUP=${WARMUP:-30}
LOG=benchmarks/results/gpu_sprint7_$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p benchmarks/results
exec > >(tee "$LOG") 2>&1

echo "== environment"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python3 -c "import tensorrt, ultralytics, torch; print('tensorrt', tensorrt.__version__, '| ultralytics', ultralytics.__version__, '| torch', torch.__version__, torch.cuda.is_available())"
[ -f "$CLIP" ] || ffmpeg -hide_banner -loglevel error -y -i videos/pedestrian_area_1080p25.webm -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -an "$CLIP"

echo "== COCO val2017"
bash scripts/get_coco_val.sh datasets

echo "== build C++ runtime"
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-75}" >/dev/null
cmake --build cpp/build --parallel
(cd cpp/build && ctest --output-on-failure)

echo "== engines (Ultralytics export; INT8 calibrated on 500 val2017 images disjoint from the 4500 used for mAP)"
python3 scripts/export_engines_ultralytics.py --imgsz $SIZES --precision fp16 --families nms
python3 scripts/export_engines_ultralytics.py --imgsz $SIZES --precision int8 --families raw nms
ls -la models/tensorrt/yolo26n_{raw,nms}_*.engine

echo "== mAP: PyTorch FP32 reference at each size, then INT8 raw engines"
rm -f benchmarks/results/sprint7_map.json
python3 scripts/eval_map.py models/pytorch/yolo26n.pt --imgsz $SIZES --batch 16
python3 scripts/eval_map.py models/tensorrt/yolo26n_raw_*_int8.engine

echo "== speed: C++ runtime, NVDEC, production (NMS) engines"
for S in $SIZES; do for P in fp16 int8; do
  E=models/tensorrt/yolo26n_nms_${S}_${P}.engine
  [ -f "$E" ] || { echo "missing $E"; continue; }
  echo "--- $E, 1 stream"
  cpp/build/edgevision_trt --engine "$E" --source "$CLIP" --decoder nvdec --frames "$FRAMES" --warmup "$WARMUP" --label "s7_${S}_${P}" --no-stage-timing | grep -E "^backend=|^end_to_end|^decode|saved"
  echo "--- $E, 12 streams"
  cpp/build/edgevision_trt --engine "$E" --source "$CLIP" --decoder nvdec --streams 12 --frames "$FRAMES" --warmup "$WARMUP" --label "s7_${S}_${P}" --no-stage-timing | grep -E "^streams=|^end_to_end|saved"
done; done

echo "== detections sanity: nms int8 640 vs fp16 640 on the first frame"
cpp/build/edgevision_trt --engine models/tensorrt/yolo26n_nms_640_fp16.engine --source "$CLIP" --decoder nvdec --frames 1 --warmup 0 --out /tmp/r1.json --dump-detections /tmp/d_fp16.json >/dev/null
cpp/build/edgevision_trt --engine models/tensorrt/yolo26n_nms_640_int8.engine --source "$CLIP" --decoder nvdec --frames 1 --warmup 0 --out /tmp/r2.json --dump-detections /tmp/d_int8.json >/dev/null
python3 - <<'PY'
import json, numpy as np, sys
sys.path.insert(0, "python")
from edgevision.postprocess import box_iou
a = json.load(open("/tmp/d_fp16.json")); b = json.load(open("/tmp/d_int8.json"))
print(f"fp16: {len(a)} detections, int8: {len(b)} detections")
if a and b:
    ref = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in a])
    ious = [float(box_iou(np.array([d["x1"], d["y1"], d["x2"], d["y2"]]), ref).max()) for d in b]
    print("min IoU %.3f mean %.3f" % (min(ious), sum(ious) / len(ious)))
PY

echo "== summary"
python3 - <<'PY'
import json, glob, re
maps = json.load(open("benchmarks/results/sprint7_map.json"))
speed = {}
for f in glob.glob("benchmarks/results/cpp_tensorrt_nvdec*_s7_*.json"):
    r = json.load(open(f)); m = re.search(r"s7_(\d+)_(fp16|int8)", f)
    key = (int(m.group(1)), m.group(2), r.get("streams", 1))
    speed[key] = r.get("aggregate_fps") or r["fps"]
print(f"{'imgsz':>6} {'precision':>13} {'mAP50-95':>9} {'mAP50':>7} {'1-stream fps':>13} {'12-stream agg fps':>18}")
for row in maps:
    s, p = row["imgsz"], row["precision"]
    pp = "fp16" if p == "fp32-pytorch" else p
    f1 = speed.get((s, pp, 1)); f12 = speed.get((s, pp, 12))
    print(f"{s:>6} {p:>13} {row['map50_95']:>9.3f} {row['map50']:>7.3f} {f1 if f1 is None else round(f1):>13} {f12 if f12 is None else round(f12):>18}")
PY

echo "== tests"
python3 -m pytest -q
echo "== done: $LOG"
