#!/usr/bin/env bash
# Sprint 9, runs INSIDE the TensorRT container on the GPU box:
#   docker run --rm --gpus all --ipc=host -v /opt/edgevision/repo:/workspace/edgevision edgevision:trt scripts/gpu_sprint9.sh
# Closes two evidence gaps of the README:
#   1. mAP of the engines that actually ship (FP16, NMS in the graph, 512) next to the FP32 reference.
#   2. Tracking accuracy on MOT17 train (HOTA/MOTA/IDF1): C++ ByteTrack vs Ultralytics' ByteTrack.
# Every step logs and never stops the others.
set -uo pipefail
cd /workspace/edgevision || exit 1
OUT=benchmarks/results
MOT=$OUT/mot17
step() { echo; echo "== $* ($(date -u +%H:%M:%S))"; }

step "environment"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python3 -c "import tensorrt, ultralytics; print('tensorrt', tensorrt.__version__, 'ultralytics', ultralytics.__version__)"
[ -f models/pytorch/yolo26n.pt ] || curl -fsSL -o models/pytorch/yolo26n.pt \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
sha256sum models/pytorch/yolo26n.pt

step "datasets: COCO val2017 (mAP) and MOT17 train (tracking)"
bash scripts/get_coco_val.sh datasets 2>&1 | tail -2
bash scripts/get_mot17.sh datasets 2>&1 | tail -8

step "engines (x86 T4): FP16 raw 640/512 for the mAP protocol, FP16 NMS 512 at conf 0.5 and 0.1"
python3 scripts/export_engines_ultralytics.py --imgsz 640 512 --precision fp16 --families raw nms \
    --summary $OUT/sprint9_exports.json 2>&1 | grep -E "^\{|exists|rror" | cut -c1-200
[ -f models/tensorrt/yolo26n_nms_512_fp16_c10.engine ] || python3 scripts/export_engines_ultralytics.py \
    --imgsz 512 --precision fp16 --families nms --conf 0.1 --summary $OUT/sprint9_exports_c10.json 2>&1 | grep -E "^\{|rror" | cut -c1-200
ls -la models/tensorrt/*fp16*.engine

step "mAP on the 4500-image val2017 split: shipped FP16 engines vs the FP32 reference"
MAP=$OUT/sprint9_map.json
rm -f $MAP
for E in models/tensorrt/yolo26n_raw_640_fp16.engine models/tensorrt/yolo26n_raw_512_fp16.engine \
         models/tensorrt/yolo26n_nms_512_fp16.engine models/tensorrt/yolo26n_nms_512_fp16_c10.engine; do
    [ -f "$E" ] || { echo "missing $E"; continue; }
    python3 scripts/eval_map.py "$E" --out $MAP 2>&1 | grep -E "^\{|rror" | cut -c1-240
done

step "build C++ runtime"
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-75}" >/dev/null
cmake --build cpp/build --parallel 2>&1 | grep -E "error|FAILED" | head -20
(cd cpp/build && ctest --output-on-failure) | tail -4

step "MOT17: sequences to near-lossless H.264 (the runtime reads video)"
for S in datasets/MOT17/train/*-FRCNN; do
    N=$(basename "$S"); FPS=$(sed -n 's/^frameRate=//p' "$S/seqinfo.ini")
    [ -f "$S/$N.mp4" ] || ffmpeg -hide_banner -loglevel error -y -framerate "$FPS" -i "$S/img1/%06d.jpg" \
        -c:v libx264 -preset fast -crf 10 -pix_fmt yuv420p "$S/$N.mp4"
done

step "MOT17: C++ ByteTrack (engine 512 FP16 NMS conf 0.1, NVDEC off: opencv host decode)"
rm -rf $MOT && mkdir -p $MOT/cpp_bytetrack_512_fp16/data $MOT/ultralytics_bytetrack_512_fp32/data
for S in datasets/MOT17/train/*-FRCNN; do
    N=$(basename "$S"); LEN=$(sed -n 's/^seqLength=//p' "$S/seqinfo.ini")
    cpp/build/edgevision_trt --engine models/tensorrt/yolo26n_nms_512_fp16_c10.engine --source "$S/$N.mp4" \
        --decoder opencv --track --frames "$LEN" --warmup 0 --label "s9_$N" --no-stage-timing \
        --dump-tracks "$MOT/$N.jsonl" --out "$MOT/$N.report.json" | grep -E "^tracking|^end_to_end" | tr '\n' ' '
    python3 scripts/mot17_eval.py to-mot "$MOT/$N.jsonl" "$MOT/cpp_bytetrack_512_fp16/data/$N.txt"
done

step "MOT17: reference, Ultralytics ByteTrack on yolo26n.pt (FP32, 512, conf 0.1)"
for S in datasets/MOT17/train/*-FRCNN; do
    N=$(basename "$S")
    python3 scripts/mot17_eval.py reference "$S" "$MOT/ultralytics_bytetrack_512_fp32/data/$N.txt" 2>&1 | tail -1
done

step "MOT17: TrackEval (HOTA, CLEAR, Identity)"
python3 -m pip install -q "git+https://github.com/JonathonLuiten/TrackEval.git" 2>&1 | tail -1
python3 -c "import trackeval, os; print('trackeval at', os.path.dirname(trackeval.__file__))"
python3 scripts/mot17_eval.py evaluate --gt datasets/MOT17/train --trackers $MOT --out $OUT/mot17_eval.json

step "tests"
python3 -m pytest -q 2>&1 | tail -2
echo "== done ($(date -u +%H:%M:%S))"
