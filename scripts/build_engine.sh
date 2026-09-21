#!/usr/bin/env bash
# Build TensorRT engines (FP32 and FP16) from the ONNX export with trtexec and
# record trtexec's own benchmark next to the engine.
#
#   scripts/build_engine.sh [models/onnx/yolo26n.onnx] [models/tensorrt]
set -euo pipefail

ONNX=${1:-models/onnx/yolo26n.onnx}
OUT_DIR=${2:-models/tensorrt}
RESULTS_DIR=benchmarks/results
STEM=$(basename "$ONNX" .onnx)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_' | tr -cd '[:alnum:]_-')
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

mkdir -p "$OUT_DIR" "$RESULTS_DIR"

# Class names sidecar (engines carry no metadata). Read from the ONNX export.
python3 - "$ONNX" "$OUT_DIR/$STEM.names.json" <<'PY'
import ast, json, sys
import onnx
model = onnx.load(sys.argv[1])
names = ast.literal_eval({p.key: p.value for p in model.metadata_props}["names"])
json.dump(names, open(sys.argv[2], "w"))
PY

# Dynamic-batch graphs (exported with dynamic=True; file name contains "dyn") get an
# optimisation profile of 1..MAX_BATCH images; the runtime picks the batch per enqueue.
SHAPES=()
if [[ "$STEM" == *dyn* ]]; then
    MAX_BATCH=${MAX_BATCH:-16}
    OPT_BATCH=${OPT_BATCH:-8}
    SHAPES=(--minShapes=images:1x3x640x640 "--optShapes=images:${OPT_BATCH}x3x640x640" "--maxShapes=images:${MAX_BATCH}x3x640x640")
    echo "dynamic batch profile: min 1, opt $OPT_BATCH, max $MAX_BATCH"
fi

build() {
    local precision=$1; shift
    local engine="$OUT_DIR/${STEM}_${precision}.engine"
    echo "== building $engine ($*)"
    trtexec --onnx="$ONNX" --saveEngine="$engine" "$@" "${SHAPES[@]}" \
        --warmUp=1000 --duration=15 --avgRuns=100 \
        --exportTimes="$RESULTS_DIR/trtexec_${precision}_${GPU}_${STAMP}_times.json" \
        --exportProfile="$RESULTS_DIR/trtexec_${precision}_${GPU}_${STAMP}_profile.json" \
        --separateProfileRun \
        | tee "$RESULTS_DIR/trtexec_${precision}_${GPU}_${STAMP}.log" | grep -E "mean|median|percentile|Throughput|Engine built|Engine deserialized|Latency"
    cp "$OUT_DIR/$STEM.names.json" "$OUT_DIR/${STEM}_${precision}.names.json"
}

build fp32
build fp16 --fp16

ls -la "$OUT_DIR"
