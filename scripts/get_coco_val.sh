#!/usr/bin/env bash
# Download COCO val2017 (images, ~780 MB) + Ultralytics-format labels into datasets/coco and
# split the 5000-image list into a 500-image INT8 calibration list and a 4500-image mAP list.
# Enough for evaluation and calibration; the 19 GB train set is not needed.
set -euo pipefail
DIR=${1:-datasets}
mkdir -p "$DIR"
cd "$DIR"
if [ ! -f coco/val2017.txt ]; then
    curl -fsSL -o coco2017labels.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
    unzip -q -o coco2017labels.zip && rm coco2017labels.zip        # -> coco/{labels/val2017, val2017.txt, ...}
fi
N=$(ls coco/images/val2017 2>/dev/null | wc -l)
if [ "$N" -lt 5000 ]; then
    echo "downloading val2017 images ($N present)"
    curl -fSL --retry 3 -o val2017.zip http://images.cocodataset.org/zips/val2017.zip
    mkdir -p coco/images
    unzip -q -o val2017.zip -d coco/images && rm val2017.zip      # -> coco/images/val2017/*.jpg
fi
N=$(ls coco/images/val2017 | wc -l)
[ "$N" -ge 5000 ] || { echo "val2017 images missing ($N)"; exit 1; }
# Deterministic disjoint split (the list is already sorted by file name).
head -n 500 coco/val2017.txt > coco/val2017_calib.txt
tail -n +501 coco/val2017.txt > coco/val2017_eval.txt
echo "images: $N, labels: $(ls coco/labels/val2017 | wc -l), calib: $(wc -l < coco/val2017_calib.txt), eval: $(wc -l < coco/val2017_eval.txt)"
