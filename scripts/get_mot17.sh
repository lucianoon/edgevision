#!/usr/bin/env bash
# Download the MOT17 train split (ground truth is public only for train) into datasets/MOT17.
# The zip ships every sequence three times (DPM/FRCNN/SDP public detections) with identical
# frames and ground truth; only the FRCNN copy is kept, since this project runs its own detector.
set -euo pipefail
DIR=${1:-datasets}
mkdir -p "$DIR"
cd "$DIR"
if [ ! -d MOT17/train ] || [ "$(find MOT17/train -maxdepth 1 -name "*-FRCNN" | wc -l)" -lt 7 ]; then
    curl -fSL --retry 3 -o MOT17.zip https://motchallenge.net/data/MOT17.zip
    unzip -q -o MOT17.zip 'MOT17/train/*-FRCNN/*' && rm MOT17.zip
fi
N=$(find MOT17/train -maxdepth 1 -name "*-FRCNN" | wc -l)
[ "$N" -eq 7 ] || { echo "expected 7 MOT17 train sequences, found $N"; exit 1; }
for S in MOT17/train/*-FRCNN; do
    printf '%s frames=%s %sx%s fps=%s\n' "$(basename "$S")" \
        "$(sed -n 's/^seqLength=//p' "$S/seqinfo.ini")" "$(sed -n 's/^imWidth=//p' "$S/seqinfo.ini")" \
        "$(sed -n 's/^imHeight=//p' "$S/seqinfo.ini")" "$(sed -n 's/^frameRate=//p' "$S/seqinfo.ini")"
done
