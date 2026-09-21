"""Summarise a C++ runtime track dump (edgevision_trt --track --dump-tracks file.jsonl):
one JSON object per frame:
{"frame": n, "tracks": [{"id", "x1", "y1", "x2", "y2", "score", "class_name"}]}.

    python scripts/track_stats.py benchmarks/results/tracks_*.jsonl
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def summarise(path: str) -> dict:
    lengths = defaultdict(int)
    frames = 0
    tracks_total = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        frames += 1
        for t in row["tracks"]:
            lengths[t["id"]] += 1
            tracks_total += 1
    unique = len(lengths)
    return {
        "tracker": "edgevision-cpp-bytetrack",
        "dump": path,
        "frames": frames,
        "unique_ids": unique,
        "mean_track_length_frames": round(sum(lengths.values()) / unique, 1) if unique else 0,
        "tracks_per_frame": round(tracks_total / frames, 2) if frames else 0,
        "tracks_longer_than_2s": sum(1 for v in lengths.values() if v >= 50),
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(summarise(p), indent=2))
