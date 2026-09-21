"""Reference tracking statistics with Ultralytics' built-in ByteTrack on a clip, to compare
with the C++ tracker's dump (scripts/track_stats.py). CPU is fine for a few hundred frames.

    python scripts/track_reference.py --source videos/pedestrian_area_1080p25_h264.mp4 --frames 300
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/pytorch/yolo26n.pt")
    parser.add_argument("--source", default="videos/pedestrian_area_1080p25.webm")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.1, help="low threshold: ByteTrack uses weak detections too")
    parser.add_argument("--out", default="benchmarks/results/track_reference.json")
    args = parser.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.source)
    frames = 0
    per_frame_ids = []
    lengths = defaultdict(int)
    dets_per_frame = []
    while frames < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.track(frame, imgsz=args.imgsz, conf=args.conf, tracker="bytetrack.yaml", persist=True,
                              verbose=False, classes=[0])
        boxes = results[0].boxes
        ids = boxes.id.int().tolist() if boxes is not None and boxes.id is not None else []
        per_frame_ids.append(ids)
        dets_per_frame.append(int((boxes.conf >= 0.5).sum()) if boxes is not None else 0)
        for i in ids:
            lengths[i] += 1
        frames += 1

    unique = len(lengths)
    summary = {
        "tracker": "ultralytics-bytetrack",
        "source": args.source, "frames": frames, "imgsz": args.imgsz, "conf": args.conf,
        "unique_ids": unique,
        "mean_track_length_frames": round(sum(lengths.values()) / unique, 1) if unique else 0,
        "tracks_per_frame": round(sum(len(x) for x in per_frame_ids) / frames, 2) if frames else 0,
        "high_conf_dets_per_frame": round(sum(dets_per_frame) / frames, 2) if frames else 0,
        "tracks_longer_than_2s": sum(1 for v in lengths.values() if v >= 50),
    }
    print(json.dumps(summary, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
