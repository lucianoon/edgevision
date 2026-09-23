"""Multi-object tracking accuracy on MOT17 train (HOTA / MOTA / IDF1 with TrackEval).

Three subcommands, run inside the TensorRT container on the GPU box:

    # C++ runtime track dump (edgevision_trt --track --dump-tracks) -> MOTChallenge txt
    python scripts/mot17_eval.py to-mot tracks.jsonl out/MOT17-02-FRCNN.txt

    # reference: Ultralytics' built-in ByteTrack on the same weights and input size
    python scripts/mot17_eval.py reference datasets/MOT17/train/MOT17-02-FRCNN out/02.txt

    # TrackEval over every tracker folder under --trackers (each with data/<seq>.txt)
    python scripts/mot17_eval.py evaluate --trackers benchmarks/results/mot17

Only the person class is written. The detector is a COCO model, not trained on MOT17, and the
evaluation is on the train split because MOT17 test ground truth is private: the numbers are
a sanity check of the tracker, not a leaderboard entry.
"""

import argparse
import json
import sys
from pathlib import Path


def to_mot(dump: Path, out: Path, person: str = "person") -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with dump.open(encoding="utf-8") as src, out.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)
            for t in rec["tracks"]:
                if t["class_name"] != person:
                    continue
                w, h = t["x2"] - t["x1"], t["y2"] - t["y1"]
                # MOTChallenge boxes are 1-based (bb_left, bb_top, width, height).
                dst.write(
                    f"{rec['frame']},{t['id']},{t['x1'] + 1:.2f},{t['y1'] + 1:.2f},{w:.2f},{h:.2f},"
                    f"{t['score']:.4f},-1,-1,-1\n"
                )
                rows += 1
    return rows


def reference(seq: Path, out: Path, weights: str, imgsz: int, conf: float) -> int:
    from ultralytics import YOLO

    model = YOLO(weights)
    frames = sorted((seq / "img1").glob("*.jpg"))
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out.open("w", encoding="utf-8", newline="\n") as dst:
        for i, frame in enumerate(frames, start=1):
            res = model.track(
                str(frame),
                imgsz=imgsz,
                conf=conf,
                classes=[0],
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )[0]
            if res.boxes is None or res.boxes.id is None:
                continue
            for (x1, y1, x2, y2), tid, score in zip(
                res.boxes.xyxy.tolist(),
                res.boxes.id.int().tolist(),
                res.boxes.conf.tolist(),
                strict=True,
            ):
                box = f"{x1 + 1:.2f},{y1 + 1:.2f},{x2 - x1:.2f},{y2 - y1:.2f}"
                dst.write(f"{i},{tid},{box},{score:.4f},-1,-1,-1\n")
                rows += 1
    return rows


def evaluate(gt_root: Path, trackers: Path, out: Path) -> dict:
    import numpy as np

    # TrackEval still uses the numpy aliases removed in numpy 1.24.
    for alias, typ in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, alias):
            setattr(np, alias, typ)
    import trackeval

    seqs = sorted(p.name for p in gt_root.iterdir() if (p / "gt" / "gt.txt").exists())
    seqmap = out.parent / "mot17_seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n", encoding="utf-8")
    names = sorted(p.name for p in trackers.iterdir() if (p / "data").is_dir())

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update(
        {
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
            "USE_PARALLEL": False,
        }
    )
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(
        {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(trackers),
            "TRACKERS_TO_EVAL": names,
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "SEQMAP_FILE": str(seqmap),
            "SKIP_SPLIT_FOL": True,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "eval",
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
            "PRINT_CONFIG": False,
        }
    )
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    raw, _ = trackeval.Evaluator(eval_cfg).evaluate(
        [trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics
    )

    def pick(block: dict) -> dict:
        hota, clear, ident = block["HOTA"], block["CLEAR"], block["Identity"]
        return {
            "HOTA": round(float(np.mean(hota["HOTA"])) * 100, 2),
            "DetA": round(float(np.mean(hota["DetA"])) * 100, 2),
            "AssA": round(float(np.mean(hota["AssA"])) * 100, 2),
            "MOTA": round(float(clear["MOTA"]) * 100, 2),
            "IDF1": round(float(ident["IDF1"]) * 100, 2),
            "IDSW": int(clear["IDSW"]),
            "FP": int(clear["CLR_FP"]),
            "FN": int(clear["CLR_FN"]),
        }

    results = {}
    for name in names:
        per_seq = raw["MotChallenge2DBox"][name]
        results[name] = {
            "combined": pick(per_seq["COMBINED_SEQ"]["pedestrian"]),
            "per_sequence": {s: pick(per_seq[s]["pedestrian"]) for s in seqs},
        }
    report = {
        "benchmark": "MOT17 train (FRCNN copies, 7 sequences), class pedestrian",
        "trackeval": getattr(trackeval, "__version__", "git"),
        "trackers": results,
    }
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("to-mot")
    p.add_argument("dump", type=Path)
    p.add_argument("out", type=Path)
    p = sub.add_parser("reference")
    p.add_argument("seq", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--weights", default="models/pytorch/yolo26n.pt")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.1)
    p = sub.add_parser("evaluate")
    p.add_argument("--gt", type=Path, default=Path("datasets/MOT17/train"))
    p.add_argument("--trackers", type=Path, default=Path("benchmarks/results/mot17"))
    p.add_argument("--out", type=Path, default=Path("benchmarks/results/mot17_eval.json"))
    args = parser.parse_args()

    if args.cmd == "to-mot":
        print(f"{args.out}: {to_mot(args.dump, args.out)} rows")
    elif args.cmd == "reference":
        print(
            f"{args.out}: {reference(args.seq, args.out, args.weights, args.imgsz, args.conf)} rows"
        )
    else:
        report = evaluate(args.gt, args.trackers, args.out)
        for name, r in report["trackers"].items():
            print(name, json.dumps(r["combined"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
