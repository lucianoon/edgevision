"""Export TensorRT engines with Ultralytics (runs on the GPU box, inside the container).

Two families per (imgsz, precision):
  * raw head (no NMS)  -> for mAP evaluation with `ultralytics val` (standard conf=0.001 protocol)
  * NMS in the graph   -> production engines for the C++ runtime (conf/iou baked in)

INT8 uses Ultralytics calibration on the val split of --data (500 images disjoint from the mAP set).
Ultralytics prefixes engine files with a JSON metadata header; edgevision's TrtEngine
reads it (names, imgsz), and a .names.json sidecar is still written for older tooling.

    python scripts/export_engines_ultralytics.py --imgsz 640 512 416 --precision fp16 int8
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from ultralytics import YOLO


def export_one(weights: str, imgsz: int, precision: str, nms: bool, data: str, fraction: float,
               out_dir: Path, conf: float, iou: float, workspace: int = 4) -> dict:
    model = YOLO(weights)
    t0 = time.perf_counter()
    exported = Path(
        model.export(
            format="engine",
            imgsz=imgsz,
            half=precision == "fp16",
            int8=precision == "int8",
            data=data if precision == "int8" else None,
            fraction=fraction,
            nms=nms,
            conf=conf,
            iou=iou,
            batch=1,
            dynamic=False,
            simplify=True,
            workspace=workspace,
            verbose=False,
        )
    )
    seconds = time.perf_counter() - t0
    stem = f"{Path(weights).stem}_{'nms_' if nms else 'raw_'}{imgsz}_{precision}"
    target = out_dir / f"{stem}.engine"
    shutil.move(str(exported), str(target))
    names_path = out_dir / f"{stem}.names.json"
    names_path.write_text(json.dumps(model.names), encoding="utf-8")
    info = {"engine": str(target), "imgsz": imgsz, "precision": precision, "nms": nms,
            "export_seconds": round(seconds, 1), "size_mb": round(target.stat().st_size / 1e6, 1)}
    print(json.dumps(info))
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/pytorch/yolo26n.pt")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 512, 416])
    parser.add_argument("--precision", nargs="+", default=["fp16", "int8"], choices=["fp16", "int8", "fp32"])
    parser.add_argument("--families", nargs="+", default=["raw", "nms"], choices=["raw", "nms"])
    parser.add_argument("--data", default="configs/coco_calib.yaml", help="dataset yaml whose val split is used for INT8 calibration")
    parser.add_argument("--fraction", type=float, default=1.0, help="fraction of the dataset used for calibration")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--out-dir", default="models/tensorrt")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace GiB (INT8 + NMS needed more)")
    parser.add_argument("--summary", default="benchmarks/results/sprint7_exports.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for imgsz in args.imgsz:
        for precision in args.precision:
            for family in args.families:
                target = out_dir / f"{Path(args.weights).stem}_{family}_{imgsz}_{precision}.engine"
                if target.exists():
                    print(f"exists: {target}")
                    continue
                results.append(export_one(args.weights, imgsz, precision, family == "nms", args.data, args.fraction,
                                          out_dir, args.conf, args.iou, args.workspace))
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"summary: {args.summary} ({len(results)} exports)")


if __name__ == "__main__":
    main()
