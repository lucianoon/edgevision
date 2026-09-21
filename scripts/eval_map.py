"""mAP on COCO val2017 for .pt weights and TensorRT engines (Ultralytics val protocol).

    python scripts/eval_map.py models/pytorch/yolo26n.pt --imgsz 640 512 416
    python scripts/eval_map.py models/tensorrt/yolo26n_raw_*_*.engine

Engines carry their imgsz; for .pt each --imgsz is evaluated. Results are appended to a
JSON list so several runs can be merged into one table.
"""

import argparse
import json
import re
from pathlib import Path

from ultralytics import YOLO


def evaluate(model_path: str, imgsz: int | None, data: str, batch: int) -> dict:
    model = YOLO(model_path)
    kwargs = dict(data=data, batch=batch, conf=0.001, iou=0.7, plots=False, verbose=False, device=0, workers=2)
    if imgsz:
        kwargs["imgsz"] = imgsz
    metrics = model.val(**kwargs)
    speed = metrics.speed  # ms per image: preprocess, inference, postprocess (Ultralytics' own loop)
    stem = Path(model_path).stem
    m = re.search(r"_(\d+)_(fp16|int8|fp32)$", stem)
    row = {
        "model": model_path,
        "imgsz": imgsz or (int(m.group(1)) if m else None),
        "precision": m.group(2) if m else "fp32-pytorch",
        "map50_95": round(float(metrics.box.map), 4),
        "map50": round(float(metrics.box.map50), 4),
        "map75": round(float(metrics.box.map75), 4),
        "val_inference_ms": round(float(speed["inference"]), 2),
    }
    print(json.dumps(row))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--imgsz", type=int, nargs="*", default=[], help="for .pt weights; engines use their own")
    parser.add_argument("--data", default="configs/coco_val.yaml")
    parser.add_argument("--batch", type=int, default=1, help="engines are batch 1; .pt can use more")
    parser.add_argument("--out", default="benchmarks/results/sprint7_map.json")
    args = parser.parse_args()

    out = Path(args.out)
    rows = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    for model_path in args.models:
        if model_path.endswith(".pt"):
            for imgsz in (args.imgsz or [640]):
                rows.append(evaluate(model_path, imgsz, args.data, max(args.batch, 16)))
        else:
            rows.append(evaluate(model_path, None, args.data, 1))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved: {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
