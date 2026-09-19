"""Export the PyTorch YOLO weights to ONNX (static 1x3x640x640) into models/onnx/.

--nms bakes NMS into the graph (output 1x300x6, thresholds fixed at export time);
--batch N exports a static batch of N (the dynamic-batch NMS export is broken upstream).
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/pytorch/yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out-dir", default="models/onnx")
    parser.add_argument("--nms", action="store_true", help="NMS inside the graph (conf/iou below)")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--batch", type=int, default=1, help="static batch size")
    args = parser.parse_args()

    model = YOLO(args.weights)
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=False,
            simplify=True,
            nms=args.nms,
            conf=args.conf,
            iou=args.iou,
            batch=args.batch,
        )
    )

    suffix = ("_nms" if args.nms else "") + (f"_b{args.batch}" if args.batch > 1 else "")
    target = Path(args.out_dir) / f"{exported.stem}{suffix}{exported.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(target))
    print(f"exported: {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
