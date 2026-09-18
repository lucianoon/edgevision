"""Export the PyTorch YOLO weights to ONNX (static 1x3x640x640) into models/onnx/."""

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
    args = parser.parse_args()

    model = YOLO(args.weights)
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=False,
            simplify=True,
        )
    )

    target = Path(args.out_dir) / exported.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(target))
    print(f"exported: {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
