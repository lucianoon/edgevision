"""Pitfall check for Ultralytics ONNX exports with NMS in the graph and a dynamic batch.

`export(nms=True, dynamic=True)` with the default batch=1 unrolls the NMS loop for ONE
image at trace time, so a batch of N at runtime only gets detections for image 0. The
exporter warns: "'dynamic=True' export requires a maximum batch size, e.g. 'batch=16'".
Passing batch=<max batch> fixes it (any runtime batch <= max works). This script shows
the wrong and the right call side by side. (Originally mistaken for a bug in Sprint 6.)

    python scripts/check_dynamic_nms_export.py
"""

import cv2
import numpy as np
import onnxruntime as ort
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ASSETS


def letterbox(img, size=640):
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (round(w * s), round(h * s)))
    out = np.full((size, size, 3), 114, np.uint8)
    out[: r.shape[0], : r.shape[1]] = r
    x = out[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    return np.ascontiguousarray(x)


def detections_per_image(onnx_path, batch):
    s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = s.run(None, {s.get_inputs()[0].name: batch})[0]  # (N, 300, 6)
    return [int((out[i, :, 4] > 0).sum()) for i in range(out.shape[0])]


print(
    f"ultralytics {ultralytics.__version__} | torch {torch.__version__} "
    f"| onnxruntime {ort.__version__}"
)
bus = letterbox(cv2.imread(str(ASSETS / "bus.jpg")))
zidane = letterbox(cv2.imread(str(ASSETS / "zidane.jpg")))
model = YOLO("models/pytorch/yolo26n.pt")  # exports land next to the weights (git-ignored)

dyn = model.export(
    format="onnx", nms=True, dynamic=True, imgsz=640, conf=0.5, iou=0.45, verbose=False
)
print("dynamic=True  [bus]          ->", detections_per_image(dyn, bus))
print("dynamic=True  [zidane]       ->", detections_per_image(dyn, zidane))
print("dynamic=True  [bus, zidane]  ->", detections_per_image(dyn, np.concatenate([bus, zidane])))
print("dynamic=True  [zidane, bus]  ->", detections_per_image(dyn, np.concatenate([zidane, bus])))

static = model.export(
    format="onnx", nms=True, dynamic=False, batch=2, imgsz=640, conf=0.5, iou=0.45, verbose=False
)
print(
    "static batch=2        [bus, zidane]  ->",
    detections_per_image(static, np.concatenate([bus, zidane])),
)

fixed = model.export(
    format="onnx", nms=True, dynamic=True, batch=4, imgsz=640, conf=0.5, iou=0.45, verbose=False
)
print(
    "dynamic=True batch=4  [bus, zidane]  ->",
    detections_per_image(fixed, np.concatenate([bus, zidane])),
    "(correct: max batch given)",
)
print("dynamic=True batch=4  [zidane]       ->", detections_per_image(fixed, zidane))
