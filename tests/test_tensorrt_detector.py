"""Runs only where TensorRT + a CUDA GPU exist (the GPU dev box / Jetson)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

tensorrt = pytest.importorskip("tensorrt")
pytest.importorskip("cuda")

from ultralytics.utils import ASSETS

from edgevision.onnx_detector import OnnxDetector
from edgevision.postprocess import box_iou
from edgevision.tensorrt_detector import TensorRTDetector

ONNX_PATH = "models/onnx/yolo26n.onnx"
ENGINES = [
    p
    for p in ("models/tensorrt/yolo26n_fp32.engine", "models/tensorrt/yolo26n_fp16.engine")
    if Path(p).exists()
]

pytestmark = pytest.mark.skipif(
    not ENGINES, reason="no TensorRT engine built (scripts/build_engine.sh)"
)


@pytest.fixture(scope="module")
def frame():
    return cv2.imread(str(ASSETS / "bus.jpg"))


@pytest.fixture(scope="module")
def reference(frame):
    return OnnxDetector(ONNX_PATH, confidence=0.5, iou_threshold=0.45).detect(frame)


@pytest.mark.parametrize("engine_path", ENGINES)
def test_engine_matches_onnx_reference(engine_path, frame, reference):
    detector = TensorRTDetector(engine_path, confidence=0.5, iou_threshold=0.45)
    assert detector.input_shape == (1, 3, 640, 640)

    detections = detector.detect(frame)
    detector.close()

    assert sorted(d.class_name for d in detections) == sorted(d.class_name for d in reference)

    ref_boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in reference])
    for d in detections:
        ious = box_iou(np.array([d.x1, d.y1, d.x2, d.y2]), ref_boxes)
        best = int(ious.argmax())
        assert ious[best] > 0.9, f"{d.class_name}: best IoU {ious[best]:.2f}"
        assert abs(reference[best].confidence - d.confidence) < 0.05
