"""Parity of the C++ TensorRT runtime (cpp/build/edgevision_trt) with the Python ONNX path.
Runs only on the GPU box after `scripts/gpu_sprint4.sh` built the binary and the engine."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from ultralytics.utils import ASSETS

from edgevision.onnx_detector import OnnxDetector
from edgevision.postprocess import box_iou

BINARY = Path("cpp/build/edgevision_trt")
ENGINE = Path("models/tensorrt/yolo26n_nms_fp16.engine")
NMS_ONNX = "models/onnx/yolo26n_nms.onnx"

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not BINARY.exists() or not ENGINE.exists(),
    reason="C++ runtime binary or NMS engine not built",
)


def test_cpp_runtime_matches_onnx_reference(tmp_path):
    dump = tmp_path / "dets.json"
    report = tmp_path / "report.json"
    subprocess.run(
        [
            str(BINARY),
            "--engine",
            str(ENGINE),
            "--source",
            str(ASSETS / "bus.jpg"),
            "--frames",
            "3",
            "--warmup",
            "1",
            "--dump-detections",
            str(dump),
            "--out",
            str(report),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cpp = json.loads(dump.read_text())
    assert json.loads(report.read_text())["backend"] == "cpp_tensorrt"

    frame = cv2.imread(str(ASSETS / "bus.jpg"))
    reference = OnnxDetector(NMS_ONNX, confidence=0.5, iou_threshold=0.45).detect(frame)

    assert sorted(d["class_name"] for d in cpp) == sorted(d.class_name for d in reference)
    ref_boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in reference])
    for d in cpp:
        ious = box_iou(np.array([d["x1"], d["y1"], d["x2"], d["y2"]]), ref_boxes)
        best = int(ious.argmax())
        assert ious[best] > 0.9, f"{d['class_name']}: IoU {ious[best]:.3f}"
        assert abs(reference[best].confidence - d["confidence"]) < 0.05
