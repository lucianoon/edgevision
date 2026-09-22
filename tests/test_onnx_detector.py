import cv2
import numpy as np
import pytest
from ultralytics.utils import ASSETS

from edgevision.detector import YoloDetector
from edgevision.metrics import PerformanceMetrics
from edgevision.onnx_detector import OnnxDetector
from edgevision.postprocess import box_iou

ONNX_PATH = "models/onnx/yolo26n.onnx"
PT_PATH = "models/pytorch/yolo26n.pt"


@pytest.fixture(scope="module")
def frame():
    return cv2.imread(str(ASSETS / "bus.jpg"))


@pytest.fixture(scope="module")
def onnx_detections(frame):
    detector = OnnxDetector(ONNX_PATH, confidence=0.5, iou_threshold=0.45)
    return detector.detect(frame)


def test_onnx_detector_finds_people_and_bus(onnx_detections, frame):
    names = {d.class_name for d in onnx_detections}
    assert "person" in names and "bus" in names

    h, w = frame.shape[:2]
    for d in onnx_detections:
        assert 0.5 <= d.confidence <= 1.0
        assert 0 <= d.x1 < d.x2 <= w
        assert 0 <= d.y1 < d.y2 <= h


def test_onnx_matches_pytorch_reference(onnx_detections, frame):
    reference = YoloDetector(PT_PATH, confidence=0.5).detect(frame)

    assert sorted(d.class_name for d in onnx_detections) == sorted(d.class_name for d in reference)

    ref_boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in reference])
    for d in onnx_detections:
        ious = box_iou(np.array([d.x1, d.y1, d.x2, d.y2]), ref_boxes)
        best = int(ious.argmax())
        assert ious[best] > 0.85, f"{d.class_name}: best IoU {ious[best]:.2f}"
        assert reference[best].class_name == d.class_name
        # Ultralytics letterboxes .pt models to a rectangle (auto=True) while we pad to a
        # square, so confidences differ by up to ~0.13 on some platforms (Linux CI).
        assert abs(reference[best].confidence - d.confidence) < 0.15


def test_onnx_detector_records_stage_metrics(frame):
    metrics = PerformanceMetrics()
    detector = OnnxDetector(ONNX_PATH, metrics=metrics)

    detector.detect(frame)

    assert set(metrics.summary()) == {"preprocess", "inference", "postprocess"}


def test_onnx_detector_takes_input_size_from_static_graph():
    # yolo26n.onnx was exported at 640; a conflicting image_size must not reach preprocess
    # (it would feed a 512 tensor to a 640 graph). The graph is the source of truth.
    assert OnnxDetector(ONNX_PATH, image_size=512).image_size == 640
    assert OnnxDetector(ONNX_PATH).image_size == 640


def test_providers_can_be_forced_from_the_environment(monkeypatch):
    from edgevision.onnx_detector import providers_from_env

    monkeypatch.delenv("EDGEVISION_ORT_PROVIDERS", raising=False)
    assert providers_from_env() is None
    monkeypatch.setenv("EDGEVISION_ORT_PROVIDERS", " CPUExecutionProvider , ")
    assert providers_from_env() == ["CPUExecutionProvider"]
    assert OnnxDetector(ONNX_PATH).providers == ["CPUExecutionProvider"]
