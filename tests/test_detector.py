import cv2
import pytest
from ultralytics.utils import ASSETS

from edgevision.detector import Detection, YoloDetector

MODEL_PATH = "models/pytorch/yolo26n.pt"


@pytest.fixture(scope="module")
def detector():
    return YoloDetector(model_path=MODEL_PATH, confidence=0.5, image_size=640)


def test_detects_people_and_bus_in_sample_image(detector):
    frame = cv2.imread(str(ASSETS / "bus.jpg"))

    detections = detector.detect(frame)

    assert detections, "expected at least one detection"
    assert all(isinstance(d, Detection) for d in detections)

    names = {d.class_name for d in detections}
    assert "person" in names
    assert "bus" in names

    h, w = frame.shape[:2]
    for d in detections:
        assert 0.5 <= d.confidence <= 1.0
        assert 0 <= d.x1 < d.x2 <= w
        assert 0 <= d.y1 < d.y2 <= h


def test_device_can_be_forced_from_the_environment(monkeypatch):
    monkeypatch.setenv("EDGEVISION_TORCH_DEVICE", "cpu")
    assert YoloDetector(model_path=MODEL_PATH).device == "cpu"
    assert YoloDetector(model_path=MODEL_PATH, device="cuda:0").device == "cuda:0"
    monkeypatch.delenv("EDGEVISION_TORCH_DEVICE")
    assert YoloDetector(model_path=MODEL_PATH).device is None
