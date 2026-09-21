import pytest

from edgevision.detector import YoloDetector
from edgevision.factory import build_detector
from edgevision.metrics import PerformanceMetrics
from edgevision.onnx_detector import OnnxDetector

PATHS = {
    "pytorch": "models/pytorch/yolo26n.pt",
    "onnx": "models/onnx/yolo26n_nms.onnx",
    "tensorrt": "models/tensorrt/yolo26n_nms_512_fp16.engine",
}


def config(backend: str, **overrides) -> dict:
    cfg = {
        "backend": backend,
        "paths": dict(PATHS),
        "confidence": 0.5,
        "iou_threshold": 0.45,
        "image_size": 640,
        "device": "auto",
    }
    cfg.update(overrides)
    return cfg


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        build_detector({"backend": "openvino", "paths": {"openvino": "x"}})


def test_pytorch_backend_maps_auto_device_to_none():
    metrics = PerformanceMetrics()

    detector = build_detector(config("pytorch"), metrics)

    assert isinstance(detector, YoloDetector)
    assert detector.device is None and detector.image_size == 640
    assert detector.metrics is metrics


def test_pytorch_backend_keeps_an_explicit_device():
    assert build_detector(config("pytorch", device="cpu")).device == "cpu"


def test_onnx_backend_takes_the_size_from_the_graph_not_the_config():
    detector = build_detector(config("onnx", image_size=512))

    assert isinstance(detector, OnnxDetector)
    assert detector.image_size == 640  # static 640 export wins over image_size: 512
    assert detector.confidence == 0.5 and detector.iou_threshold == 0.45


def test_onnx_backend_works_without_image_size_in_the_config():
    cfg = config("onnx")
    del cfg["image_size"]

    assert build_detector(cfg).image_size == 640
