from edgevision.metrics import PerformanceMetrics


def build_detector(model_cfg: dict, metrics: PerformanceMetrics | None = None):
    """Create a detector from the `model` section of app.yaml.

    backend: pytorch (Ultralytics, opaque pre/post) | onnx (ONNX Runtime, ours)
             | tensorrt (TensorRT engine, ours).
    image_size: used by the pytorch backend; onnx/tensorrt graphs exported with a static
             shape carry their own size and only fall back to it when the graph is dynamic.
    """
    backend = model_cfg["backend"]
    path = model_cfg["paths"][backend]

    if backend == "pytorch":
        from edgevision.detector import YoloDetector

        device = model_cfg.get("device")
        return YoloDetector(
            model_path=path,
            confidence=model_cfg["confidence"],
            image_size=model_cfg["image_size"],
            device=None if device == "auto" else device,
            metrics=metrics,
        )

    if backend == "onnx":
        from edgevision.onnx_detector import OnnxDetector

        return OnnxDetector(
            model_path=path,
            confidence=model_cfg["confidence"],
            iou_threshold=model_cfg["iou_threshold"],
            image_size=model_cfg.get("image_size"),
            metrics=metrics,
        )

    if backend == "tensorrt":
        from edgevision.tensorrt_detector import TensorRTDetector

        return TensorRTDetector(
            engine_path=path,
            confidence=model_cfg["confidence"],
            iou_threshold=model_cfg["iou_threshold"],
            image_size=model_cfg.get("image_size"),
            metrics=metrics,
        )

    raise ValueError(
        f"Unknown backend: {backend!r} (expected 'pytorch', 'onnx' or 'tensorrt')"
    )
