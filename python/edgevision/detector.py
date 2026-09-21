from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from edgevision.metrics import PerformanceMetrics


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class Detector(Protocol):
    """What every backend (PyTorch, ONNX Runtime, TensorRT) exposes to the app and the benchmark."""

    image_size: int
    metrics: PerformanceMetrics | None

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        image_size: int = 640,
        device: str | None = None,
        metrics: PerformanceMetrics | None = None,
    ):
        self.model = YOLO(model_path)
        self.metrics = metrics
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # Ultralytics does its own pre/post-processing inside predict(), so the
        # whole call is timed as a single 'inference' stage for this backend.
        timer = self.metrics.stage("inference") if self.metrics else nullcontext()
        with timer:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )

        # predict() returns a list (or a generator when streaming); one frame in, one result out.
        result = next(iter(results))
        if not isinstance(result, Results):
            raise TypeError(f"unexpected Ultralytics result type: {type(result).__name__}")

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.cpu().numpy()  # tensors live on the GPU when device is cuda
        xyxy = np.asarray(boxes.xyxy, dtype=np.float64)
        class_ids = np.asarray(boxes.cls, dtype=np.int64)
        confidences = np.asarray(boxes.conf, dtype=np.float64)

        return [
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(confidence),
                class_id=int(class_id),
                class_name=result.names[int(class_id)],
            )
            for (x1, y1, x2, y2), class_id, confidence in zip(
                xyxy, class_ids, confidences, strict=True
            )
        ]
