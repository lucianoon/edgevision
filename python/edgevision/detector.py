from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        image_size: int = 640,
        device: str | None = None,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None:
            return []

        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=result.names[class_id],
                )
            )

        return detections
