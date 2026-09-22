import ast
import os
from contextlib import nullcontext

import numpy as np
import onnxruntime as ort

from edgevision.detector import Detection
from edgevision.metrics import PerformanceMetrics
from edgevision.postprocess import postprocess
from edgevision.preprocess import preprocess, resolve_input_size

PREFERRED_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
PROVIDERS_ENV = "EDGEVISION_ORT_PROVIDERS"


def providers_from_env() -> list[str] | None:
    """Comma-separated override, e.g. EDGEVISION_ORT_PROVIDERS=CPUExecutionProvider.

    Needed where onnxruntime-gpu has no kernels for the GPU: the aarch64 wheel lacks sm_75,
    so on a g5g (T4G) every CUDA node fails with cudaErrorNoKernelImageForDevice."""
    value = os.environ.get(PROVIDERS_ENV, "").strip()
    return [p.strip() for p in value.split(",") if p.strip()] or None


def _load_class_names(session: ort.InferenceSession) -> dict[int, str]:
    metadata = session.get_modelmeta().custom_metadata_map
    if "names" not in metadata:
        raise ValueError("ONNX model has no 'names' metadata; export it with Ultralytics")
    return ast.literal_eval(metadata["names"])


class OnnxDetector:
    """Runs the exported YOLO graph with ONNX Runtime using our own pre/post-processing."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        image_size: int | None = None,
        providers: list[str] | None = None,
        metrics: PerformanceMetrics | None = None,
    ):
        available = ort.get_available_providers()
        if providers is None:
            providers = providers_from_env() or [p for p in PREFERRED_PROVIDERS if p in available]

        self.session = ort.InferenceSession(model_path, providers=providers)
        graph_input = self.session.get_inputs()[0]
        self.input_name = graph_input.name
        self.names = _load_class_names(self.session)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        # Static graphs carry their size; `image_size` only matters for dynamic exports.
        self.image_size = resolve_input_size(graph_input.shape, image_size, model_path)
        self.metrics = metrics

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def _stage(self, name: str):
        return self.metrics.stage(name) if self.metrics else nullcontext()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        with self._stage("preprocess"):
            tensor, info = preprocess(frame, self.image_size)

        with self._stage("inference"):
            output = self.session.run(None, {self.input_name: tensor})[0]

        with self._stage("postprocess"):
            return postprocess(output, info, self.names, self.confidence, self.iou_threshold)
