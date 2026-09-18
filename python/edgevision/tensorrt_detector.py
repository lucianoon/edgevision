"""TensorRT backend: deserializes an engine built with trtexec from our ONNX export
and runs it with the TensorRT 10 API (execute_async_v3 + explicit tensor
addresses). Pre/post-processing are the same modules used by the ONNX backend.

Requires `tensorrt` and `cuda-python` (only available on NVIDIA hardware).
"""

import ast
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tensorrt as trt

try:  # cuda-python >= 12.6
    from cuda.bindings import runtime as cudart
except ImportError:  # older cuda-python
    from cuda import cudart

from edgevision.detector import Detection
from edgevision.metrics import PerformanceMetrics
from edgevision.postprocess import postprocess
from edgevision.preprocess import preprocess


def _check(result):
    """cuda-python returns (error, *values); raise on error, unwrap values."""
    err, *values = result if isinstance(result, tuple) else (result,)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {cudart.cudaGetErrorString(err)[1]}")
    return values[0] if len(values) == 1 else values


class _DeviceBuffer:
    def __init__(self, shape, dtype):
        self.host = np.empty(shape, dtype=dtype)
        self.nbytes = self.host.nbytes
        self.device = _check(cudart.cudaMalloc(self.nbytes))

    def upload(self, array: np.ndarray, stream):
        np.copyto(self.host, array.reshape(self.host.shape))
        _check(
            cudart.cudaMemcpyAsync(
                self.device,
                self.host.ctypes.data,
                self.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            )
        )

    def download(self, stream):
        _check(
            cudart.cudaMemcpyAsync(
                self.host.ctypes.data,
                self.device,
                self.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            )
        )

    def free(self):
        if self.device:
            cudart.cudaFree(self.device)
            self.device = 0


class TensorRTDetector:
    def __init__(
        self,
        engine_path: str,
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        class_names_path: str | None = None,
        metrics: PerformanceMetrics | None = None,
    ):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.stream = _check(cudart.cudaStreamCreate())
        self.buffers: dict[str, _DeviceBuffer] = {}
        self.input_name = None
        self.output_name = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            self.buffers[name] = _DeviceBuffer(shape, dtype)
            self.context.set_tensor_address(name, self.buffers[name].device)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Engine must have one input and one output tensor")

        self.names = _load_class_names(class_names_path or _sidecar_names(engine_path))
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.metrics = metrics

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.buffers[self.input_name].host.shape

    @property
    def input_dtype(self):
        return self.buffers[self.input_name].host.dtype

    def _stage(self, name: str):
        return self.metrics.stage(name) if self.metrics else nullcontext()

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        """Synchronous inference: H2D copy, execute, D2H copy, stream sync."""
        inp, out = self.buffers[self.input_name], self.buffers[self.output_name]
        inp.upload(tensor.astype(inp.host.dtype, copy=False), self.stream)
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        out.download(self.stream)
        _check(cudart.cudaStreamSynchronize(self.stream))
        return out.host.astype(np.float32, copy=False)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        with self._stage("preprocess"):
            tensor, info = preprocess(frame, self.image_size)

        with self._stage("inference"):
            output = self.infer(tensor)

        with self._stage("postprocess"):
            return postprocess(output, info, self.names, self.confidence, self.iou_threshold)

    def close(self):
        for buffer in self.buffers.values():
            buffer.free()
        if self.stream:
            cudart.cudaStreamDestroy(self.stream)
            self.stream = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _sidecar_names(engine_path: str) -> str:
    """Engines carry no metadata; class names live in <engine>.names.json next to it."""
    return str(Path(engine_path).with_suffix(".names.json"))


def _load_class_names(path: str) -> dict[int, str]:
    import json

    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)
    return {int(k): v for k, v in data.items()}
