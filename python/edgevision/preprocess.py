from dataclasses import dataclass

import cv2
import numpy as np

PAD_COLOR = (114, 114, 114)
DEFAULT_IMAGE_SIZE = 640


def resolve_input_size(shape, requested: int | None, source: str = "graph") -> int:
    """Square input size the pre-processing must produce for a graph of input `shape`.

    `shape` is (N, 3, H, W) as reported by ONNX Runtime (ints or symbolic strings) or
    TensorRT (ints, -1 for dynamic). A static graph fixed H = W at export time and is the
    single source of truth: the configured `requested` size is ignored for it, so an
    engine exported at 512 is fed 512 whatever app.yaml says. Only a dynamic graph uses
    `requested` (or DEFAULT_IMAGE_SIZE).
    """
    h, w = shape[-2], shape[-1]
    static = all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in (h, w))
    if static:
        if h != w:
            raise ValueError(f"{source}: non-square input {h}x{w} is not supported")
        return int(h)
    return int(requested) if requested is not None else DEFAULT_IMAGE_SIZE


@dataclass(frozen=True)
class LetterboxInfo:
    """Geometry needed to map letterboxed coordinates back to the source frame."""

    scale: float
    pad_x: float
    pad_y: float
    source_height: int
    source_width: int


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize keeping aspect ratio and center-pad to a size x size BGR image."""
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)

    resized = frame
    if (new_w, new_h) != (w, h):
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_COLOR
    )
    info = LetterboxInfo(scale=scale, pad_x=left, pad_y=top, source_height=h, source_width=w)
    return padded, info


def to_tensor(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> RGB float32 NCHW in [0, 1]."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1)
    return np.ascontiguousarray(chw[None], dtype=np.float32) / 255.0


def preprocess(frame: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxInfo]:
    padded, info = letterbox(frame, size)
    return to_tensor(padded), info
