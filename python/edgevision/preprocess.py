from dataclasses import dataclass

import cv2
import numpy as np

PAD_COLOR = (114, 114, 114)


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
