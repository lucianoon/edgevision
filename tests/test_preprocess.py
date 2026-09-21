import numpy as np
import pytest

from edgevision.preprocess import (
    DEFAULT_IMAGE_SIZE,
    PAD_COLOR,
    letterbox,
    preprocess,
    resolve_input_size,
    to_tensor,
)


def test_letterbox_portrait_pads_horizontally():
    frame = np.full((1080, 810, 3), 255, dtype=np.uint8)

    padded, info = letterbox(frame, 640)

    assert padded.shape == (640, 640, 3)
    assert info.scale == 640 / 1080
    assert info.pad_y == 0
    assert info.pad_x == 80  # (640 - 480) / 2
    assert tuple(padded[320, 0]) == PAD_COLOR  # left pad column
    assert tuple(padded[320, 320]) == (255, 255, 255)  # image content


def test_letterbox_landscape_pads_vertically():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    padded, info = letterbox(frame, 640)

    assert padded.shape == (640, 640, 3)
    assert info.scale == 1.0
    assert (info.pad_x, info.pad_y) == (0, 80)
    assert tuple(padded[0, 320]) == PAD_COLOR


def test_letterbox_square_frame_is_only_resized():
    frame = np.zeros((1280, 1280, 3), dtype=np.uint8)

    padded, info = letterbox(frame, 640)

    assert padded.shape == (640, 640, 3)
    assert (info.scale, info.pad_x, info.pad_y) == (0.5, 0, 0)


def test_to_tensor_converts_bgr_hwc_to_rgb_nchw_unit_range():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue channel in BGR

    tensor = to_tensor(image)

    assert tensor.shape == (1, 3, 2, 2)
    assert tensor.dtype == np.float32
    assert tensor[0, 2].max() == 1.0  # blue is the last RGB channel
    assert tensor[0, 0].max() == 0.0


def test_preprocess_output_shape():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tensor, info = preprocess(frame, 640)

    assert tensor.shape == (1, 3, 640, 640)
    assert info.source_height == 720 and info.source_width == 1280


def test_resolve_input_size_static_graph_wins_over_config():
    # ONNX Runtime reports ints for a static export; the configured size is ignored.
    assert resolve_input_size([1, 3, 512, 512], 640) == 512
    # TensorRT reports a tuple of ints for a static engine.
    assert resolve_input_size((1, 3, 640, 640), None) == 640


def test_resolve_input_size_dynamic_graph_uses_config_or_default():
    # ONNX Runtime: symbolic names; TensorRT: -1 for dynamic dims.
    assert resolve_input_size(["batch", 3, "height", "width"], 512) == 512
    assert resolve_input_size((-1, 3, -1, -1), None) == DEFAULT_IMAGE_SIZE


def test_resolve_input_size_rejects_non_square_graph():
    with pytest.raises(ValueError, match="non-square"):
        resolve_input_size([1, 3, 384, 640], None)
