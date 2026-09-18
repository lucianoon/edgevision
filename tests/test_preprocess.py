import numpy as np

from edgevision.preprocess import PAD_COLOR, letterbox, preprocess, to_tensor


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
