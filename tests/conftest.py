"""Shared fixtures."""

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    """A 5-frame 64x48 mp4 (blank frames): enough to drive decoders and CLIs, no detections."""
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 40, dtype=np.uint8))
    writer.release()
    return path
