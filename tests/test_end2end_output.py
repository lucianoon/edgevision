"""Graph-with-NMS output (1, max_det, 6) vs our raw-head decode + NMS."""

import cv2
import numpy as np
import pytest
from ultralytics.utils import ASSETS

from edgevision.onnx_detector import OnnxDetector
from edgevision.postprocess import box_iou, decode_end2end, is_end2end_output, postprocess
from edgevision.preprocess import LetterboxInfo

RAW_ONNX = "models/onnx/yolo26n.onnx"
NMS_ONNX = "models/onnx/yolo26n_nms.onnx"


def test_is_end2end_output_by_shape():
    assert is_end2end_output(np.zeros((1, 300, 6), dtype=np.float32))
    assert not is_end2end_output(np.zeros((1, 84, 8400), dtype=np.float32))


def test_decode_end2end_drops_padding_and_low_scores():
    out = np.zeros((1, 4, 6), dtype=np.float32)
    out[0, 0] = [10, 10, 50, 50, 0.9, 2]
    out[0, 1] = [20, 20, 60, 60, 0.3, 0]  # below threshold

    boxes, scores, class_ids = decode_end2end(out, conf_threshold=0.5)

    assert boxes.shape == (1, 4)
    assert scores.tolist() == pytest.approx([0.9])
    assert class_ids.tolist() == [2]


def test_postprocess_end2end_skips_nms_and_rescales():
    info = LetterboxInfo(scale=0.5, pad_x=0, pad_y=25, source_height=100, source_width=200)
    out = np.zeros((1, 2, 6), dtype=np.float32)
    out[0, 0] = [10, 35, 60, 55, 0.9, 0]
    out[0, 1] = [10, 35, 60, 55, 0.8, 0]  # duplicate: must survive, NMS is the graph's job

    detections = postprocess(out, info, {0: "person"}, conf_threshold=0.5, iou_threshold=0.45)

    assert len(detections) == 2
    assert (detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2) == (
        20,
        20,
        120,
        60,
    )


def test_nms_graph_matches_our_nms_on_sample_image():
    frame = cv2.imread(str(ASSETS / "bus.jpg"))
    ours = OnnxDetector(RAW_ONNX, confidence=0.5, iou_threshold=0.45).detect(frame)
    graph = OnnxDetector(NMS_ONNX, confidence=0.5, iou_threshold=0.45).detect(frame)

    assert sorted(d.class_name for d in graph) == sorted(d.class_name for d in ours)

    ref = np.array([[d.x1, d.y1, d.x2, d.y2] for d in ours])
    for d in graph:
        ious = box_iou(np.array([d.x1, d.y1, d.x2, d.y2]), ref)
        best = int(ious.argmax())
        # 1.0 on the machine both graphs were exported on; 0.91 seen on Linux CI, where
        # each run re-exports the graphs with a different torch build.
        assert ious[best] > 0.85, f"{d.class_name}: IoU {ious[best]:.3f}"
        # Same weights, but the two graphs are separate exports (fusions differ) and ORT
        # kernels differ per platform: 0.02-0.06 seen on Linux CI. Boxes/classes are the check.
        assert abs(ours[best].confidence - d.confidence) < 0.1
