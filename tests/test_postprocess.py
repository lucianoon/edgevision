import numpy as np
import pytest

from edgevision.postprocess import (
    batched_nms,
    box_iou,
    decode_raw,
    nms,
    postprocess,
    scale_boxes,
    xywh_to_xyxy,
)
from edgevision.preprocess import LetterboxInfo


def test_xywh_to_xyxy():
    out = xywh_to_xyxy(np.array([[10.0, 20.0, 4.0, 6.0]]))
    np.testing.assert_allclose(out, [[8.0, 17.0, 12.0, 23.0]])


def test_iou_identical_disjoint_and_partial():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    others = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],  # identical
            [20.0, 20.0, 30.0, 30.0],  # disjoint
            [5.0, 0.0, 15.0, 10.0],  # half overlap: inter 50, union 150
        ]
    )
    np.testing.assert_allclose(box_iou(box, others), [1.0, 0.0, 50 / 150])


def test_iou_with_zero_area_boxes_is_zero():
    box = np.array([0.0, 0.0, 0.0, 0.0])
    assert box_iou(box, np.array([[0.0, 0.0, 0.0, 0.0]]))[0] == 0.0


def test_nms_suppresses_overlapping_keeps_distant():
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # overlaps the first
            [50.0, 50.0, 60.0, 60.0],  # far away
        ]
    )
    scores = np.array([0.8, 0.9, 0.5])

    keep = nms(boxes, scores, iou_threshold=0.5)

    assert keep.tolist() == [1, 2]  # highest score first, overlap removed


def test_nms_empty_input():
    assert nms(np.empty((0, 4)), np.empty(0), 0.5).size == 0


def test_batched_nms_keeps_overlapping_boxes_of_different_classes():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    scores = np.array([0.9, 0.8])

    same_class = batched_nms(boxes, scores, np.array([0, 0]), 0.5)
    different_class = batched_nms(boxes, scores, np.array([0, 1]), 0.5)

    assert same_class.tolist() == [0]
    assert sorted(different_class.tolist()) == [0, 1]


def _raw_output(candidates, num_classes=3):
    """Build a (1, 4+nc, N) head output from (cx, cy, w, h, class_id, score) rows."""
    out = np.zeros((1, 4 + num_classes, len(candidates)), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(candidates):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + cls, i] = score
    return out


def test_decode_raw_filters_by_confidence_and_picks_best_class():
    output = _raw_output([(50, 50, 20, 20, 1, 0.9), (100, 100, 10, 10, 2, 0.3)])

    boxes, scores, class_ids = decode_raw(output, conf_threshold=0.5)

    assert len(boxes) == 1
    np.testing.assert_allclose(boxes[0], [40, 40, 60, 60])
    assert scores[0] == pytest.approx(0.9)
    assert class_ids[0] == 1


def test_scale_boxes_undoes_letterbox_and_clips():
    # source 200x100 (w x h) letterboxed into 100x100: scale 0.5, pad_y 25
    info = LetterboxInfo(scale=0.5, pad_x=0, pad_y=25, source_height=100, source_width=200)
    boxes = np.array([[10.0, 35.0, 60.0, 55.0], [-5.0, 0.0, 200.0, 100.0]])

    scaled = scale_boxes(boxes, info)

    np.testing.assert_allclose(scaled[0], [20, 20, 120, 60])
    np.testing.assert_allclose(scaled[1], [0, 0, 200, 100])  # clipped to frame


def test_postprocess_end_to_end_builds_detections():
    info = LetterboxInfo(scale=1.0, pad_x=0, pad_y=0, source_height=200, source_width=200)
    output = _raw_output(
        [
            (50, 50, 20, 20, 0, 0.9),
            (51, 51, 20, 20, 0, 0.7),  # duplicate of the first, same class
            (150, 150, 20, 20, 2, 0.8),
        ]
    )
    names = {0: "person", 1: "bicycle", 2: "car"}

    detections = postprocess(output, info, names, conf_threshold=0.5, iou_threshold=0.5)

    assert [(d.class_name, round(d.confidence, 1)) for d in detections] == [
        ("person", 0.9),
        ("car", 0.8),
    ]
    assert (detections[0].x1, detections[0].y1) == (40.0, 40.0)
