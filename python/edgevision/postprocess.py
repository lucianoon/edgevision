"""Decode the raw YOLO detect head, run NMS and map boxes back to the source frame.

Raw head layout: (1, 4 + num_classes, num_anchors). Rows 0-3 are the box as
(cx, cy, w, h) in letterboxed pixels; the remaining rows are per-class scores.
"""

import numpy as np

from edgevision.detector import Detection
from edgevision.preprocess import LetterboxInfo


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box (4,) and N boxes (N, 4), all xyxy."""
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    return np.divide(inter, union, out=np.zeros_like(union, dtype=np.float64), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy class-agnostic NMS. Returns indices to keep, highest score first."""
    order = np.argsort(-scores)
    keep = []

    while order.size > 0:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        ious = box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]

    return np.array(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Class-aware NMS: boxes of different classes never suppress each other.

    Implemented by shifting each class into its own coordinate region so a single
    class-agnostic pass can be used.
    """
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)
    max_coord = boxes.max() + 1
    offsets = class_ids.astype(np.float32)[:, None] * max_coord
    return nms(boxes + offsets, scores, iou_threshold)


def decode_raw(
    output: np.ndarray, conf_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(1, 4+nc, N) -> candidate boxes xyxy, scores and class ids above threshold."""
    predictions = output[0].T  # (N, 4+nc)
    class_scores = predictions[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]

    mask = scores >= conf_threshold
    boxes = xywh_to_xyxy(predictions[mask, :4])
    return boxes, scores[mask], class_ids[mask]


def scale_boxes(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Letterboxed xyxy -> source-frame xyxy, clipped to the frame."""
    scaled = boxes.copy()
    scaled[:, [0, 2]] = (scaled[:, [0, 2]] - info.pad_x) / info.scale
    scaled[:, [1, 3]] = (scaled[:, [1, 3]] - info.pad_y) / info.scale
    scaled[:, [0, 2]] = scaled[:, [0, 2]].clip(0, info.source_width)
    scaled[:, [1, 3]] = scaled[:, [1, 3]].clip(0, info.source_height)
    return scaled


def decode_end2end(
    output: np.ndarray, conf_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(1, max_det, 6) from a graph with NMS built in: rows are
    (x1, y1, x2, y2, score, class) in letterboxed pixels, zero-padded."""
    rows = output[0]
    mask = rows[:, 4] >= conf_threshold
    rows = rows[mask]
    return rows[:, :4], rows[:, 4], rows[:, 5].astype(np.int64)


def is_end2end_output(output: np.ndarray) -> bool:
    return output.ndim == 3 and output.shape[2] == 6


def postprocess(
    output: np.ndarray,
    info: LetterboxInfo,
    names: dict[int, str],
    conf_threshold: float,
    iou_threshold: float,
    max_det: int = 300,
) -> list[Detection]:
    if is_end2end_output(output):
        # NMS already ran inside the graph; nothing left to suppress.
        boxes, scores, class_ids = decode_end2end(output, conf_threshold)
        keep = np.arange(len(scores))[:max_det]
    else:
        boxes, scores, class_ids = decode_raw(output, conf_threshold)
        keep = batched_nms(boxes, scores, class_ids, iou_threshold)[:max_det]

    boxes = scale_boxes(boxes[keep], info)

    return [
        Detection(
            x1=float(box[0]),
            y1=float(box[1]),
            x2=float(box[2]),
            y2=float(box[3]),
            confidence=float(scores[i]),
            class_id=int(class_ids[i]),
            class_name=names[int(class_ids[i])],
        )
        for box, i in zip(boxes, keep)
    ]
