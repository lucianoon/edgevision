"""The frame loop and CLI helpers of edgevision.main, driven by fake sources and detectors."""

from pathlib import Path

import numpy as np
import pytest

from edgevision.detector import Detection
from edgevision.main import (
    draw_detections,
    format_stages,
    load_config,
    main,
    parse_args,
    parse_source,
    process_stream,
)
from edgevision.metrics import PerformanceMetrics

PERSON = Detection(x1=10, y1=20, x2=60, y2=120, confidence=0.9, class_id=0, class_name="person")


class FakeSource:
    """Yields `n` blank frames, then reports end of stream. Counts release() calls."""

    def __init__(self, n: int, shape=(48, 64, 3)):
        self.remaining = n
        self.shape = shape
        self.released = 0

    def read(self):
        if self.remaining == 0:
            return False, None
        self.remaining -= 1
        return True, np.zeros(self.shape, dtype=np.uint8)

    def release(self):
        self.released += 1


class FakeDetector:
    image_size = 640

    def __init__(self, detections=(PERSON,)):
        self.metrics = None
        self.detections = list(detections)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return self.detections


def test_process_stream_runs_until_the_source_ends_and_releases_it():
    source, detector, metrics = FakeSource(5), FakeDetector(), PerformanceMetrics()

    frames = process_stream(detector, source, metrics, log_interval=0)

    assert frames == 5 == detector.calls == metrics.frame_count
    assert source.released == 1
    assert set(metrics.summary()) == {"decode", "end_to_end"}


def test_process_stream_stops_at_max_frames():
    source, detector, metrics = FakeSource(100), FakeDetector(), PerformanceMetrics()

    assert process_stream(detector, source, metrics, max_frames=7, log_interval=0) == 7
    assert source.released == 1


def test_process_stream_stops_when_the_callback_says_so():
    seen = []

    def on_frame(frame, detections):
        seen.append(len(detections))
        return len(seen) < 3  # "q" pressed on the third frame

    frames = process_stream(
        FakeDetector(), FakeSource(100), PerformanceMetrics(), log_interval=0, on_frame=on_frame
    )

    assert frames == 3
    assert seen == [1, 1, 1]


def test_process_stream_logs_every_interval():
    lines = []

    process_stream(
        FakeDetector(), FakeSource(10), PerformanceMetrics(), log_interval=4, log=lines.append
    )

    assert len(lines) == 2  # frames 4 and 8
    assert lines[0].startswith("frame=4 ") and "detections=1" in lines[0]
    assert "decode=" in lines[0] and "end_to_end=" in lines[0]


def test_process_stream_releases_the_source_when_the_detector_fails():
    class Broken(FakeDetector):
        def detect(self, frame):
            raise RuntimeError("boom")

    source = FakeSource(3)
    with pytest.raises(RuntimeError, match="boom"):
        process_stream(Broken(), source, PerformanceMetrics(), log_interval=0)
    assert source.released == 1


def test_draw_detections_paints_the_box():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    draw_detections(frame, [PERSON])

    assert frame[20, 10:60].any()  # top edge of the box is green now
    assert not frame[150:, 100:].any()  # untouched region stays black


def test_format_stages_is_compact():
    metrics = PerformanceMetrics()
    metrics.record("decode", 2.0)
    metrics.record("end_to_end", 10.0)

    assert format_stages(metrics) == "decode=2.0ms end_to_end=10.0ms"


def test_parse_source_turns_webcam_index_into_int():
    assert parse_source("0") == 0
    assert parse_source(2) == 2
    assert parse_source("videos/x.mp4") == "videos/x.mp4"
    assert parse_source("rtsp://cam/1") == "rtsp://cam/1"


def test_parse_args_defaults_and_overrides():
    args = parse_args([])
    assert args.backend is None and args.max_frames == 0 and not args.no_display

    args = parse_args(["--backend", "onnx", "--source", "0", "--no-display", "--max-frames", "5"])
    assert (args.backend, args.source, args.no_display, args.max_frames) == ("onnx", "0", True, 5)


def test_load_config_reads_the_shipped_app_yaml():
    config = load_config(Path("configs/app.yaml"))

    assert config["model"]["backend"] in {"pytorch", "onnx", "tensorrt"}
    assert set(config["model"]["paths"]) == {"pytorch", "onnx", "tensorrt"}
    assert config["metrics"]["log_interval_frames"] > 0


def test_main_runs_headless_on_a_clip_with_the_onnx_backend(clip, capsys):
    # Full wiring: app.yaml -> factory -> VideoSource -> process_stream, no window.
    main(["--backend", "onnx", "--source", str(clip), "--no-display", "--max-frames", "3"])

    out = capsys.readouterr().out
    assert "done: backend=onnx frames=3" in out
    assert "decode=" in out and "end_to_end=" in out
