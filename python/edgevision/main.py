"""Real-time detection over a webcam, file or stream, with per-stage metrics.

    edgevision --backend onnx --source videos/sample.mp4
    python -m edgevision.main --backend onnx --source 0 --no-display

The frame loop (`process_stream`) is independent of OpenCV windows and of argparse so it can
be driven by tests with fake sources and detectors; `main()` wires the real ones.
"""

import argparse
import importlib.metadata
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import yaml

from edgevision.detector import Detection, Detector
from edgevision.factory import build_detector
from edgevision.metrics import PerformanceMetrics
from edgevision.observability import (
    MetricsExporter,
    MetricsServer,
    StreamCounters,
    compose_callbacks,
    counting_callback,
)
from edgevision.video import VideoSource

DEFAULT_CONFIG = Path("configs/app.yaml")


def _version() -> str:
    try:
        return importlib.metadata.version("edgevision")
    except importlib.metadata.PackageNotFoundError:  # running from PYTHONPATH without an install
        return "0.0.0"


__version__ = _version()
BOX_COLOR = (0, 255, 0)


class FrameSource(Protocol):
    """The subset of VideoSource the loop needs (cv2.VideoCapture-style read())."""

    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...


# Called after each frame with the frame and its detections; return False to stop the loop.
FrameCallback = Callable[[np.ndarray, list[Detection]], bool]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_source(value: int | str) -> int | str:
    """Webcam indices arrive as strings from the CLI; everything else is a path or URL."""
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    for detection in detections:
        p1 = (int(detection.x1), int(detection.y1))
        p2 = (int(detection.x2), int(detection.y2))
        cv2.rectangle(frame, p1, p2, BOX_COLOR, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (p1[0], max(p1[1] - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BOX_COLOR,
            2,
        )


def format_stages(metrics: PerformanceMetrics) -> str:
    return " ".join(f"{name}={stats['mean_ms']:.1f}ms" for name, stats in metrics.summary().items())


def process_stream(
    detector: Detector,
    video: FrameSource,
    metrics: PerformanceMetrics,
    *,
    max_frames: int = 0,
    log_interval: int = 30,
    log: Callable[[str], None] = print,
    on_frame: FrameCallback | None = None,
) -> int:
    """Detect on every frame until the source ends, `max_frames` is reached (0 = no limit)
    or `on_frame` returns False. Decode and end-to-end latencies land in `metrics`.
    Returns the number of processed frames. Always releases the source."""
    try:
        while True:
            with metrics.stage("decode"):
                ok, frame = video.read()
            if not ok:
                break

            metrics.start_frame()
            detections = detector.detect(frame)
            metrics.end_frame()

            if log_interval and metrics.frame_count % log_interval == 0:
                log(
                    f"frame={metrics.frame_count} fps={metrics.fps:.1f} "
                    f"detections={len(detections)} {format_stages(metrics)}"
                )

            if on_frame is not None and not on_frame(frame, detections):
                break
            if max_frames and metrics.frame_count >= max_frames:
                break
    finally:
        video.release()

    return metrics.frame_count


def display_callback(backend: str, metrics: PerformanceMetrics) -> FrameCallback:
    """Annotates and shows each frame in an OpenCV window; 'q' stops the loop."""

    def show(frame: np.ndarray, detections: list[Detection]) -> bool:
        draw_detections(frame, detections)
        cv2.putText(
            frame,
            f"{backend} FPS: {metrics.fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow("EdgeVision", frame)
        return cv2.waitKey(1) & 0xFF != ord("q")

    return show


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EdgeVision real-time detection")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnx", "tensorrt"],
        help="Override model.backend",
    )
    parser.add_argument(
        "--source",
        help="Override video.source (webcam index or file/URL path)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless (no cv2.imshow window)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run until source ends or 'q')",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        help="serve Prometheus metrics (/metrics, /healthz) on this port; 0 = any free port",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)

    model_cfg = config["model"]
    video_cfg = config["video"]
    if args.backend:
        model_cfg["backend"] = args.backend
    backend = model_cfg["backend"]

    metrics = PerformanceMetrics()
    detector = build_detector(model_cfg, metrics)
    source = parse_source(video_cfg["source"] if args.source is None else args.source)
    display = video_cfg["display"] and not args.no_display

    counters = StreamCounters()
    callbacks: list[FrameCallback] = [counting_callback(counters)]
    if display:
        callbacks.append(display_callback(backend, metrics))

    server = None
    if args.metrics_port is not None:
        exporter = MetricsExporter(
            metrics=metrics, counters=counters, labels={"backend": backend}, version=__version__
        )
        server = MetricsServer(exporter.render, port=args.metrics_port).start()
        print(f"metrics: {server.url}")

    try:
        process_stream(
            detector,
            VideoSource(source),
            metrics,
            max_frames=args.max_frames,
            log_interval=config["metrics"]["log_interval_frames"],
            on_frame=compose_callbacks(*callbacks),
        )
    finally:
        if server is not None:
            server.stop()
        cv2.destroyAllWindows()

    print(
        f"done: backend={backend} frames={metrics.frame_count} "
        f"fps={metrics.fps:.1f} {format_stages(metrics)}"
    )


if __name__ == "__main__":
    main()
