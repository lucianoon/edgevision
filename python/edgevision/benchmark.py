"""Measure per-stage latency of a detector backend over a video source.

    python -m edgevision.benchmark --backend onnx --source videos/sample.mp4 --frames 100

Writes a JSON report to benchmarks/results/ so runs can be compared over time.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from edgevision.factory import build_detector
from edgevision.main import load_config
from edgevision.metrics import PerformanceMetrics
from edgevision.video import VideoSource

RESULTS_DIR = Path("benchmarks/results")


def environment(detector) -> dict:
    import numpy
    import onnxruntime
    import torch
    import ultralytics

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "onnxruntime": onnxruntime.__version__,
        "ort_providers": getattr(detector, "providers", None),
        "ultralytics": ultralytics.__version__,
    }


def run(model_cfg: dict, source, frames: int, warmup: int) -> tuple[PerformanceMetrics, object]:
    metrics = PerformanceMetrics(window_size=frames)
    detector = build_detector(model_cfg, metrics)
    video = VideoSource(source)

    processed = 0
    try:
        while processed < warmup + frames:
            with metrics.stage("decode"):
                ok, frame = video.read()

            if not ok:  # loop short clips until enough frames were processed
                video.release()
                video = VideoSource(source)
                continue

            metrics.start_frame()
            detector.detect(frame)
            metrics.end_frame()
            processed += 1

            if processed == warmup:
                metrics = PerformanceMetrics(window_size=frames)
                detector.metrics = metrics
    finally:
        video.release()

    return metrics, detector


def print_report(backend: str, summary: dict, fps: float):
    print(f"\nbackend={backend}  fps={fps:.1f}")
    print(f"{'stage':<14}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}  n")
    for name, s in summary.items():
        print(
            f"{name:<14}{s['mean_ms']:>9.1f}{s['p50_ms']:>9.1f}"
            f"{s['p95_ms']:>9.1f}{s['max_ms']:>9.1f}  {s['samples']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="EdgeVision detector benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/app.yaml"))
    parser.add_argument("--backend", choices=["pytorch", "onnx"], required=True)
    parser.add_argument("--source", default="videos/sample.mp4")
    parser.add_argument("--frames", type=int, default=100, help="measured frames")
    parser.add_argument("--warmup", type=int, default=10, help="frames discarded before measuring")
    parser.add_argument("--out", type=Path, help="JSON path (default: benchmarks/results/<backend>_<utc>.json)")
    return parser.parse_args()


def main():
    args = parse_args()
    model_cfg = load_config(args.config)["model"]
    model_cfg["backend"] = args.backend

    source = int(args.source) if args.source.isdigit() else args.source
    metrics, detector = run(model_cfg, source, args.frames, args.warmup)

    summary = metrics.summary()
    print_report(args.backend, summary, metrics.fps)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or RESULTS_DIR / f"{args.backend}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp_utc": stamp,
        "backend": args.backend,
        "model_path": model_cfg["paths"][args.backend],
        "image_size": model_cfg["image_size"],
        "confidence": model_cfg["confidence"],
        "source": str(args.source),
        "frames": args.frames,
        "warmup": args.warmup,
        "fps": metrics.fps,
        "stages": summary,
        "environment": environment(detector),
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
