import argparse
from pathlib import Path

import cv2
import yaml

from edgevision.detector import Detection, YoloDetector
from edgevision.metrics import PerformanceMetrics
from edgevision.video import VideoSource

DEFAULT_CONFIG = Path("configs/app.yaml")


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_detections(frame, detections: list[Detection]):
    for detection in detections:
        p1 = (int(detection.x1), int(detection.y1))
        p2 = (int(detection.x2), int(detection.y2))

        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

        label = f"{detection.class_name} {detection.confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (p1[0], max(p1[1] - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="EdgeVision PyTorch baseline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    model_cfg = config["model"]
    video_cfg = config["video"]
    log_interval = config["metrics"]["log_interval_frames"]

    device = model_cfg.get("device")
    if device == "auto":
        device = None

    detector = YoloDetector(
        model_path=model_cfg["path"],
        confidence=model_cfg["confidence"],
        image_size=model_cfg["image_size"],
        device=device,
    )

    source = video_cfg["source"] if args.source is None else args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    display = video_cfg["display"] and not args.no_display

    video = VideoSource(source)
    metrics = PerformanceMetrics()

    try:
        while True:
            ok, frame = video.read()

            if not ok:
                break

            metrics.start_frame()
            detections = detector.detect(frame)
            metrics.end_frame()

            if metrics.frame_count % log_interval == 0:
                print(
                    f"frame={metrics.frame_count} "
                    f"latency={metrics.average_latency_ms:.1f}ms "
                    f"fps={metrics.fps:.1f} "
                    f"detections={len(detections)}"
                )

            if display:
                draw_detections(frame, detections)

                cv2.putText(
                    frame,
                    f"FPS: {metrics.fps:.1f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("EdgeVision", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and metrics.frame_count >= args.max_frames:
                break

    finally:
        video.release()
        cv2.destroyAllWindows()

    print(
        f"done: frames={metrics.frame_count} "
        f"avg_latency={metrics.average_latency_ms:.1f}ms "
        f"fps={metrics.fps:.1f}"
    )


if __name__ == "__main__":
    main()
