"""benchmark.measure() and the report schema, without models or GPUs."""

import argparse
import json

from edgevision.benchmark import build_report, environment, main, measure, parse_args
from edgevision.metrics import PerformanceMetrics
from tests.test_main import FakeDetector, FakeSource


def test_measure_discards_warmup_and_loops_short_clips():
    detector = FakeDetector()
    opened = []

    def open_source():
        opened.append(1)
        return FakeSource(4)  # shorter than warmup + frames: must be reopened

    metrics = measure(detector, open_source, frames=6, warmup=3)

    assert detector.calls == 9  # 3 warm-up + 6 measured
    assert metrics.frame_count == 6  # only measured frames remain
    assert len(metrics.latencies) == 6
    assert len(opened) == 3  # 4 + 4 + 1 frames
    assert detector.metrics is metrics  # the detector times its stages into the returned window


def test_measure_with_zero_warmup_keeps_every_frame():
    metrics = measure(FakeDetector(), lambda: FakeSource(10), frames=5, warmup=0)

    assert metrics.frame_count == 5


def test_build_report_has_the_schema_the_cpp_runtime_writes():
    args = argparse.Namespace(
        backend="onnx", label="fp16", source="videos/clip.mp4", frames=100, warmup=10
    )
    model_cfg = {"paths": {"onnx": "models/onnx/yolo26n.onnx"}, "confidence": 0.5}
    metrics = PerformanceMetrics()
    metrics.record("inference", 4.0)
    metrics.latencies.append(10.0)

    report = build_report(
        args, model_cfg, FakeDetector(), metrics, "20260921T000000Z", {"python": "3.12"}
    )

    assert set(report) == {
        "timestamp_utc", "backend", "label", "model_path", "image_size", "confidence",
        "source", "frames", "warmup", "fps", "stages", "environment",
    }  # fmt: skip
    assert report["image_size"] == 640 and report["fps"] == 100.0
    assert set(report["stages"]) == {"inference", "end_to_end"}
    json.dumps(report)  # serialisable as written


def test_environment_reports_the_stack():
    env = environment(FakeDetector())

    assert env["python"].startswith("3.")
    assert env["torch"] and env["onnxruntime"] and env["ultralytics"]
    assert env["ort_providers"] is None  # FakeDetector has none
    assert "tensorrt" in env  # None on CPU, a version on the GPU box


def test_parse_args_requires_backend_and_has_defaults():
    args = parse_args(["--backend", "onnx"])

    assert (args.frames, args.warmup, args.label, args.out) == (100, 10, "", None)


def test_main_writes_a_report_for_the_onnx_backend(clip, tmp_path, capsys):
    out = tmp_path / "report.json"

    main(
        [
            "--backend",
            "onnx",
            "--source",
            str(clip),
            "--frames",
            "3",
            "--warmup",
            "1",
            "--out",
            str(out),
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["backend"] == "onnx" and report["frames"] == 3 and report["warmup"] == 1
    assert report["image_size"] == 640 and report["environment"]["onnxruntime"]
    assert set(report["stages"]) >= {
        "decode",
        "preprocess",
        "inference",
        "postprocess",
        "end_to_end",
    }
    assert "saved:" in capsys.readouterr().out
