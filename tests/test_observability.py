"""Prometheus exposition and the metrics server, plus the --metrics-port wiring of both CLIs."""

import json
import re
import urllib.request
from pathlib import Path

from edgevision.events import EventEngine, TrackObservation, load_rules, parse_rules
from edgevision.events.cli import main as events_main
from edgevision.main import main as app_main
from edgevision.metrics import PerformanceMetrics
from edgevision.observability import (
    CONTENT_TYPE,
    MetricsExporter,
    MetricsServer,
    StreamCounters,
    compose_callbacks,
    counting_callback,
)

DUMP = Path("tests/data/tracks_pedestrian_512.jsonl")
RULES = Path("configs/rules.example.yaml")


def sample(text: str, name: str, labels: str = "") -> float:
    match = re.search(rf"^{re.escape(name)}{re.escape(labels)} (\S+)$", text, re.M)
    assert match, f"{name}{labels} missing in:\n{text}"
    return float(match.group(1))


def test_render_metrics_window_and_counters():
    metrics = PerformanceMetrics()
    for _ in range(4):
        metrics.record("decode", 2.0)
        metrics.start_frame()
        metrics.end_frame()
    metrics.latencies.clear()
    metrics.latencies.extend([10.0, 10.0, 10.0, 10.0])
    counters = StreamCounters()
    counters.observe(3)
    counters.observe(0)

    text = MetricsExporter(
        metrics=metrics, counters=counters, labels={"backend": "onnx"}, version="1.2.3"
    ).render()

    assert sample(text, "edgevision_build_info", '{version="1.2.3",backend="onnx"}') == 1
    assert sample(text, "edgevision_fps", '{backend="onnx"}') == 100
    assert sample(text, "edgevision_frames_total", '{backend="onnx"}') == 4
    assert sample(text, "edgevision_detections_total", '{backend="onnx"}') == 3
    assert (
        sample(
            text, "edgevision_stage_latency_window_ms", '{backend="onnx",stage="decode",stat="p95"}'
        )
        == 2
    )
    assert (
        sample(text, "edgevision_stage_window_samples", '{backend="onnx",stage="end_to_end"}') == 4
    )
    assert text.count("# TYPE edgevision_stage_latency_window_ms gauge") == 1
    assert text.count("# TYPE edgevision_frames_total counter") == 1
    assert text.endswith("\n")


def test_render_events_and_label_escaping():
    rules = parse_rules(
        {
            "zones": [
                {"name": 'shop "A"', "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]]}
            ],
            "lines": [{"name": "door", "a": [100, 0], "b": [100, 400]}],
        }
    )
    engine = EventEngine(rules, fps=10)
    person = TrackObservation(1, "person", 40, 110, 60, 150)
    engine.update(1, [person])
    engine.update(2, [TrackObservation(1, "person", 140, 110, 160, 150)])

    text = MetricsExporter(events=engine).render()

    assert sample(text, "edgevision_line_crossings_total", '{line="door",label="in"}') == 1
    assert sample(text, "edgevision_line_crossings_total", '{line="door",label="out"}') == 0
    assert sample(text, "edgevision_zone_entries_total", '{zone="shop \\"A\\""}') == 1
    assert sample(text, "edgevision_zone_occupancy", '{zone="shop \\"A\\""}') == 1
    assert sample(text, "edgevision_tracks_active") == 1
    assert sample(text, "edgevision_events_total", '{type="line_cross"}') == 1
    assert sample(text, "edgevision_events_total", '{type="dwell_exceeded"}') == 0


def test_engine_summary_carries_counts():
    engine = EventEngine(parse_rules({}), fps=10)
    engine.update(1, [TrackObservation(1, "person", 0, 0, 1, 1)])
    summary = engine.summary()
    assert summary["active_tracks"] == 1
    assert set(summary["events"]) == {"zone_enter", "zone_exit", "dwell_exceeded", "line_cross"}


def test_server_serves_metrics_and_health():
    server = MetricsServer(lambda: "edgevision_fps 42\n", host="127.0.0.1", port=0).start()
    try:
        assert server.url == f"http://127.0.0.1:{server.port}/metrics"
        with urllib.request.urlopen(server.url) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == CONTENT_TYPE
            assert response.read() == b"edgevision_fps 42\n"
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/healthz") as response:
            assert response.read() == b"ok\n"
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{server.port}/nope")
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:  # pragma: no cover
            raise AssertionError("expected 404")
    finally:
        server.stop()


def test_compose_callbacks_runs_all_and_aggregates_the_stop_signal():
    calls = []
    keep = compose_callbacks(lambda *a: calls.append("a"), lambda *a: calls.append("b") or True)
    stop = compose_callbacks(lambda *a: False, lambda *a: calls.append("c") or True)

    assert keep(1, 2) is True
    assert stop(1, 2) is False
    assert calls == ["a", "b", "c"]  # the False one did not short-circuit the others


def test_counting_callback():
    counters = StreamCounters()
    cb = counting_callback(counters)
    assert cb(None, [1, 2, 3]) is True and cb(None, []) is True
    assert (counters.frames, counters.detections) == (2, 3)


def test_app_main_exposes_metrics_while_running(clip, capsys):
    app_main(
        [
            "--backend",
            "onnx",
            "--source",
            str(clip),
            "--no-display",
            "--max-frames",
            "3",
            "--metrics-port",
            "0",
        ]
    )

    out = capsys.readouterr().out
    assert re.search(r"^metrics: http://localhost:\d+/metrics$", out, re.M)
    assert "done: backend=onnx frames=3" in out


def test_events_cli_exposes_metrics_while_running(tmp_path, capsys):
    events_main(
        [
            str(DUMP),
            "--rules",
            str(RULES),
            "--out",
            str(tmp_path / "e.jsonl"),
            "--metrics-port",
            "0",
        ]
    )

    captured = capsys.readouterr()
    assert re.search(r"^metrics: http://localhost:\d+/metrics$", captured.err, re.M)
    summary = json.loads(captured.out)
    assert summary["events"]["line_cross"] == 14 and summary["active_tracks"] == 0
    assert isinstance(load_rules(RULES).lines, tuple)
