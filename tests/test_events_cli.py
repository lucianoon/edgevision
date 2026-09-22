"""The events CLI end to end on the real 300-frame track dump of the pedestrian clip, plus sinks."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from edgevision.events import EventEngine, load_rules
from edgevision.events.cli import main, read_dump, run
from edgevision.events.engine import Event
from edgevision.events.sinks import JsonlSink, MultiSink, WebhookSink

DUMP = Path(
    "tests/data/tracks_pedestrian_512.jsonl"
)  # edgevision_trt --track --dump-tracks, T4, Sprint 8
RULES = Path("configs/rules.example.yaml")


def test_read_dump_parses_the_cpp_format():
    frames = list(read_dump(DUMP.read_text(encoding="utf-8").splitlines()))

    assert len(frames) == 300 and frames[0][0] == 1 and frames[-1][0] == 300
    first = frames[0][1]
    assert {o.class_name for o in first} == {"person"}
    assert first[0].track_id == 3 and first[0].x2 > first[0].x1


def test_real_clip_counts_are_stable():
    # Regression on the measured dump: people walk both ways across the platform.
    engine = EventEngine(load_rules(RULES), fps=25.0)
    events = []
    for frame, observations in read_dump(DUMP.read_text(encoding="utf-8").splitlines()):
        events += engine.update(frame, observations)
    events += engine.close()
    summary = engine.summary()

    assert (
        summary["frames"] == 300 and summary["duration_s"] == 12.0 and summary["tracks_seen"] == 45
    )
    # Exact values are a regression snapshot of the engine on this dump (people walk both ways).
    assert summary["lines"]["gate"] == {"east": 6, "west": 8}
    assert summary["zones"]["platform"]["entries"] == 44
    assert summary["zones"]["platform"]["dwell_alerts"] == 3  # e.g. id 21 stays 232 frames = 9.3 s
    assert len(events) == 105
    assert summary["zones"]["platform"]["occupancy"] == 0  # close() emptied the zone
    assert {e.type for e in events} == {"zone_enter", "zone_exit", "line_cross", "dwell_exceeded"}
    reasons = {e.data["reason"] for e in events if e.type == "zone_exit"}
    assert reasons <= {"left", "lost", "end_of_stream"} and "end_of_stream" in reasons


def test_cli_writes_events_and_prints_summary(tmp_path, capsys):
    out = tmp_path / "events.jsonl"

    main([str(DUMP), "--rules", str(RULES), "--fps", "25", "--out", str(out)])

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 20
    first = json.loads(lines[0])
    assert set(first) == {"type", "frame", "time_s", "track_id", "class_name", "name", "data"}
    summary = json.loads(capsys.readouterr().out)
    assert summary["lines"]["gate"].keys() == {"east", "west"}


def test_cli_stdout_events_keep_stdout_pure_jsonl(capsys):
    main([str(DUMP), "--rules", str(RULES), "--out", "-"])

    captured = capsys.readouterr()
    for line in captured.out.splitlines():
        json.loads(line)  # every stdout line is an event
    assert json.loads(captured.err)["frames"] == 300  # summary went to stderr


def test_cli_no_summary_flag(tmp_path, capsys):
    main([str(DUMP), "--rules", str(RULES), "--out", str(tmp_path / "e.jsonl"), "--no-summary"])
    assert capsys.readouterr().out == ""


class _Collector(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _Collector.received.append(json.loads(body))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        return None


@pytest.fixture
def webhook_server():
    _Collector.received = []
    server = HTTPServer(("127.0.0.1", 0), _Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/events"
    server.shutdown()


def test_webhook_sink_posts_every_event_and_never_raises(webhook_server, tmp_path):
    engine = EventEngine(load_rules(RULES), fps=25.0)
    jsonl, hook = JsonlSink(tmp_path / "e.jsonl"), WebhookSink(webhook_server)

    run(engine, DUMP.read_text(encoding="utf-8").splitlines()[:60], MultiSink([jsonl, hook]))

    written = (tmp_path / "e.jsonl").read_text(encoding="utf-8").splitlines()
    assert hook.sent == len(written) > 0 and hook.failed == 0
    assert _Collector.received[0] == json.loads(written[0])

    dead = WebhookSink("http://127.0.0.1:9/nothing", timeout_s=0.2)
    dead.emit(Event("zone_enter", 1, 0.04, 1, "person", "z"))
    assert dead.failed == 1 and dead.sent == 0
