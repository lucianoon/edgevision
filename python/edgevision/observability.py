"""Prometheus exposition for the app and the events layer, without extra dependencies.

`MetricsExporter.render()` writes the text format (version 0.0.4) from what already exists:
the per-stage `PerformanceMetrics` window, the stream counters and the `EventEngine` summary.
`MetricsServer` serves it on /metrics (plus /healthz) from a daemon thread, so a scrape never
touches the frame loop. Latency values are window statistics (the last N frames), exposed as
gauges with a `stat` label; counters are monotonic totals since the process started.

    edgevision --backend onnx --source rtsp://cam/1 --no-display --metrics-port 9108
    curl -s localhost:9108/metrics | grep edgevision_fps

observability/ has a Prometheus + Grafana stack that scrapes this endpoint.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import TYPE_CHECKING

from edgevision.metrics import PerformanceMetrics

if TYPE_CHECKING:
    from edgevision.events.engine import EventEngine

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
NAMESPACE = "edgevision"


@dataclass
class StreamCounters:
    """Monotonic totals the frame loop updates once per frame."""

    frames: int = 0
    detections: int = 0

    def observe(self, detections: int) -> None:
        self.frames += 1
        self.detections += detections


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{_escape(str(v))}"' for k, v in labels.items()) + "}"


def _fmt(value: float | int) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


class MetricsExporter:
    def __init__(
        self,
        *,
        metrics: PerformanceMetrics | None = None,
        counters: StreamCounters | None = None,
        events: "EventEngine | None" = None,
        labels: Mapping[str, str] | None = None,
        version: str = "0.0.0",
    ):
        self.metrics = metrics
        self.counters = counters
        self.events = events
        self.labels = dict(labels or {})
        self.version = version

    def render(self) -> str:
        out: list[str] = []
        self._metric(out, "build_info", "gauge", "Constant 1 with version and static labels.")
        self._sample(out, "build_info", 1, {"version": self.version, **self.labels})

        if self.metrics is not None:
            self._metric(
                out, "fps", "gauge", "Frames per second from the end-to-end mean of the window."
            )
            self._sample(out, "fps", self.metrics.fps)
            self._metric(out, "frames_total", "counter", "Frames processed since start.")
            self._sample(out, "frames_total", self.metrics.frame_count)
            self._metric(
                out,
                "stage_latency_window_ms",
                "gauge",
                "Per-stage latency statistics over the current window of frames.",
            )
            summary = self.metrics.summary()
            for stage, stats in summary.items():
                for stat in ("mean", "p50", "p95", "max"):
                    self._sample(
                        out,
                        "stage_latency_window_ms",
                        stats[f"{stat}_ms"],
                        {"stage": stage, "stat": stat},
                    )
            self._metric(out, "stage_window_samples", "gauge", "Samples in the window per stage.")
            for stage, stats in summary.items():
                self._sample(out, "stage_window_samples", stats["samples"], {"stage": stage})

        if self.counters is not None:
            self._metric(out, "detections_total", "counter", "Detections emitted since start.")
            self._sample(out, "detections_total", self.counters.detections)

        if self.events is not None:
            self._events(out, self.events.summary())

        return "\n".join(out) + "\n"

    # -- helpers ------------------------------------------------------------------------------

    def _events(self, out: list[str], summary: Mapping) -> None:
        self._metric(
            out, "line_crossings_total", "counter", "Line crossings per line and direction label."
        )
        for line, counts in summary["lines"].items():
            for label, n in counts.items():
                self._sample(out, "line_crossings_total", n, {"line": line, "label": label})
        self._metric(out, "zone_entries_total", "counter", "Tracks that entered the zone.")
        self._metric(out, "zone_occupancy", "gauge", "Tracks currently inside the zone.")
        self._metric(out, "zone_dwell_alerts_total", "counter", "Dwell threshold alerts per zone.")
        for zone, z in summary["zones"].items():
            self._sample(out, "zone_entries_total", z["entries"], {"zone": zone})
            self._sample(out, "zone_occupancy", z["occupancy"], {"zone": zone})
            self._sample(out, "zone_dwell_alerts_total", z["dwell_alerts"], {"zone": zone})
        self._metric(out, "tracks_active", "gauge", "Tracks currently known to the events engine.")
        self._sample(out, "tracks_active", summary["active_tracks"])
        self._metric(out, "events_total", "counter", "Events emitted per type.")
        for event_type, n in summary["events"].items():
            self._sample(out, "events_total", n, {"type": event_type})

    def _metric(self, out: list[str], name: str, kind: str, help_text: str) -> None:
        out.append(f"# HELP {NAMESPACE}_{name} {help_text}")
        out.append(f"# TYPE {NAMESPACE}_{name} {kind}")

    def _sample(
        self, out: list[str], name: str, value: float | int, labels: Mapping[str, str] | None = None
    ) -> None:
        merged = {**self.labels, **(labels or {})} if name != "build_info" else dict(labels or {})
        out.append(f"{NAMESPACE}_{name}{_labels(merged)} {_fmt(value)}")


class MetricsServer:
    """Serves /metrics and /healthz from a daemon thread. Port 0 picks a free port."""

    def __init__(self, render: Callable[[], str], host: str = "0.0.0.0", port: int = 9108):
        self._render = render
        self._server = ThreadingHTTPServer((host, port), self._handler())
        self._thread = Thread(
            target=self._server.serve_forever, name="edgevision-metrics", daemon=True
        )

    def start(self) -> "MetricsServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        host = str(self._server.server_address[0])
        return f"http://{'localhost' if host in ('0.0.0.0', '') else host}:{self.port}/metrics"

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        render = self._render

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # http.server API name
                if self.path.split("?")[0] == "/metrics":
                    self._reply(200, render().encode("utf-8"), CONTENT_TYPE)
                elif self.path == "/healthz":
                    self._reply(200, b"ok\n", "text/plain; charset=utf-8")
                else:
                    self._reply(404, b"not found\n", "text/plain; charset=utf-8")

            def _reply(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                return None

        return Handler


def compose_callbacks(
    *callbacks: Callable[..., bool | None],
) -> Callable[..., bool]:
    """Runs every callback with the same arguments; False only if one of them returned False."""

    def run(*args: object) -> bool:
        keep = True
        for callback in callbacks:
            if callback(*args) is False:
                keep = False
        return keep

    return run


def counting_callback(counters: StreamCounters) -> Callable[[object, Iterable], bool]:
    def observe(_frame: object, detections: Iterable) -> bool:
        counters.observe(len(list(detections)))
        return True

    return observe
