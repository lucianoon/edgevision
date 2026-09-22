"""Apply a rule file to a track dump and emit events.

    edgevision-events benchmarks/results/tracks_cpp_512.jsonl --rules configs/rules.example.yaml \\
        --fps 25 --out events.jsonl

`tracks` is the JSONL the C++ runtime writes with --dump-tracks ("-" reads stdin, so the runtime
can be piped in live). The summary (counts per line label, zone entries/occupancy/dwell alerts)
is printed as JSON at the end.
"""

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO

from edgevision.events.engine import EventEngine, TrackObservation
from edgevision.events.rules import load_rules
from edgevision.events.sinks import JsonlSink, MultiSink, Sink, WebhookSink
from edgevision.observability import MetricsExporter, MetricsServer


def read_dump(lines: Iterable[str]) -> Iterator[tuple[int, list[TrackObservation]]]:
    """(frame, observations) per non-empty line of a --dump-tracks file."""
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        yield int(row["frame"]), [TrackObservation.from_dump(t) for t in row["tracks"]]


def run(engine: EventEngine, dump: Iterable[str], sink: Sink) -> dict:
    """Feed every frame of `dump` to `engine`, send events to `sink`; returns the summary."""
    try:
        for frame, observations in read_dump(dump):
            for event in engine.update(frame, observations):
                sink.emit(event)
        for event in engine.close():
            sink.emit(event)
    finally:
        sink.close()
    return engine.summary()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EdgeVision events: rules over tracks")
    parser.add_argument(
        "tracks", help="track dump (JSONL from edgevision_trt --dump-tracks) or - for stdin"
    )
    parser.add_argument(
        "--rules", type=Path, required=True, help="YAML rule file (see configs/rules.example.yaml)"
    )
    parser.add_argument(
        "--fps", type=float, default=25.0, help="frame rate of the source (frames -> seconds)"
    )
    parser.add_argument("--out", default="-", help="events JSONL path, or - for stdout (default)")
    parser.add_argument("--webhook", help="also POST every event to this URL")
    parser.add_argument(
        "--no-summary", action="store_true", help="do not print the summary at the end"
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        help="serve Prometheus metrics (/metrics, /healthz) on this port while running; "
        "0 = any free port",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    engine = EventEngine(load_rules(args.rules), fps=args.fps)
    sinks: list[Sink] = [JsonlSink(args.out)]
    if args.webhook:
        sinks.append(WebhookSink(args.webhook))

    source: IO[str]
    if args.tracks == "-":
        source = sys.stdin
    else:
        source = open(args.tracks, encoding="utf-8")
    server = None
    if args.metrics_port is not None:
        exporter = MetricsExporter(events=engine, labels={"source": args.tracks})
        server = MetricsServer(exporter.render, port=args.metrics_port).start()
        print(f"metrics: {server.url}", file=sys.stderr)
    try:
        with source:
            summary = run(engine, source, MultiSink(sinks))
    finally:
        if server is not None:
            server.stop()

    if not args.no_summary:
        out = (
            sys.stderr if args.out == "-" else sys.stdout
        )  # keep stdout pure JSONL when events go there
        print(json.dumps(summary, indent=2), file=out)


if __name__ == "__main__":
    main()
