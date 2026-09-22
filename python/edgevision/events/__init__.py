"""Events over tracks: zones, line counting, dwell time and alerts, driven by a YAML rule file.

The runtime (C++ or Python) produces tracks; this package turns them into things a business
acts on: "person entered zone A", "3 people crossed the gate inwards", "someone stayed 30 s at
the counter". It consumes the JSONL the C++ runtime writes with --dump-tracks (one object per
frame) and is independent of the GPU code, so it is fully tested with synthetic tracks.
"""

from edgevision.events.engine import Event, EventEngine, TrackObservation
from edgevision.events.rules import Line, Rules, Zone, load_rules, parse_rules

__all__ = [
    "Event",
    "EventEngine",
    "Line",
    "Rules",
    "TrackObservation",
    "Zone",
    "load_rules",
    "parse_rules",
]
