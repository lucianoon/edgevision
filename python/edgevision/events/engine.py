"""Turns per-frame track observations into events and running counts.

Event types:
  zone_enter      track's anchor moved into a zone            data: {}
  zone_exit       track left a zone, or was lost / stream ended  data: {dwell_s, reason}
  dwell_exceeded  track stayed in a zone >= dwell_seconds (once per stay)  data: {dwell_s}
  line_cross      track's anchor crossed a line               data: {direction, label}

Time is derived from frame numbers and `fps` (the C++ dump carries no timestamps yet).
"""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field

from edgevision.events.geometry import Point, box_anchor, crossing_direction, point_in_polygon
from edgevision.events.rules import Rules

EVENT_TYPES = ("zone_enter", "zone_exit", "dwell_exceeded", "line_cross")


@dataclass(frozen=True)
class TrackObservation:
    """One tracked box in one frame; what the C++ dump and the Python tracker both provide."""

    track_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_dump(cls, item: Mapping) -> "TrackObservation":
        return cls(
            track_id=int(item["id"]),
            class_name=str(item["class_name"]),
            x1=float(item["x1"]),
            y1=float(item["y1"]),
            x2=float(item["x2"]),
            y2=float(item["y2"]),
        )


@dataclass(frozen=True)
class Event:
    type: str
    frame: int
    time_s: float
    track_id: int
    class_name: str
    name: str  # zone or line name
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _TrackState:
    class_name: str
    last_anchor: Point
    last_frame: int
    in_zones: dict[str, int] = field(default_factory=dict)  # zone name -> entry frame
    dwell_fired: set[str] = field(default_factory=set)


class EventEngine:
    def __init__(self, rules: Rules, fps: float = 25.0):
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.rules = rules
        self.fps = fps
        self._states: dict[int, _TrackState] = {}
        self._line_counts: dict[str, dict[str, int]] = {
            ln.name: dict.fromkeys(ln.labels.values(), 0) for ln in rules.lines
        }
        self._zone_entries: dict[str, int] = dict.fromkeys((z.name for z in rules.zones), 0)
        self._dwell_alerts: dict[str, int] = dict.fromkeys((z.name for z in rules.zones), 0)
        self._frames = 0
        self._last_frame = 0
        self._tracks_seen: set[int] = set()
        self._event_counts: dict[str, int] = dict.fromkeys(EVENT_TYPES, 0)

    # -- public ------------------------------------------------------------------------------

    def update(self, frame: int, observations: Iterable[TrackObservation]) -> list[Event]:
        """Feed one frame; returns the events it produced, in a deterministic order."""
        events: list[Event] = []
        self._frames += 1
        self._last_frame = frame
        for obs in sorted(observations, key=lambda o: o.track_id):
            self._tracks_seen.add(obs.track_id)
            anchor = box_anchor(obs.x1, obs.y1, obs.x2, obs.y2, self.rules.anchor)
            state = self._states.get(obs.track_id)
            if state is None:
                state = _TrackState(obs.class_name, anchor, frame)
                self._states[obs.track_id] = state
            else:
                events.extend(self._lines(obs, state, anchor, frame))
            events.extend(self._zones(obs, state, anchor, frame))
            state.last_anchor = anchor
            state.last_frame = frame
        events.extend(self._expire(frame))
        self._count(events)
        return events

    def close(self) -> list[Event]:
        """End of stream: every track still inside a zone leaves it (reason end_of_stream)."""
        events: list[Event] = []
        for track_id in sorted(self._states):
            events.extend(self._leave_all(track_id, self._last_frame, "end_of_stream"))
        self._states.clear()
        self._count(events)
        return events

    @property
    def active_tracks(self) -> int:
        return len(self._states)

    @property
    def event_counts(self) -> dict[str, int]:
        return dict(self._event_counts)

    def _count(self, events: list[Event]) -> None:
        for event in events:
            self._event_counts[event.type] += 1

    def summary(self) -> dict:
        occupancy = {
            z.name: sum(1 for s in self._states.values() if z.name in s.in_zones)
            for z in self.rules.zones
        }
        return {
            "frames": self._frames,
            "duration_s": round(self._frames / self.fps, 3),
            "tracks_seen": len(self._tracks_seen),
            "active_tracks": self.active_tracks,
            "events": self.event_counts,
            "lines": {name: dict(counts) for name, counts in self._line_counts.items()},
            "zones": {
                z.name: {
                    "entries": self._zone_entries[z.name],
                    "occupancy": occupancy[z.name],
                    "dwell_alerts": self._dwell_alerts[z.name],
                }
                for z in self.rules.zones
            },
        }

    # -- rules ---------------------------------------------------------------------------------

    def _time(self, frame: int) -> float:
        return round(frame / self.fps, 3)

    def _lines(
        self, obs: TrackObservation, state: _TrackState, anchor: Point, frame: int
    ) -> list[Event]:
        events = []
        for line in self.rules.lines:
            if not line.applies_to(obs.class_name):
                continue
            direction = crossing_direction(state.last_anchor, anchor, line.a, line.b)
            if direction is None:
                continue
            label = line.labels[direction]
            self._line_counts[line.name][label] += 1
            events.append(
                Event(
                    "line_cross",
                    frame,
                    self._time(frame),
                    obs.track_id,
                    obs.class_name,
                    line.name,
                    {"direction": direction, "label": label},
                )
            )
        return events

    def _zones(
        self, obs: TrackObservation, state: _TrackState, anchor: Point, frame: int
    ) -> list[Event]:
        events = []
        for zone in self.rules.zones:
            if not zone.applies_to(obs.class_name):
                continue
            inside = point_in_polygon(anchor, zone.polygon)
            entry = state.in_zones.get(zone.name)
            if inside and entry is None:
                state.in_zones[zone.name] = frame
                self._zone_entries[zone.name] += 1
                events.append(
                    Event(
                        "zone_enter",
                        frame,
                        self._time(frame),
                        obs.track_id,
                        obs.class_name,
                        zone.name,
                    )
                )
            elif not inside and entry is not None:
                events.append(self._exit(obs.track_id, state, zone.name, frame, "left"))
            elif inside and zone.dwell_seconds is not None and zone.name not in state.dwell_fired:
                dwell = (frame - entry) / self.fps  # type: ignore[operator]  # entry is int here
                if dwell >= zone.dwell_seconds:
                    state.dwell_fired.add(zone.name)
                    self._dwell_alerts[zone.name] += 1
                    events.append(
                        Event(
                            "dwell_exceeded",
                            frame,
                            self._time(frame),
                            obs.track_id,
                            obs.class_name,
                            zone.name,
                            {"dwell_s": round(dwell, 3)},
                        )
                    )
        return events

    def _exit(self, track_id: int, state: _TrackState, zone: str, frame: int, reason: str) -> Event:
        entry = state.in_zones.pop(zone)
        state.dwell_fired.discard(zone)
        return Event(
            "zone_exit",
            frame,
            self._time(frame),
            track_id,
            state.class_name,
            zone,
            {"dwell_s": round((frame - entry) / self.fps, 3), "reason": reason},
        )

    def _leave_all(self, track_id: int, frame: int, reason: str) -> list[Event]:
        state = self._states[track_id]
        return [self._exit(track_id, state, zone, frame, reason) for zone in sorted(state.in_zones)]

    def _expire(self, frame: int) -> list[Event]:
        events: list[Event] = []
        for track_id in sorted(self._states):
            state = self._states[track_id]
            if frame - state.last_frame >= self.rules.lost_after_frames:
                events.extend(self._leave_all(track_id, state.last_frame, "lost"))
                del self._states[track_id]
        return events
