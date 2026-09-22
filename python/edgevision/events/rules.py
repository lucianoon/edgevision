"""Rule file: which zones and lines exist, for which classes, with which thresholds.

anchor: bottom_center          # bottom_center | center (point of the box used for geometry)
lost_after_frames: 30          # a track unseen for this long leaves every zone it was in
zones:
  - name: platform
    polygon: [[0, 700], [1920, 700], [1920, 1080], [0, 1080]]
    classes: [person]          # omit for all classes
    dwell_seconds: 5           # omit for no dwell alert
lines:
  - name: gate
    a: [960, 0]
    b: [960, 1080]
    labels: {left_to_right: in, right_to_left: out}   # sides relative to a -> b; default shown
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from edgevision.events.geometry import ANCHORS, Point

DEFAULT_LABELS = {"left_to_right": "in", "right_to_left": "out"}


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: tuple[Point, ...]
    classes: frozenset[str] | None = None  # None: every class
    dwell_seconds: float | None = None

    def applies_to(self, class_name: str) -> bool:
        return self.classes is None or class_name in self.classes


@dataclass(frozen=True)
class Line:
    name: str
    a: Point
    b: Point
    classes: frozenset[str] | None = None
    labels: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_LABELS))

    def applies_to(self, class_name: str) -> bool:
        return self.classes is None or class_name in self.classes


@dataclass(frozen=True)
class Rules:
    zones: tuple[Zone, ...] = ()
    lines: tuple[Line, ...] = ()
    anchor: str = "bottom_center"
    lost_after_frames: int = 30


def load_rules(path: str | Path) -> Rules:
    with open(path, encoding="utf-8") as f:
        return parse_rules(yaml.safe_load(f) or {})


def parse_rules(data: Mapping) -> Rules:
    anchor = str(data.get("anchor", "bottom_center"))
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
    lost = int(data.get("lost_after_frames", 30))
    if lost < 1:
        raise ValueError("lost_after_frames must be >= 1")

    zones = tuple(_parse_zone(z, i) for i, z in enumerate(data.get("zones") or []))
    lines = tuple(_parse_line(ln, i) for i, ln in enumerate(data.get("lines") or []))
    names = [z.name for z in zones] + [ln.name for ln in lines]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"zone/line names must be unique, duplicated: {sorted(duplicates)}")
    return Rules(zones=zones, lines=lines, anchor=anchor, lost_after_frames=lost)


def _parse_zone(z: Mapping, index: int) -> Zone:
    name = _name(z, f"zones[{index}]")
    polygon = tuple(_point(p, f"zone {name!r} polygon") for p in z.get("polygon") or [])
    if len(polygon) < 3:
        raise ValueError(f"zone {name!r}: polygon needs at least 3 points")
    dwell = z.get("dwell_seconds")
    if dwell is not None and float(dwell) <= 0:
        raise ValueError(f"zone {name!r}: dwell_seconds must be > 0")
    return Zone(
        name=name,
        polygon=polygon,
        classes=_classes(z.get("classes")),
        dwell_seconds=None if dwell is None else float(dwell),
    )


def _parse_line(ln: Mapping, index: int) -> Line:
    name = _name(ln, f"lines[{index}]")
    if "a" not in ln or "b" not in ln:
        raise ValueError(f"line {name!r}: needs points a and b")
    a, b = _point(ln["a"], f"line {name!r} a"), _point(ln["b"], f"line {name!r} b")
    if a == b:
        raise ValueError(f"line {name!r}: a and b must differ")
    labels = dict(DEFAULT_LABELS)
    for key, value in (ln.get("labels") or {}).items():
        if key not in DEFAULT_LABELS:
            raise ValueError(f"line {name!r}: label keys are {sorted(DEFAULT_LABELS)}, got {key!r}")
        labels[key] = str(value)
    return Line(name=name, a=a, b=b, classes=_classes(ln.get("classes")), labels=labels)


def _name(item: Mapping, where: str) -> str:
    name = item.get("name")
    if not name or not isinstance(name, str):
        raise ValueError(f"{where}: name is required")
    return name


def _point(value: Sequence, where: str) -> Point:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ValueError(f"{where}: a point is [x, y], got {value!r}")
    return (float(value[0]), float(value[1]))


def _classes(value) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    classes = frozenset(str(c) for c in value)
    if not classes:
        raise ValueError("classes must not be empty (omit it for all classes)")
    return classes
