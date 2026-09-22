"""EventEngine on synthetic tracks: known trajectories -> known events and counts."""

import pytest

from edgevision.events import EventEngine, TrackObservation, parse_rules

FPS = 10.0
RULES = parse_rules(
    {
        "anchor": "bottom_center",
        "lost_after_frames": 5,
        "zones": [
            {
                "name": "shop",
                "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
                "dwell_seconds": 1.0,
            },
            {
                "name": "cars_only",
                "polygon": [[0, 300], [50, 300], [50, 350], [0, 350]],
                "classes": ["car"],
            },
        ],
        "lines": [{"name": "door", "a": [100, 0], "b": [100, 400], "classes": ["person"]}],
    }
)


def person(track_id: int, x: float, y: float, w: float = 20, h: float = 40) -> TrackObservation:
    # anchor (bottom_center) lands exactly on (x, y)
    return TrackObservation(track_id, "person", x - w / 2, y - h, x + w / 2, y)


def car(track_id: int, x: float, y: float) -> TrackObservation:
    return TrackObservation(track_id, "car", x - 10, y - 10, x + 10, y)


def types(events):
    return [(e.type, e.track_id, e.name) for e in events]


def test_walk_through_the_door_into_the_shop_and_back():
    engine = EventEngine(RULES, fps=FPS)
    engine.update(1, [person(1, 50, 150)])  # west of the door, outside the shop
    events = engine.update(2, [person(1, 150, 150)])  # crossed x=100 and is inside the shop

    assert types(events) == [("line_cross", 1, "door"), ("zone_enter", 1, "shop")]
    assert events[0].data == {"direction": "left_to_right", "label": "in"}
    assert events[0].time_s == 0.2

    events = engine.update(3, [person(1, 50, 150)])
    assert types(events) == [("line_cross", 1, "door"), ("zone_exit", 1, "shop")]
    assert events[0].data["label"] == "out"
    assert events[1].data == {"dwell_s": 0.1, "reason": "left"}

    summary = engine.summary()
    assert summary["lines"]["door"] == {"in": 1, "out": 1}
    assert summary["zones"]["shop"] == {"entries": 1, "occupancy": 0, "dwell_alerts": 0}


def test_dwell_alert_fires_once_per_stay():
    engine = EventEngine(RULES, fps=FPS)
    fired = []
    for frame in range(1, 30):  # 2.9 s inside the shop
        fired += [
            e for e in engine.update(frame, [person(7, 150, 150)]) if e.type == "dwell_exceeded"
        ]

    assert len(fired) == 1
    assert fired[0].frame == 11 and fired[0].data == {"dwell_s": 1.0}  # entered at frame 1, 10 fps
    assert engine.summary()["zones"]["shop"]["dwell_alerts"] == 1

    engine.update(30, [person(7, 50, 150)])  # leave
    engine.update(31, [person(7, 150, 150)])  # re-enter: a new stay can fire again
    events = [e for f in range(32, 45) for e in engine.update(f, [person(7, 150, 150)])]
    assert [e.type for e in events] == ["dwell_exceeded"]


def test_class_filters():
    engine = EventEngine(RULES, fps=FPS)
    engine.update(1, [car(2, 50, 150), car(3, 25, 325)])
    events = engine.update(
        2, [car(2, 150, 150), car(3, 25, 325)]
    )  # car crosses the persons-only door

    assert [e.type for e in events] == ["zone_enter"]  # shop has no class filter: car enters it
    assert events[0].name == "shop" and events[0].class_name == "car"
    # cars_only accepted the car on frame 1 already
    assert engine.summary()["zones"]["cars_only"]["entries"] == 1
    assert engine.summary()["lines"]["door"] == {"in": 0, "out": 0}

    engine.update(3, [person(4, 25, 325)])
    assert engine.summary()["zones"]["cars_only"]["entries"] == 1  # person ignored by cars_only


def test_lost_track_leaves_its_zones_with_reason_lost():
    engine = EventEngine(RULES, fps=FPS)
    engine.update(1, [person(9, 150, 150)])
    quiet = [engine.update(f, []) for f in range(2, 6)]
    assert all(not ev for ev in quiet)  # 4 frames unseen: still tracked (lost_after_frames=5)

    events = engine.update(6, [])
    assert types(events) == [("zone_exit", 9, "shop")]
    assert events[0].frame == 1 and events[0].data["reason"] == "lost"
    assert engine.summary()["zones"]["shop"]["occupancy"] == 0


def test_close_flushes_open_stays():
    engine = EventEngine(RULES, fps=FPS)
    engine.update(1, [person(1, 150, 150), person(2, 150, 150)])

    events = engine.close()

    assert types(events) == [("zone_exit", 1, "shop"), ("zone_exit", 2, "shop")]
    assert {e.data["reason"] for e in events} == {"end_of_stream"}
    assert engine.close() == []


def test_first_observation_never_counts_a_crossing():
    engine = EventEngine(RULES, fps=FPS)
    events = engine.update(1, [person(1, 150, 150)])  # appears east of the door
    assert [e.type for e in events] == ["zone_enter"]


def test_events_are_serialisable_and_ordered_by_track_id():
    engine = EventEngine(RULES, fps=FPS)
    engine.update(1, [person(5, 50, 150), person(2, 50, 150)])
    events = engine.update(2, [person(5, 150, 150), person(2, 150, 150)])

    assert [e.track_id for e in events] == [2, 2, 5, 5]
    assert set(events[0].to_dict()) == {
        "type",
        "frame",
        "time_s",
        "track_id",
        "class_name",
        "name",
        "data",
    }


def test_invalid_fps_rejected():
    with pytest.raises(ValueError, match="fps"):
        EventEngine(RULES, fps=0)
