from pathlib import Path

import pytest

from edgevision.events import load_rules, parse_rules
from edgevision.events.rules import DEFAULT_LABELS


def test_example_rules_file_parses():
    rules = load_rules(Path("configs/rules.example.yaml"))

    assert [z.name for z in rules.zones] == ["platform"]
    assert rules.zones[0].dwell_seconds == 5.0 and rules.zones[0].classes == {"person"}
    assert [ln.name for ln in rules.lines] == ["gate"]
    assert rules.lines[0].labels == {"left_to_right": "east", "right_to_left": "west"}
    assert rules.anchor == "bottom_center" and rules.lost_after_frames == 30


def test_defaults_and_empty_file():
    rules = parse_rules({})
    assert rules.zones == () and rules.lines == ()
    assert rules.anchor == "bottom_center" and rules.lost_after_frames == 30


def test_line_defaults_and_single_class_string():
    rules = parse_rules({"lines": [{"name": "l", "a": [0, 0], "b": [0, 1], "classes": "person"}]})
    assert rules.lines[0].labels == DEFAULT_LABELS
    assert rules.lines[0].classes == {"person"}
    assert rules.lines[0].applies_to("person") and not rules.lines[0].applies_to("car")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"anchor": "top"}, "anchor must be"),
        ({"lost_after_frames": 0}, "lost_after_frames"),
        ({"zones": [{"polygon": [[0, 0], [1, 0], [1, 1]]}]}, "name is required"),
        ({"zones": [{"name": "z", "polygon": [[0, 0], [1, 0]]}]}, "at least 3 points"),
        ({"zones": [{"name": "z", "polygon": [[0, 0], [1, 0], [1]]}]}, r"a point is \[x, y\]"),
        (
            {"zones": [{"name": "z", "polygon": [[0, 0], [1, 0], [1, 1]], "dwell_seconds": 0}]},
            "dwell_seconds",
        ),
        (
            {"zones": [{"name": "z", "polygon": [[0, 0], [1, 0], [1, 1]], "classes": []}]},
            "classes must not be empty",
        ),
        ({"lines": [{"name": "l", "a": [0, 0]}]}, "needs points a and b"),
        ({"lines": [{"name": "l", "a": [0, 0], "b": [0, 0]}]}, "must differ"),
        (
            {"lines": [{"name": "l", "a": [0, 0], "b": [1, 1], "labels": {"up": "x"}}]},
            "label keys are",
        ),
        (
            {
                "zones": [{"name": "same", "polygon": [[0, 0], [1, 0], [1, 1]]}],
                "lines": [{"name": "same", "a": [0, 0], "b": [1, 1]}],
            },
            "must be unique",
        ),
    ],
)
def test_validation_errors(data, message):
    with pytest.raises(ValueError, match=message):
        parse_rules(data)
