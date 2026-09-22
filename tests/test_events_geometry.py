import pytest

from edgevision.events.geometry import box_anchor, crossing_direction, point_in_polygon, side

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
GATE_A, GATE_B = (5, 0), (5, 10)  # vertical gate, a -> b points down: left = x < 5


def test_box_anchor_modes():
    assert box_anchor(0, 0, 10, 20) == (5, 20)  # feet
    assert box_anchor(0, 0, 10, 20, "center") == (5, 10)
    with pytest.raises(ValueError, match="unknown anchor"):
        box_anchor(0, 0, 1, 1, "top")


def test_point_in_polygon_inside_outside_and_edges():
    assert point_in_polygon((5, 5), SQUARE)
    assert not point_in_polygon((15, 5), SQUARE)
    assert not point_in_polygon((5, -1), SQUARE)
    assert point_in_polygon((0, 5), SQUARE)  # on an edge counts as inside
    assert point_in_polygon((10, 10), SQUARE)  # vertex too


def test_point_in_polygon_concave():
    # L shape: the notch at (7, 7) is outside although inside the bounding box.
    l_shape = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
    assert point_in_polygon((2, 8), l_shape)
    assert not point_in_polygon((7, 7), l_shape)


def test_side_sign_convention():
    assert side(GATE_A, GATE_B, (0, 5)) > 0  # west of a downward gate is "left"
    assert side(GATE_A, GATE_B, (9, 5)) < 0
    assert side(GATE_A, GATE_B, (5, 3)) == 0


def test_crossing_direction_both_ways():
    assert crossing_direction((2, 5), (8, 5), GATE_A, GATE_B) == "left_to_right"
    assert crossing_direction((8, 5), (2, 5), GATE_A, GATE_B) == "right_to_left"


def test_no_crossing_when_staying_on_one_side_or_missing_the_segment():
    assert crossing_direction((2, 5), (4, 6), GATE_A, GATE_B) is None  # same side
    assert crossing_direction((2, 15), (8, 15), GATE_A, GATE_B) is None  # passes beyond b
    assert crossing_direction((2, 5), (2, 5), GATE_A, GATE_B) is None  # no movement


def test_landing_on_the_line_counts_on_the_next_move_only():
    assert crossing_direction((2, 5), (5, 5), GATE_A, GATE_B) is None
    assert (
        crossing_direction((5, 5), (8, 5), GATE_A, GATE_B) is None
    )  # leaving the line: no side change seen
    assert crossing_direction((4.9, 5), (5.1, 5), GATE_A, GATE_B) == "left_to_right"
