"""Plane geometry for zones (polygons) and counting lines (segments), in source-frame pixels."""

from collections.abc import Sequence

Point = tuple[float, float]

ANCHORS = ("bottom_center", "center")


def box_anchor(x1: float, y1: float, x2: float, y2: float, mode: str = "bottom_center") -> Point:
    """The point of a box that represents the object's position on the ground plane.

    bottom_center (feet of a person, wheels of a car) is what floor zones and gates want;
    center is better for top-down cameras."""
    if mode == "bottom_center":
        return ((x1 + x2) / 2.0, y2)
    if mode == "center":
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    raise ValueError(f"unknown anchor {mode!r}; expected one of {ANCHORS}")


def side(a: Point, b: Point, p: Point) -> float:
    """Twice the signed area of (a, b, p): > 0 when p is left of the directed line a -> b,
    < 0 when right, 0 when on it."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def point_in_polygon(p: Point, polygon: Sequence[Point]) -> bool:
    """Even-odd ray casting. Points exactly on an edge count as inside (stable for gates drawn on
    zone borders)."""
    x, y = p
    inside = False
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        if _on_segment(p, (ax, ay), (bx, by)):
            return True
        if (ay > y) != (by > y):
            x_cross = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < x_cross:
                inside = not inside
    return inside


def _on_segment(p: Point, a: Point, b: Point, eps: float = 1e-9) -> bool:
    if abs(side(a, b, p)) > eps:
        return False
    return min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and (
        min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def crossing_direction(prev: Point, cur: Point, a: Point, b: Point) -> str | None:
    """Did the move prev -> cur cross the segment a -> b? Returns "left_to_right" or
    "right_to_left" (sides relative to the direction a -> b), or None.

    A point landing exactly on the line is not a crossing yet; it counts on the next move that
    leaves the line, so a track hovering on a gate is counted once, not on every frame."""
    s_prev = side(a, b, prev)
    s_cur = side(a, b, cur)
    if s_prev == 0.0 or s_cur == 0.0 or (s_prev > 0) == (s_cur > 0):
        return None
    # The two endpoints of the gate must straddle the movement segment as well.
    t_a = side(prev, cur, a)
    t_b = side(prev, cur, b)
    if (t_a > 0) == (t_b > 0) and t_a != 0.0 and t_b != 0.0:
        return None
    return "left_to_right" if s_prev > 0 else "right_to_left"
