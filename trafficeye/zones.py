"""Road zones, directional IN/OUT counting, wrong-way and speed estimation."""
from __future__ import annotations

from typing import Optional

from .geometry import bottom_center, point_in_polygon, segments_intersect, side_of_line
from .structures import VehicleState


class ZoneFilter:
    def __init__(self, zones: list[dict]):
        self.zones = [z for z in (zones or []) if len(z.get("points") or []) >= 3]

    def contains(self, v: VehicleState) -> bool:
        if not self.zones:
            return True
        pt = bottom_center(v.box)
        return any(point_in_polygon(pt, z["points"]) for z in self.zones)

    def zone_name(self, v: VehicleState) -> Optional[str]:
        pt = bottom_center(v.box)
        for z in self.zones:
            if point_in_polygon(pt, z["points"]):
                return z.get("name")
        return None if self.zones else "frame"


class LineCounter:
    """Counts vehicles crossing a line, split by direction. Each vehicle is counted at most once."""

    def __init__(self, line: list, in_label: str = "IN", out_label: str = "OUT", assign_direction: bool = True):
        self.assign_direction = assign_direction
        self.line = line if line and len(line) == 2 else None
        self.in_label, self.out_label = in_label, out_label
        self.counts = {in_label: 0, out_label: 0}
        self.by_class: dict[str, dict[str, int]] = {}
        self.crossed: set[int] = set()
        self._last: dict[int, tuple[float, float]] = {}

    def update(self, v: VehicleState) -> Optional[str]:
        """Returns the direction label on the frame the vehicle crosses, else None."""
        if self.line is None:
            return None
        pt = bottom_center(v.box)
        prev = self._last.get(v.vid)
        self._last[v.vid] = pt
        if prev is None or v.vid in self.crossed:
            return None
        p1, p2 = self.line
        if not segments_intersect(prev, pt, p1, p2):
            return None
        # moving from the negative (right) side of p1->p2 to the positive (left) side = IN
        direction = self.in_label if side_of_line(prev, p1, p2) < 0 else self.out_label
        self.crossed.add(v.vid)
        self.counts[direction] += 1
        cls = self.by_class.setdefault(v.vehicle_type, {self.in_label: 0, self.out_label: 0})
        cls[direction] += 1
        if self.assign_direction:
            v.direction = direction
        return direction


class SpeedEstimator:
    """Average speed between two lines a known distance apart (time-of-flight method)."""

    def __init__(self, line_a: list, line_b: list, distance_m: float, fps: float):
        self.ok = bool(line_a and line_b and len(line_a) == 2 and len(line_b) == 2 and fps > 0)
        self.a = LineCounter(line_a, assign_direction=False)
        self.b = LineCounter(line_b, assign_direction=False)
        self.distance_m = distance_m
        self.fps = fps
        self.t_a: dict[int, int] = {}
        self.t_b: dict[int, int] = {}

    def update(self, v: VehicleState, frame_idx: int) -> Optional[float]:
        if not self.ok:
            return None
        if self.a.update(v):
            self.t_a[v.vid] = frame_idx
        if self.b.update(v):
            self.t_b[v.vid] = frame_idx
        if v.vid in self.t_a and v.vid in self.t_b and v.speed_kmh is None:
            frames = abs(self.t_b[v.vid] - self.t_a[v.vid])
            if frames > 0:
                v.speed_kmh = self.distance_m / (frames / self.fps) * 3.6
                return v.speed_kmh
        return None
