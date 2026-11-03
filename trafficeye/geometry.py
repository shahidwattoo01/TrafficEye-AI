"""Small, dependency-free (numpy only) geometry helpers."""
from __future__ import annotations

from typing import Sequence

import numpy as np

Box = np.ndarray  # [x1, y1, x2, y2]


def area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def center(b: Sequence[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def bottom_center(b: Sequence[float]) -> tuple[float, float]:
    """Ground-contact point of a vehicle; best anchor for zones and lines."""
    return (b[0] + b[2]) / 2.0, float(b[3])


def expand(b: Sequence[float], top: float = 0.0, side: float = 0.0, bottom: float = 0.0) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] - side * w, b[1] - top * h, b[2] + side * w, b[3] + bottom * h], dtype=float)


def clip_box(b: Sequence[float], width: int, height: int) -> np.ndarray:
    return np.array(
        [min(max(b[0], 0), width - 1), min(max(b[1], 0), height - 1),
         min(max(b[2], 0), width - 1), min(max(b[3], 0), height - 1)],
        dtype=float,
    )


def point_in_polygon(pt: tuple[float, float], poly: Sequence[Sequence[float]]) -> bool:
    """Ray casting. An empty polygon means 'everywhere'."""
    if len(poly) < 3:
        return True
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def side_of_line(pt: tuple[float, float], p1: Sequence[float], p2: Sequence[float]) -> float:
    """>0 left of p1->p2, <0 right, 0 on the line (2D cross product)."""
    return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])


def segments_intersect(a1, a2, b1, b2) -> bool:
    d1 = side_of_line(a1, b1, b2)
    d2 = side_of_line(a2, b1, b2)
    d3 = side_of_line(b1, a1, a2)
    d4 = side_of_line(b2, a1, a2)
    # a point exactly on a line counts as being on its positive side, so touching = crossing once
    return ((d1 < 0) != (d2 < 0)) and ((d3 < 0) != (d4 < 0))
