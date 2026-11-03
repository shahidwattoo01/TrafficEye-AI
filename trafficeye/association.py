"""Link helmet / seatbelt detections to the vehicle they belong to.

Helmets are matched to bikes (search region expanded upward, because the rider's head sits above
the bike box). Seatbelts are matched to cars / jeeps / vans (upper part of the box, where the
windscreen and driver are). Each part goes to the single best vehicle so it is never double-counted.
"""
from __future__ import annotations

from .geometry import center, expand, overlap_ratio
from .structures import Detection, VehicleState

HELMET_CLASSES = {"Helmet": "ok", "No-Helmet": "violation"}
SEATBELT_CLASSES = {"Seatbelt": "ok", "No-Seatbelt": "violation"}


class PartAssociator:
    def __init__(self, cfg):
        a = cfg.association
        self.min_overlap = float(a.min_overlap)
        self.bike_top = float(a.bike_expand_top)
        self.bike_side = float(a.bike_expand_side)
        self.car_upper = float(a.car_upper_fraction)
        self.two_wheelers = set(cfg.classes.two_wheelers)
        self.four_wheelers = set(cfg.classes.four_wheelers)

    def _region(self, v: VehicleState):
        b = v.box
        if v.vehicle_type in self.two_wheelers:
            return expand(b, top=self.bike_top, side=self.bike_side)
        h = b[3] - b[1]
        return [b[0], b[1], b[2], b[1] + self.car_upper * h]

    def associate(self, parts: list[Detection], vehicles: list[VehicleState]) -> dict[int, list[tuple[str, str, float]]]:
        """Return {vid: [(kind, label, conf), ...]} where kind is 'helmet' or 'seatbelt'."""
        out: dict[int, list[tuple[str, str, float]]] = {}
        for p in parts:
            if p.name in HELMET_CLASSES:
                kind, label, allowed = "helmet", HELMET_CLASSES[p.name], self.two_wheelers
            elif p.name in SEATBELT_CLASSES:
                kind, label, allowed = "seatbelt", SEATBELT_CLASSES[p.name], self.four_wheelers
            else:
                continue
            best, best_score = None, 0.0
            pcx, pcy = center(p.box)
            for v in vehicles:
                if v.vehicle_type not in allowed:
                    continue
                region = self._region(v)
                ov = overlap_ratio(p.box, region)
                if ov < self.min_overlap:
                    continue
                # tie-break: prefer the vehicle whose centre-x is closest to the part
                vcx = (region[0] + region[2]) / 2
                width = max(region[2] - region[0], 1)
                score = ov + (1 - min(abs(pcx - vcx) / width, 1))
                if score > best_score:
                    best, best_score = v, score
            if best is not None:
                out.setdefault(best.vid, []).append((kind, label, p.conf))
        return out


def count_riders(person_boxes, bike: VehicleState, min_overlap: float = 0.4) -> int:
    """Number of person boxes that sit on a bike (used for triple-riding detection)."""
    region = expand(bike.box, top=0.8, side=0.1)
    return sum(1 for pb in person_boxes if overlap_ratio(pb, region) >= min_overlap)
