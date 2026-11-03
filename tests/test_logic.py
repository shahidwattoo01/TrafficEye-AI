"""Unit tests for the decision logic (no model or GPU required)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trafficeye.association import PartAssociator, count_riders  # noqa: E402
from trafficeye.config import load_config  # noqa: E402
from trafficeye.geometry import iou, point_in_polygon, segments_intersect  # noqa: E402
from trafficeye.structures import Detection, VehicleState  # noqa: E402
from trafficeye.tracking import VehicleRegistry  # noqa: E402
from trafficeye.voting import VoteBox  # noqa: E402
from trafficeye.zones import LineCounter, SpeedEstimator, ZoneFilter  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()


def det(box, name="Car", tid=1, conf=0.9):
    return Detection(box=np.array(box, dtype=float), cls=0, name=name, conf=conf, track_id=tid)


def vehicle(vid, box, name="Car"):
    return VehicleState(vid=vid, name=name, box=np.array(box, dtype=float), conf=0.9,
                        first_frame=0, last_seen=0, class_votes={name: 5})


# ------------------------------------------------------------------ geometry
def test_geometry_basics():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon((5, 5), square)
    assert not point_in_polygon((15, 5), square)
    assert point_in_polygon((999, 999), [])  # empty polygon = whole frame
    assert segments_intersect((0, -1), (0, 1), (-1, 0), (1, 0))


# -------------------------------------------------------------------- voting
def test_single_bad_frame_never_triggers():
    vb = VoteBox(window=30, min_votes=6, ratio=0.6, min_conf=0.3)
    vb.add(1, "helmet", "violation", 0.9)
    for _ in range(9):
        vb.add(1, "helmet", "ok", 0.8)
    assert vb.status(1, "helmet") == "ok"
    assert not vb.try_latch(1, "helmet")


def test_unknown_until_enough_votes_and_latch_once():
    vb = VoteBox(window=30, min_votes=6, ratio=0.6, min_conf=0.3)
    for _ in range(5):
        vb.add(7, "seatbelt", "violation", 0.8)
    assert vb.status(7, "seatbelt") == "unknown"
    vb.add(7, "seatbelt", "violation", 0.8)
    assert vb.try_latch(7, "seatbelt") is True
    assert vb.try_latch(7, "seatbelt") is False  # reported exactly once
    assert vb.status(7, "seatbelt") == "violation"


def test_low_confidence_votes_ignored():
    vb = VoteBox(window=30, min_votes=2, ratio=0.6, min_conf=0.5)
    for _ in range(10):
        vb.add(1, "helmet", "violation", 0.2)
    assert vb.status(1, "helmet") == "unknown"


# --------------------------------------------------------------- association
def test_helmet_goes_to_bike_seatbelt_to_car(cfg):
    a = PartAssociator(cfg)
    bike = vehicle(1, [100, 200, 160, 320], "Bike")
    car = vehicle(2, [300, 150, 500, 330], "Car")
    parts = [
        det([115, 160, 145, 195], "No-Helmet", None),   # head above the bike box
        det([360, 180, 420, 240], "Seatbelt", None),    # windscreen area of the car
        det([360, 300, 420, 330], "No-Seatbelt", None), # bottom of car -> outside the upper region
    ]
    out = a.associate(parts, [bike, car])
    assert out[1] == [("helmet", "violation", 0.9)]
    assert out[2] == [("seatbelt", "ok", 0.9)]


def test_count_riders():
    bike = vehicle(1, [100, 200, 160, 320], "Bike")
    persons = [[105, 150, 135, 300], [125, 150, 158, 300], [400, 100, 450, 300]]
    assert count_riders(persons, bike) == 2


# ------------------------------------------------------------------ tracking
def test_relink_keeps_one_id(cfg):
    reg = VehicleRegistry(cfg, (1920, 1080), appearance_fn=None)
    for f in range(10):
        reg.update([det([100 + 5 * f, 100, 200 + 5 * f, 200], tid=1)], None, f)
    # tracker loses the car for 5 frames, then re-detects it with a NEW tracker id
    reg.update([det([180, 100, 280, 200], tid=9)], None, 15)
    assert reg.total_unique == 1
    assert reg.relinked == 1


def test_distinct_vehicles_get_distinct_ids(cfg):
    reg = VehicleRegistry(cfg, (1920, 1080), appearance_fn=None)
    reg.update([det([100, 100, 200, 200], tid=1)], None, 0)
    reg.update([det([1500, 800, 1700, 1000], tid=2)], None, 5)
    assert reg.total_unique == 2


def test_hold_buffer_and_smoothing(cfg):
    reg = VehicleRegistry(cfg, (1920, 1080), appearance_fn=None)
    reg.update([det([0, 0, 100, 100], tid=1)], None, 0)
    reg.update([det([10, 0, 110, 100], tid=1)], None, 1)
    v = reg.vehicles[1]
    assert 0 < v.box[0] < 10  # smoothed, not snapped
    assert len(reg.visible(1 + cfg.tracker.hold_frames)) == 1
    assert len(reg.visible(2 + cfg.tracker.hold_frames)) == 0


# --------------------------------------------------------------------- zones
def test_zone_filter_uses_bottom_centre():
    zf = ZoneFilter([{"name": "road", "points": [[0, 500], [1000, 500], [1000, 1000], [0, 1000]]}])
    assert zf.contains(vehicle(1, [100, 300, 200, 600]))
    assert not zf.contains(vehicle(2, [100, 100, 200, 400]))


def test_line_counter_direction_and_once():
    lc = LineCounter([[0, 500], [1000, 500]])
    v = vehicle(1, [100, 300, 200, 480])
    lc.update(v)
    v.box = np.array([100, 340, 200, 520.0])
    d1 = lc.update(v)
    v.box = np.array([100, 300, 200, 480.0])
    lc.update(v)
    v.box = np.array([100, 340, 200, 520.0])
    lc.update(v)  # crossing again must not double count
    assert d1 in ("IN", "OUT")
    assert sum(lc.counts.values()) == 1
    assert v.direction == d1


def test_speed_estimator():
    se = SpeedEstimator([[0, 400], [1000, 400]], [[0, 600], [1000, 600]], distance_m=10.0, fps=25)
    v = vehicle(1, [100, 350, 200, 390])
    speeds = []
    for f in range(0, 40):
        y = 390 + f * 10  # 10 px per frame -> 200 px between lines in 20 frames = 0.8 s
        v.box = np.array([100, y - 40, 200, y], dtype=float)
        s = se.update(v, f)
        if s:
            speeds.append(s)
    assert speeds and speeds[0] == pytest.approx(45.0, rel=0.1)  # 10 m / 0.8 s = 45 km/h


def test_config_overrides():
    cfg = load_config(overrides={"voting": {"min_votes": 2}})
    assert cfg.voting.min_votes == 2
    assert cfg.voting.window == 30
