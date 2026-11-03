"""Vehicle registry on top of BoT-SORT.

Adds three things the raw tracker does not give you:
  * lost-vehicle relinking  - a new tracker ID that appears where a vehicle was recently lost is
    mapped back to the old vehicle ID, so every physical vehicle keeps ONE ID and counts stay exact.
  * frame-hold buffer       - a vehicle missed for a few frames keeps being drawn (no flicker).
  * box smoothing           - exponential moving average of the box to remove jitter.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from .geometry import bottom_center, center, iou
from .structures import Detection, VehicleState

AppearanceFn = Callable[[np.ndarray, np.ndarray], Optional[np.ndarray]]


def color_histogram(frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
    """Normalised HSV histogram of a crop — cheap appearance signature for relinking."""
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    total = hist.sum()
    return hist / total if total > 0 else None


def hist_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Bhattacharyya coefficient in [0, 1]; 1 = identical. None when either appearance is unknown."""
    if a is None or b is None:
        return None
    return float(np.sum(np.sqrt(a * b)))


class VehicleRegistry:
    def __init__(self, cfg, frame_size: tuple[int, int], appearance_fn: Optional[AppearanceFn] = color_histogram):
        t = cfg.tracker
        r = t.relink
        self.relink_enabled = bool(r.enabled)
        self.max_lost = int(r.max_lost_frames)
        self.max_dist = float(r.max_center_dist) * math.hypot(*frame_size)
        self.min_iou = float(r.min_iou)
        self.min_sim = float(r.min_appearance_sim)
        self.hold_frames = int(t.hold_frames)
        self.alpha = float(t.smoothing_alpha)
        self.appearance_fn = appearance_fn

        self.vehicles: dict[int, VehicleState] = {}
        self.track_to_vid: dict[int, int] = {}
        self.appearance: dict[int, np.ndarray] = {}
        self.velocity: dict[int, np.ndarray] = {}
        self.next_vid = 1
        self.relinked = 0

    # ------------------------------------------------------------------ public
    def update(self, dets: list[Detection], frame: Optional[np.ndarray], frame_idx: int) -> list[VehicleState]:
        seen: set[int] = set()
        for d in dets:
            if d.track_id is None:
                continue
            vid = self.track_to_vid.get(d.track_id)
            if vid is None or vid in seen:
                vid = self._relink(d, frame, frame_idx, exclude=seen)
                self.track_to_vid[d.track_id] = vid
            seen.add(vid)
            self._update_vehicle(vid, d, frame, frame_idx)
        self._forget_old(frame_idx)
        return self.visible(frame_idx)

    def visible(self, frame_idx: int) -> list[VehicleState]:
        """Vehicles seen this frame, plus those still inside the hold buffer."""
        return [v for v in self.vehicles.values() if frame_idx - v.last_seen <= self.hold_frames]

    def active_this_frame(self, frame_idx: int) -> list[VehicleState]:
        return [v for v in self.vehicles.values() if v.last_seen == frame_idx]

    @property
    def total_unique(self) -> int:
        return self.next_vid - 1

    # ----------------------------------------------------------------- private
    def _relink(self, d: Detection, frame, frame_idx: int, exclude: set[int]) -> int:
        if self.relink_enabled:
            app = self.appearance_fn(frame, d.box) if (self.appearance_fn and frame is not None) else None
            best_vid, best_score = None, -1.0
            cx, cy = center(d.box)
            for vid, v in self.vehicles.items():
                lost_for = frame_idx - v.last_seen
                if vid in exclude or lost_for <= 0 or lost_for > self.max_lost:
                    continue
                # predict where the lost vehicle should be now
                vel = self.velocity.get(vid, np.zeros(2))
                px, py = np.array(center(v.box)) + vel * lost_for
                dist = math.hypot(cx - px, cy - py)
                ov = iou(d.box, v.box)
                if dist > self.max_dist and ov < self.min_iou:
                    continue
                sim = hist_similarity(app, self.appearance.get(vid))
                if sim is None:
                    sim = 0.5  # no appearance evidence: decide on motion alone
                elif sim < self.min_sim:
                    continue
                same_kind = 1.0 if self._kind(d.name) == self._kind(v.vehicle_type) else 0.0
                score = (1 - dist / max(self.max_dist, 1e-6)) + ov + sim + 0.5 * same_kind
                if score > best_score:
                    best_vid, best_score = vid, score
            if best_vid is not None:
                self.relinked += 1
                return best_vid
        vid = self.next_vid
        self.next_vid += 1
        return vid

    @staticmethod
    def _kind(name: str) -> str:
        return "two" if name == "Bike" else "four"

    def _update_vehicle(self, vid: int, d: Detection, frame, frame_idx: int) -> None:
        v = self.vehicles.get(vid)
        if v is None:
            v = VehicleState(vid=vid, name=d.name, box=d.box.copy(), conf=d.conf,
                             first_frame=frame_idx, last_seen=frame_idx)
            self.vehicles[vid] = v
        else:
            gap = max(1, frame_idx - v.last_seen)
            old_c = np.array(center(v.box))
            a = self.alpha if gap == 1 else 1.0  # don't smooth across gaps — snap to new position
            v.box = a * d.box + (1 - a) * v.box
            new_v = (np.array(center(v.box)) - old_c) / gap
            self.velocity[vid] = 0.7 * self.velocity.get(vid, new_v) + 0.3 * new_v
            v.conf = d.conf
            v.last_seen = frame_idx
        v.class_votes[d.name] = v.class_votes.get(d.name, 0) + 1
        v.name = v.vehicle_type
        v.trail.append(bottom_center(v.box))
        if len(v.trail) > 40:
            del v.trail[:-40]
        if self.appearance_fn is not None and frame is not None and frame_idx % 5 == 0:
            app = self.appearance_fn(frame, d.box)
            if app is not None:
                prev = self.appearance.get(vid)
                self.appearance[vid] = app if prev is None else 0.8 * prev + 0.2 * app

    def _forget_old(self, frame_idx: int) -> None:
        """Drop tracker-ID mappings for long-gone vehicles (vehicle records are kept for the report)."""
        stale = [t for t, vid in self.track_to_vid.items()
                 if frame_idx - self.vehicles[vid].last_seen > self.max_lost]
        for t in stale:
            del self.track_to_vid[t]
