"""Triple-riding detection: count people on each bike with a COCO person detector."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .association import count_riders
from .config import resolve_path
from .structures import VehicleState


class RiderCounter:
    def __init__(self, cfg, device=None):
        t = cfg.triple_riding
        self.enabled = bool(t.enabled)
        self.max_riders = int(t.max_riders)
        self.conf = float(t.person_conf)
        self.device = device
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))
        if self.enabled:
            from ultralytics import YOLO

            self.model = YOLO(str(resolve_path(t.person_weights)))

    def update(self, frame: np.ndarray, bikes: list[VehicleState]) -> list[VehicleState]:
        """Updates rider counts; returns bikes that are now confirmed as triple-riding."""
        if not self.enabled or not bikes:
            return []
        res = self.model(frame, classes=[0], conf=self.conf, device=self.device, verbose=False)[0]
        persons = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        flagged = []
        for b in bikes:
            n = count_riders(persons, b)
            hist = self.history[b.vid]
            hist.append(n)
            # median over recent frames -> robust to an occluding pedestrian
            b.riders = int(np.median(hist))
            if len(hist) >= 5 and b.riders > self.max_riders:
                flagged.append(b)
        return flagged
