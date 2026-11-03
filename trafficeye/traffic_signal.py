"""Red-light violation: traffic-signal state from a colour ROI + stop-line crossing."""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .zones import LineCounter
from .structures import VehicleState


class SignalStateDetector:
    """Classifies the signal head inside a fixed ROI as RED / GREEN / UNKNOWN using HSV masks.

    The state is smoothed over a few frames so a single glare frame does not flip it.
    """

    def __init__(self, roi: list, red_ratio: float = 0.08, smooth: int = 5):
        self.roi = [int(v) for v in roi] if roi and len(roi) == 4 else None
        self.red_ratio = red_ratio
        self.history: deque = deque(maxlen=smooth)

    def classify(self, frame: np.ndarray) -> str:
        if self.roi is None:
            return "UNKNOWN"
        import cv2

        x1, y1, x2, y2 = self.roi
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "UNKNOWN"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        bright = (hsv[..., 2] > 150) & (hsv[..., 1] > 90)
        hue = hsv[..., 0]
        red = bright & ((hue < 10) | (hue > 165))
        green = bright & (hue > 40) & (hue < 95)
        n = crop.shape[0] * crop.shape[1]
        r, g = red.sum() / n, green.sum() / n
        state = "RED" if r >= self.red_ratio and r > g else ("GREEN" if g >= self.red_ratio else "UNKNOWN")
        self.history.append(state)
        return max(set(self.history), key=list(self.history).count)


class RedLightMonitor:
    def __init__(self, cfg):
        rl = cfg.red_light
        self.enabled = bool(rl.enabled) and bool(rl.stop_line)
        self.signal = SignalStateDetector(rl.light_roi, float(rl.red_ratio))
        self.stop_line = LineCounter(rl.stop_line, assign_direction=False)
        self.state = "UNKNOWN"

    def update_signal(self, frame: np.ndarray) -> str:
        if self.enabled:
            self.state = self.signal.classify(frame)
        return self.state

    def check(self, v: VehicleState) -> Optional[str]:
        """Returns 'Red-Light' when the vehicle crosses the stop line while the signal is RED."""
        if not self.enabled:
            return None
        crossed = self.stop_line.update(v)
        return "Red-Light" if crossed and self.state == "RED" else None
