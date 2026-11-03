"""Rendering: annotated boxes, zones, lines and the live analytics side panel."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .structures import VehicleState

# BGR palette
COLORS = {
    "Bike": (255, 170, 0), "Car": (80, 200, 80), "Jeep": (200, 120, 255), "Van": (0, 200, 255),
    "Helmet": (60, 220, 60), "No-Helmet": (40, 40, 230), "Seatbelt": (60, 220, 60), "No-Seatbelt": (40, 40, 230),
}
RED, GREEN, WHITE, GREY, YELLOW = (40, 40, 230), (60, 200, 60), (255, 255, 255), (150, 150, 150), (0, 215, 255)
PANEL_BG, CARD_BG = (28, 24, 22), (48, 42, 38)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, x, y, color, scale=0.5):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), FONT, scale, WHITE, 1, cv2.LINE_AA)


def draw_zones(img, cfg) -> None:
    overlay = img.copy()
    for z in cfg.zones.detection or []:
        pts = np.array(z.get("points") or [], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], (90, 60, 20))
            cv2.polylines(img, [pts], True, (255, 160, 60), 2)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    line = cfg.counting.line
    if cfg.counting.enabled and line and len(line) == 2:
        cv2.line(img, tuple(map(int, line[0])), tuple(map(int, line[1])), YELLOW, 3)
    for key in ("line_a", "line_b"):
        ln = cfg.speed.get(key)
        if cfg.speed.enabled and ln and len(ln) == 2:
            cv2.line(img, tuple(map(int, ln[0])), tuple(map(int, ln[1])), (255, 255, 0), 1)
    sl = cfg.red_light.stop_line
    if cfg.red_light.enabled and sl and len(sl) == 2:
        cv2.line(img, tuple(map(int, sl[0])), tuple(map(int, sl[1])), (0, 0, 255), 2)


def draw_vehicle(img, v: VehicleState, held: bool) -> None:
    violating = bool(v.violations)
    color = RED if violating else COLORS.get(v.vehicle_type, GREEN)
    x1, y1, x2, y2 = v.box.astype(int)
    thickness = 3 if violating else 2
    if held:  # drawn from the hold buffer: dashed look via thinner box
        thickness = 1
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    parts = [f"ID {v.vid}", v.vehicle_type]
    if v.is_two_wheeler and v.helmet != "unknown":
        parts.append("Helmet" if v.helmet == "ok" else "NO HELMET")
    if not v.is_two_wheeler and v.seatbelt != "unknown":
        parts.append("Belt" if v.seatbelt == "ok" else "NO BELT")
    if v.speed_kmh:
        parts.append(f"{v.speed_kmh:.0f} km/h")
    if v.plate:
        parts.append(v.plate)
    _label(img, " | ".join(parts), x1, y1 - 2, color)
    for extra in [x for x in v.violations if x not in ("No-Helmet", "No-Seatbelt")]:
        _label(img, extra.upper(), x1, y2 + 18, RED, 0.45)
    if len(v.trail) > 1:
        pts = np.array(v.trail[-20:], dtype=np.int32)
        cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)


def draw_parts(img, parts) -> None:
    for p in parts:
        c = COLORS.get(p.name, GREY)
        x1, y1, x2, y2 = p.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        _label(img, f"{p.name} {p.conf:.2f}", x1, y1 - 2, c, 0.4)


class DashboardPanel:
    """Side panel with live KPIs, class breakdown, IN/OUT counts and recent violation thumbnails."""

    def __init__(self, width: int, height: int, max_thumbs: int = 4):
        self.w, self.h = width, height
        self.thumbs: deque = deque(maxlen=max_thumbs)

    def add_thumbnail(self, img: np.ndarray, caption: str) -> None:
        if img is None or img.size == 0:
            return
        tw = self.w // 2 - 18
        th = int(tw * 0.75)
        self.thumbs.appendleft((cv2.resize(img, (tw, th)), caption))

    def render(self, stats: dict) -> np.ndarray:
        p = np.full((self.h, self.w, 3), PANEL_BG, dtype=np.uint8)
        cv2.putText(p, "TrafficEye AI", (16, 34), FONT, 0.85, WHITE, 2, cv2.LINE_AA)
        cv2.putText(p, f"{stats['time']}   {stats['fps']:.1f} FPS", (16, 58), FONT, 0.45, GREY, 1, cv2.LINE_AA)

        y = 72
        half = (self.w - 36) // 2
        kpis = [("Vehicles", stats["total_vehicles"], WHITE), ("Violations", stats["total_violations"], RED),
                (stats["in_label"], stats["in"], GREEN), (stats["out_label"], stats["out"], YELLOW)]
        for i, (t, v, c) in enumerate(kpis):
            x = 12 + (i % 2) * (half + 12)
            yy = y + (i // 2) * 60
            cv2.rectangle(p, (x, yy), (x + half, yy + 52), CARD_BG, -1)
            cv2.putText(p, t, (x + 10, yy + 20), FONT, 0.45, GREY, 1, cv2.LINE_AA)
            cv2.putText(p, str(v), (x + 10, yy + 44), FONT, 0.8, c, 2, cv2.LINE_AA)
        y += 128

        cv2.putText(p, "Vehicle classes", (16, y + 8), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        y += 18
        cls = stats["by_class"]
        peak = max(cls.values(), default=1) or 1
        for name in ("Bike", "Car", "Jeep", "Van"):
            n = cls.get(name, 0)
            cv2.putText(p, name, (16, y + 14), FONT, 0.45, GREY, 1, cv2.LINE_AA)
            bw = int((self.w - 140) * n / peak)
            cv2.rectangle(p, (80, y + 3), (80 + max(bw, 2), y + 17), COLORS[name], -1)
            cv2.putText(p, str(n), (self.w - 50, y + 14), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
            y += 22
        y += 6

        cv2.putText(p, "Violations", (16, y + 8), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        y += 18
        for name, n in stats["by_violation"].items():
            cv2.putText(p, name, (16, y + 14), FONT, 0.45, GREY, 1, cv2.LINE_AA)
            cv2.putText(p, str(n), (self.w - 50, y + 14), FONT, 0.45, RED if n else WHITE, 1, cv2.LINE_AA)
            y += 20
        if stats.get("signal"):
            sig = stats["signal"]
            cv2.putText(p, f"Signal: {sig}", (16, y + 14), FONT, 0.45,
                        RED if sig == "RED" else GREEN, 1, cv2.LINE_AA)
            y += 20
        y += 8

        cv2.putText(p, "Recent violations", (16, y + 8), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        y += 16
        for i, (img, cap) in enumerate(self.thumbs):
            x = 12 + (i % 2) * (self.w // 2 - 6)
            yy = y + (i // 2) * (img.shape[0] + 24)
            if yy + img.shape[0] + 20 > self.h:
                break
            p[yy:yy + img.shape[0], x:x + img.shape[1]] = img
            cv2.putText(p, cap, (x, yy + img.shape[0] + 14), FONT, 0.38, WHITE, 1, cv2.LINE_AA)
        return p
