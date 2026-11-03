"""Automatic Number Plate Recognition: YOLO plate detector + EasyOCR, voted across frames."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

from .config import resolve_path
from .geometry import overlap_ratio
from .structures import VehicleState

PLATE_CLEAN = re.compile(r"[^A-Z0-9]")


class PlateReader:
    def __init__(self, cfg):
        a = cfg.anpr
        self.enabled = bool(a.enabled)
        self.min_conf = float(a.min_conf)
        self.reads: dict[int, Counter] = defaultdict(Counter)
        if not self.enabled:
            return
        from ultralytics import YOLO
        import easyocr

        self.detector = YOLO(str(resolve_path(a.plate_weights)))
        self.ocr = easyocr.Reader(list(a.languages), gpu=False if cfg.model.device == "cpu" else True, verbose=False)

    def update(self, frame: np.ndarray, vehicles: list[VehicleState], frame_idx: int, every: int = 3) -> None:
        if not self.enabled or frame_idx % every or not vehicles:
            return
        res = self.detector(frame, conf=self.min_conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return
        for pb in res.boxes.xyxy.cpu().numpy():
            owner = max(vehicles, key=lambda v: overlap_ratio(pb, v.box))
            if overlap_ratio(pb, owner.box) < 0.7:
                continue
            text, conf = self._read(frame, pb)
            if text:
                self.reads[owner.vid][text] += conf
                best, score = self.reads[owner.vid].most_common(1)[0]
                owner.plate, owner.plate_conf = best, score

    def _read(self, frame, box) -> tuple[str, float]:
        import cv2

        x1, y1, x2, y2 = [int(v) for v in box]
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return "", 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        results = self.ocr.readtext(gray, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if not results:
            return "", 0.0
        text = PLATE_CLEAN.sub("", "".join(r[1] for r in results).upper())
        conf = float(np.mean([r[2] for r in results]))
        return (text, conf) if len(text) >= 4 and conf >= self.min_conf else ("", 0.0)
