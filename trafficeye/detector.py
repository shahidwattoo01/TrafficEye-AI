"""YOLO26 detection + BoT-SORT/Re-ID tracking wrapper around Ultralytics."""
from __future__ import annotations

import numpy as np

from .config import Config, resolve_path
from .structures import Detection


class Detector:
    def __init__(self, cfg: Config):
        from ultralytics import YOLO  # imported lazily so the logic modules stay testable without it

        m = cfg.model
        self.model = YOLO(str(resolve_path(m.weights)))
        self.imgsz = m.imgsz
        self.conf = m.conf
        self.iou = m.iou
        self.device = m.device or None
        self.half = bool(m.half)
        self.tracker_cfg = str(resolve_path(cfg.tracker.config))

        names = cfg.classes.names
        model_names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        # Prefer the names baked into the weights; fall back to the config.
        self.names = {int(k): str(v) for k, v in model_names.items()} if model_names else dict(enumerate(names))
        # optional renaming of model class names, e.g. to run a COCO model as a demo fallback
        aliases = cfg.classes.get("aliases") or {}
        self.names = {k: aliases.get(v, v) for k, v in self.names.items()}
        self.class_conf = {str(k): float(v) for k, v in (m.get("class_conf") or {}).items()}

        # Run at the lowest per-class threshold, then filter each class on its own threshold.
        self.run_conf = min([self.conf, *self.class_conf.values()]) if self.class_conf else self.conf

    def track(self, frame: np.ndarray) -> list[Detection]:
        kwargs = {"half": True} if self.half else {}  # only pass when requested (deprecated flag in newer releases)
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_cfg,
            imgsz=self.imgsz,
            conf=self.run_conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            **kwargs,
        )
        return self._parse(results[0])

    def _parse(self, result) -> list[Detection]:
        dets: list[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)
        for b, c, s, t in zip(xyxy, cls, conf, ids):
            name = self.names.get(int(c), str(c))
            if s < self.class_conf.get(name, self.conf):
                continue
            dets.append(Detection(box=b.astype(float), cls=int(c), name=name, conf=float(s),
                                  track_id=None if t is None else int(t)))
        return dets
