"""End-to-end pipeline: raw video in -> tracked, analysed dashboard video + evidence + reports out."""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .anpr import PlateReader
from .association import HELMET_CLASSES, SEATBELT_CLASSES, PartAssociator
from .config import Config, resolve_path
from .dashboard import DashboardPanel, draw_parts, draw_vehicle, draw_zones
from .detector import Detector
from .evidence import EvidenceStore, fmt_time
from .riders import RiderCounter
from .structures import VehicleState, ViolationEvent
from .tracking import VehicleRegistry
from .traffic_signal import RedLightMonitor
from .voting import VoteBox
from .zones import LineCounter, SpeedEstimator, ZoneFilter

VIOLATION_TYPES = ["No-Helmet", "No-Seatbelt", "Wrong-Way", "Over-Speed", "Red-Light", "Triple-Riding"]
MIN_FRAMES_TO_COUNT = 3  # ignore one- or two-frame ghost tracks in the totals

ProgressFn = Callable[[int, int, np.ndarray, dict], Optional[bool]]


def open_source(source: str) -> tuple[cv2.VideoCapture, bool]:
    """Video file, webcam index ("0") or network stream (rtsp://, http://)."""
    is_live = source.isdigit() or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {source}")
    return cap, is_live


class TrafficEyePipeline:
    def __init__(self, cfg: Config, source: str, out_dir: str | Path | None = None,
                 max_frames: int | None = None, progress: ProgressFn | None = None):
        self.cfg = cfg
        self.source = str(source)
        self.max_frames = max_frames
        self.progress = progress

        self.cap, self.is_live = open_source(self.source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self.fps <= 1 or self.fps > 240:
            self.fps = 25.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = 0 if self.is_live else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        stem = "live" if self.is_live else Path(self.source).stem
        run_name = f"{stem}_{datetime.now():%Y%m%d_%H%M%S}"
        self.out_dir = Path(out_dir) if out_dir else resolve_path(cfg.output.dir) / run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.detector = Detector(cfg)
        self.registry = VehicleRegistry(cfg, (self.w, self.h))
        self.associator = PartAssociator(cfg)
        self.votes = VoteBox.from_config(cfg)
        self.zone = ZoneFilter(cfg.zones.detection)
        c = cfg.counting
        self.counter = LineCounter(c.line if c.enabled else [], c.in_label, c.out_label)
        s = cfg.speed
        self.speed = SpeedEstimator(s.line_a if s.enabled else [], s.line_b, float(s.distance_m), self.fps)
        self.red_light = RedLightMonitor(cfg)
        self.riders = RiderCounter(cfg, device=self.detector.device)
        self.plates = PlateReader(cfg)
        self.evidence = EvidenceStore(self.out_dir, cfg, source_name=Path(self.source).name)
        self.panel = DashboardPanel(int(cfg.output.dashboard_width), self.h) if cfg.output.dashboard else None
        self.vehicle_names = set(cfg.classes.vehicles)

        self.writer = None
        if cfg.output.save_video:
            out_w = self.w + (self.panel.w if self.panel else 0)
            self.video_path = self.out_dir / "annotated.mp4"
            self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.fps, (out_w, self.h))
        self._fps_times: deque = deque(maxlen=30)
        self.frame_idx = -1

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        t0 = time.time()
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                self.frame_idx += 1
                out = self.process(frame)
                if self.writer is not None:
                    self.writer.write(out)
                if self.cfg.output.show:
                    cv2.imshow("TrafficEye AI", out)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                if self.progress and self.progress(self.frame_idx, self.total_frames, out, self.stats()) is False:
                    break
                if self.max_frames and self.frame_idx + 1 >= self.max_frames:
                    break
        finally:
            self.cap.release()
            if self.writer is not None:
                self.writer.release()
            if self.cfg.output.show:
                cv2.destroyAllWindows()
        return self.finalize(time.time() - t0)

    def video_time(self) -> float:
        return self.frame_idx / self.fps

    # -------------------------------------------------------------- per frame
    def process(self, frame: np.ndarray) -> np.ndarray:
        idx = self.frame_idx
        dets = self.detector.track(frame)
        vehicle_dets = [d for d in dets if d.name in self.vehicle_names]
        part_dets = [d for d in dets if d.name in HELMET_CLASSES or d.name in SEATBELT_CLASSES]

        self.registry.update(vehicle_dets, frame, idx)
        active = self.registry.active_this_frame(idx)
        for v in active:
            v.in_zone = self.zone.contains(v)
        judged = [v for v in active if v.in_zone]

        # helmet / seatbelt -> vehicle, then multi-frame vote
        for vid, obs in self.associator.associate(part_dets, judged).items():
            for kind, label, conf in obs:
                self.votes.add(vid, kind, label, conf)
        for v in judged:
            kind = "helmet" if v.is_two_wheeler else "seatbelt"
            status = self.votes.status(v.vid, kind)
            if v.is_two_wheeler:
                v.helmet = status
            else:
                v.seatbelt = status
            if self.votes.try_latch(v.vid, kind):
                self._violation(frame, v, "No-Helmet" if kind == "helmet" else "No-Seatbelt",
                                self.votes.score(v.vid, kind))

        signal = self.red_light.update_signal(frame)
        for v in active:
            direction = self.counter.update(v)
            if (direction and self.cfg.wrong_way.enabled
                    and direction != self.cfg.wrong_way.allowed_direction):
                self._violation(frame, v, "Wrong-Way", 1.0, {"direction": direction})
            kmh = self.speed.update(v, idx)
            if kmh and kmh > float(self.cfg.speed.limit_kmh):
                self._violation(frame, v, "Over-Speed", 1.0,
                                {"speed_kmh": round(kmh, 1), "limit_kmh": self.cfg.speed.limit_kmh})
            if self.red_light.check(v):
                self._violation(frame, v, "Red-Light", 1.0, {"signal": signal})

        if idx % 2 == 0:
            for b in self.riders.update(frame, [v for v in judged if v.is_two_wheeler]):
                self._violation(frame, b, "Triple-Riding", 1.0, {"riders": b.riders})
        self.plates.update(frame, judged, idx)

        return self.render(frame, part_dets, signal)

    def _violation(self, frame, v: VehicleState, kind: str, conf: float, details: dict | None = None) -> None:
        if kind in v.violations:
            return
        ev = ViolationEvent(vid=v.vid, vehicle_type=v.vehicle_type, violation=kind, frame=self.frame_idx,
                            timestamp_s=self.video_time(), confidence=conf, box=v.box.copy(),
                            plate=v.plate, details=details or {})
        self.evidence.record(frame, ev, v)
        if self.panel is not None and ev.snapshot:
            self.panel.add_thumbnail(cv2.imread(ev.snapshot), f"ID {v.vid} {kind}")

    # ------------------------------------------------------------ rendering
    def render(self, frame: np.ndarray, parts, signal: str) -> np.ndarray:
        img = frame.copy()
        draw_zones(img, self.cfg)
        draw_parts(img, parts)
        for v in self.registry.visible(self.frame_idx):
            draw_vehicle(img, v, held=v.last_seen != self.frame_idx)
        now = time.time()
        self._fps_times.append(now)
        if self.panel is None:
            return img
        stats = self.stats()
        stats["signal"] = signal if self.red_light.enabled else None
        return np.hstack([img, self.panel.render(stats)])

    def counted_vehicles(self) -> list[VehicleState]:
        return [v for v in self.registry.vehicles.values() if sum(v.class_votes.values()) >= MIN_FRAMES_TO_COUNT]

    def stats(self) -> dict:
        vehicles = self.counted_vehicles()
        by_class: dict[str, int] = {}
        for v in vehicles:
            by_class[v.vehicle_type] = by_class.get(v.vehicle_type, 0) + 1
        by_violation = {k: 0 for k in VIOLATION_TYPES if self._violation_enabled(k)}
        for e in self.evidence.events:
            by_violation[e.violation] = by_violation.get(e.violation, 0) + 1
        ft = self._fps_times
        fps = (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 1 and ft[-1] > ft[0] else 0.0
        return {
            "frame": self.frame_idx, "time": fmt_time(max(self.video_time(), 0)), "fps": fps,
            "total_vehicles": len(vehicles), "total_violations": len(self.evidence.events),
            "in": self.counter.counts[self.counter.in_label], "out": self.counter.counts[self.counter.out_label],
            "in_label": self.counter.in_label, "out_label": self.counter.out_label,
            "by_class": by_class, "by_violation": by_violation,
        }

    def _violation_enabled(self, kind: str) -> bool:
        return {
            "Wrong-Way": self.cfg.wrong_way.enabled, "Over-Speed": self.cfg.speed.enabled,
            "Red-Light": self.cfg.red_light.enabled, "Triple-Riding": self.cfg.triple_riding.enabled,
        }.get(kind, True)

    # ------------------------------------------------------------- finalize
    def finalize(self, elapsed: float) -> dict:
        # plates are often read after the violation fired — back-fill them
        for e in self.evidence.events:
            e.plate = e.plate or self.registry.vehicles[e.vid].plate
        vehicles = self.counted_vehicles()
        outputs = {}
        if self.cfg.output.save_csv:
            outputs["vehicle_report"] = str(self.evidence.write_vehicle_report(vehicles, self.fps))
            outputs["violation_log"] = str(self.evidence.write_violation_log())
        challans = self.evidence.write_challans()
        if challans:
            outputs["challans"] = str(self.evidence.challan_dir)
        if self.writer is not None:
            outputs["video"] = str(self.video_path)
        outputs["snapshots"] = str(self.evidence.snap_dir)

        s = self.stats()
        frames = self.frame_idx + 1
        summary = {
            "source": self.source,
            "frames_processed": frames,
            "video_duration": fmt_time(frames / self.fps),
            "processing_seconds": round(elapsed, 2),
            "avg_fps": round(frames / elapsed, 2) if elapsed > 0 else 0,
            "total_vehicles": s["total_vehicles"],
            "relinked_ids": self.registry.relinked,
            "vehicles_by_class": s["by_class"],
            "counts": self.counter.counts,
            "counts_by_class": self.counter.by_class,
            "total_violations": s["total_violations"],
            "violations_by_type": s["by_violation"],
            "outputs": outputs,
        }
        outputs["summary"] = str(self.evidence.write_summary(summary))
        return summary
