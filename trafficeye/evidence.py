"""Evidence and reporting: violation snapshots, CSV reports, e-challans and a run summary."""
from __future__ import annotations

import csv
import html
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .geometry import clip_box, expand
from .structures import VehicleState, ViolationEvent

FINES = {  # indicative amounts, edit for your jurisdiction
    "No-Helmet": 1000,
    "No-Seatbelt": 1000,
    "Wrong-Way": 5000,
    "Over-Speed": 2000,
    "Red-Light": 5000,
    "Triple-Riding": 1000,
}


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


class EvidenceStore:
    def __init__(self, out_dir: Path, cfg, source_name: str, start_time: datetime | None = None):
        self.out_dir = Path(out_dir)
        self.snap_dir = self.out_dir / "snapshots"
        self.challan_dir = self.out_dir / "challans"
        self.cfg = cfg
        self.source_name = source_name
        self.start_time = start_time or datetime.now()
        self.events: list[ViolationEvent] = []
        o = cfg.output
        self.save_snapshots = bool(o.save_snapshots)
        self.save_challans = bool(o.save_challans)
        self.padding = float(o.snapshot_padding)
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        if self.save_challans:
            self.challan_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- snapshots
    def snapshot(self, frame: np.ndarray, ev: ViolationEvent) -> str | None:
        if not self.save_snapshots:
            return None
        import cv2

        h, w = frame.shape[:2]
        crop_box = clip_box(expand(ev.box, top=self.padding + (0.6 if ev.vehicle_type == "Bike" else 0),
                                   side=self.padding, bottom=self.padding), w, h).astype(int)
        x1, y1, x2, y2 = crop_box
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None
        # stamp the evidence: violation, ID, time
        banner = np.zeros((44, crop.shape[1], 3), dtype=np.uint8)
        banner[:] = (30, 30, 200)
        text = f"{ev.violation} | ID {ev.vid} | {fmt_time(ev.timestamp_s)}"
        cv2.putText(banner, text, (6, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        img = np.vstack([banner, crop])
        name = f"{ev.violation}_ID{ev.vid:04d}_F{ev.frame:06d}.jpg"
        path = self.snap_dir / name
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        # full-frame context image as well
        ctx = frame.copy()
        bx = ev.box.astype(int)
        cv2.rectangle(ctx, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 3)
        cv2.imwrite(str(self.snap_dir / name.replace(".jpg", "_context.jpg")), ctx, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return str(path)

    def record(self, frame: np.ndarray, ev: ViolationEvent, vehicle: VehicleState) -> ViolationEvent:
        ev.snapshot = self.snapshot(frame, ev)
        vehicle.violations.append(ev.violation)
        if ev.snapshot:
            vehicle.snapshots[ev.violation] = ev.snapshot
        self.events.append(ev)
        return ev

    def wall_time(self, seconds: float) -> datetime:
        return self.start_time + timedelta(seconds=seconds)

    # --------------------------------------------------------------- reports
    def write_vehicle_report(self, vehicles: list[VehicleState], fps: float) -> Path:
        path = self.out_dir / "vehicle_report.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["vehicle_id", "vehicle_type", "first_seen", "last_seen", "frames_tracked",
                        "direction", "helmet", "seatbelt", "riders", "speed_kmh", "plate",
                        "violations", "snapshots"])
            for v in sorted(vehicles, key=lambda x: x.vid):
                two = v.is_two_wheeler
                w.writerow([
                    v.vid, v.vehicle_type,
                    fmt_time(v.first_frame / fps), fmt_time(v.last_seen / fps),
                    v.last_seen - v.first_frame + 1,
                    v.direction or "",
                    v.helmet if two else "n/a",
                    "n/a" if two else v.seatbelt,
                    v.riders if two else "",
                    f"{v.speed_kmh:.1f}" if v.speed_kmh else "",
                    v.plate or "",
                    ";".join(v.violations),
                    ";".join(Path(p).name for p in v.snapshots.values()),
                ])
        return path

    def write_violation_log(self) -> Path:
        path = self.out_dir / "violations.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["event_id", "vehicle_id", "vehicle_type", "violation", "frame", "video_time",
                        "wall_time", "confidence", "plate", "snapshot", "details"])
            for i, e in enumerate(self.events, 1):
                w.writerow([i, e.vid, e.vehicle_type, e.violation, e.frame, fmt_time(e.timestamp_s),
                            self.wall_time(e.timestamp_s).isoformat(timespec="seconds"),
                            f"{e.confidence:.2f}", e.plate or "", Path(e.snapshot).name if e.snapshot else "",
                            json.dumps(e.details) if e.details else ""])
        return path

    def write_challans(self) -> list[Path]:
        if not self.save_challans:
            return []
        paths = []
        for e in self.events:
            cid = f"TE-{self.start_time:%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"
            data = {
                "challan_id": cid,
                "issued_at": datetime.now().isoformat(timespec="seconds"),
                "offence_time": self.wall_time(e.timestamp_s).isoformat(timespec="seconds"),
                "location": self.source_name,
                "vehicle_id": e.vid,
                "vehicle_type": e.vehicle_type,
                "plate": e.plate or "UNREAD",
                "violation": e.violation,
                "fine_amount": FINES.get(e.violation, 0),
                "confidence": round(e.confidence, 3),
                "evidence": Path(e.snapshot).name if e.snapshot else None,
                "details": e.details,
                "status": "PENDING_REVIEW",  # a human officer must verify before issuing
            }
            jpath = self.challan_dir / f"{cid}.json"
            jpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
            hpath = self.challan_dir / f"{cid}.html"
            hpath.write_text(self._challan_html(data, e), encoding="utf-8")
            paths.append(hpath)
        return paths

    def _challan_html(self, d: dict, e: ViolationEvent) -> str:
        img = f'<img src="../snapshots/{html.escape(d["evidence"])}" alt="evidence">' if d["evidence"] else ""
        rows = "".join(
            f"<tr><th>{html.escape(k.replace('_', ' ').title())}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in d.items() if k not in ("evidence", "details")
        )
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>e-Challan {d['challan_id']}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:720px;margin:24px auto;color:#1b1f24}}
h1{{font-size:20px;border-bottom:3px solid #c62828;padding-bottom:8px}}table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid #e3e6ea}}th{{width:38%;color:#555}}
img{{max-width:100%;margin-top:16px;border:1px solid #ccc}}.note{{color:#777;font-size:12px;margin-top:16px}}</style>
</head><body><h1>TrafficEye AI — Electronic Challan (Draft)</h1><table>{rows}</table>{img}
<p class="note">Generated automatically. Requires verification by an authorised officer before issue.</p></body></html>"""

    def write_summary(self, summary: dict) -> Path:
        path = self.out_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return path
