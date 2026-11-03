"""Evaluate trained weights and write a metrics table (mAP@50, mAP@50-95, precision, recall).

    python training/evaluate.py --weights weights/trafficeye_yolo26.pt --data training/data.yaml --split test
Also benchmarks inference speed so you can quote real FPS numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/trafficeye_yolo26.pt")
    ap.add_argument("--data", default="training/data.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default="runs/eval")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.weights)
    m = model.val(data=a.data, split=a.split, imgsz=a.imgsz, device=a.device or None,
                  project=a.out, name="metrics", plots=True)

    names = model.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = {"precision": round(p, 4), "recall": round(r, 4),
                                    "mAP50": round(ap50, 4), "mAP50-95": round(ap, 4)}
    speed = m.speed  # ms per image: preprocess / inference / postprocess
    total_ms = sum(speed.values())
    report = {
        "overall": {"precision": round(m.box.mp, 4), "recall": round(m.box.mr, 4),
                    "mAP50": round(m.box.map50, 4), "mAP50-95": round(m.box.map, 4)},
        "per_class": per_class,
        "speed_ms": {k: round(v, 2) for k, v in speed.items()},
        "est_fps": round(1000 / total_ms, 1) if total_ms else None,
    }
    out = Path(a.out) / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'class':<14}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}")
    for n, r in {"all": report["overall"], **per_class}.items():
        print(f"{n:<14}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['mAP50']:>9.3f}{r['mAP50-95']:>10.3f}")
    print(f"\nInference ~{report['est_fps']} FPS  |  saved {out}")


if __name__ == "__main__":
    main()
