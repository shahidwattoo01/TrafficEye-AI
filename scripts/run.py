"""Run TrafficEye AI on a video file, webcam or RTSP stream.

Examples:
    python scripts/run.py --source data/videos/road.mp4
    python scripts/run.py --source road.mp4 --config configs/site_junction.yaml --show
    python scripts/run.py --source rtsp://user:pass@192.168.1.20/stream1 --max-frames 9000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trafficeye.config import load_config  # noqa: E402
from trafficeye.pipeline import TrafficEyePipeline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="TrafficEye AI — traffic violation monitoring")
    p.add_argument("--source", required=True, help="video path, webcam index (0) or rtsp/http URL")
    p.add_argument("--config", default=None, help="site config YAML (merged over configs/default.yaml)")
    p.add_argument("--weights", default=None, help="override model weights")
    p.add_argument("--out", default=None, help="output directory (default: runs/<video>_<timestamp>)")
    p.add_argument("--conf", type=float, default=None, help="override confidence threshold")
    p.add_argument("--device", default=None, help='"cpu", "0", "cuda:0"')
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--show", action="store_true", help="live preview window")
    p.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    p.add_argument("--enable", nargs="*", default=[],
                   choices=["wrong_way", "speed", "red_light", "triple_riding", "anpr"],
                   help="turn on optional modules")
    return p.parse_args()


def main():
    a = parse_args()
    overrides: dict = {"model": {}, "output": {}}
    if a.weights:
        overrides["model"]["weights"] = a.weights
    if a.conf is not None:
        overrides["model"]["conf"] = a.conf
    if a.device is not None:
        overrides["model"]["device"] = a.device
    if a.show:
        overrides["output"]["show"] = True
    if a.no_video:
        overrides["output"]["save_video"] = False
    for module in a.enable:
        overrides[module] = {"enabled": True}
    cfg = load_config(a.config, overrides)

    def progress(i, total, _frame, stats):
        if i % 50 == 0:
            pct = f"{100 * (i + 1) / total:5.1f}%" if total else f"frame {i}"
            print(f"\r{pct} | vehicles {stats['total_vehicles']} | violations {stats['total_violations']}"
                  f" | {stats['fps']:.1f} FPS", end="", flush=True)

    summary = TrafficEyePipeline(cfg, a.source, a.out, a.max_frames, progress).run()
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
