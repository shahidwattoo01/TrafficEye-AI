"""Interactive zone & line editor. Click on the first frame of a video and save a site config.

Keys:
    z  start a detection-zone polygon (click points, ENTER to close)
    c  counting line (2 clicks)
    a / b  speed lines A and B (2 clicks each)
    s  red-light stop line (2 clicks)
    l  traffic-light ROI (2 clicks: top-left, bottom-right)
    u  undo last point     r  reset all
    w  write config        q  quit

    python scripts/draw_zones.py --source road.mp4 --out configs/site_road.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

HELP = "z:zone c:count a/b:speed s:stop l:light u:undo r:reset w:write q:quit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="configs/site.yaml")
    ap.add_argument("--frame", type=int, default=0, help="frame index to draw on")
    a = ap.parse_args()

    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, base = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Cannot read a frame from {a.source}")

    state = {"mode": None, "pts": [], "zones": [], "line": [], "line_a": [], "line_b": [],
             "stop_line": [], "light_roi": []}
    two_point = {"c": "line", "a": "line_a", "b": "line_b", "s": "stop_line", "l": "light_roi"}

    def on_mouse(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN or state["mode"] is None:
            return
        state["pts"].append([x, y])
        mode = state["mode"]
        if mode in two_point and len(state["pts"]) == 2:
            pts = state["pts"]
            state[two_point[mode]] = pts[0] + pts[1] if mode == "l" else pts
            state["mode"], state["pts"] = None, []

    cv2.namedWindow("TrafficEye zones", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("TrafficEye zones", on_mouse)
    while True:
        img = base.copy()
        for z in state["zones"]:
            cv2.polylines(img, [np.array(z, dtype=np.int32)], True, (255, 160, 60), 2)
        for key, color in (("line", (0, 215, 255)), ("line_a", (255, 255, 0)), ("line_b", (255, 255, 0)),
                           ("stop_line", (0, 0, 255))):
            if len(state[key]) == 2:
                cv2.line(img, tuple(state[key][0]), tuple(state[key][1]), color, 2)
        if len(state["light_roi"]) == 4:
            x1, y1, x2, y2 = state["light_roi"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for p in state["pts"]:
            cv2.circle(img, tuple(p), 4, (0, 255, 0), -1)
        cv2.putText(img, f"mode: {state['mode'] or '-'} | {HELP}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("TrafficEye zones", img)
        k = cv2.waitKey(30) & 0xFF
        ch = chr(k) if k < 128 else ""
        if ch == "q" or k == 27:
            break
        if ch == "z" or ch in two_point:
            state["mode"], state["pts"] = ch, []
        elif k in (10, 13) and state["mode"] == "z" and len(state["pts"]) >= 3:
            state["zones"].append(state["pts"])
            state["mode"], state["pts"] = None, []
        elif ch == "u" and state["pts"]:
            state["pts"].pop()
        elif ch == "r":
            for key in ("zones", "line", "line_a", "line_b", "stop_line", "light_roi", "pts"):
                state[key] = []
        elif ch == "w":
            write(state, a.out)
    cv2.destroyAllWindows()


def write(state: dict, out: str) -> None:
    cfg = {"zones": {"detection": [{"name": f"zone_{i + 1}", "points": z} for i, z in enumerate(state["zones"])]}}
    if state["line"]:
        cfg["counting"] = {"enabled": True, "line": state["line"]}
    if state["line_a"] and state["line_b"]:
        cfg["speed"] = {"enabled": True, "line_a": state["line_a"], "line_b": state["line_b"]}
    if state["stop_line"]:
        cfg["red_light"] = {"enabled": bool(state["light_roi"]), "stop_line": state["stop_line"],
                            "light_roi": state["light_roi"]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
