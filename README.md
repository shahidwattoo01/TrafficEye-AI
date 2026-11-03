# 🚦 TrafficEye AI

**Real-time helmet & seatbelt violation monitoring from road video with YOLO26, BoT-SORT + Re-ID and multi-frame voting.**

Raw traffic video goes in. Out come an annotated dashboard video, timestamped violation snapshots, per-vehicle and per-violation CSV reports, draft e-challans and a JSON run summary.

```
Video / RTSP ─► YOLO26 (8 classes) ─► BoT-SORT + Re-ID ─► ID relinking + anti-flicker
            ─► zone filter ─► helmet/seatbelt ↔ vehicle association ─► multi-frame voting
            ─► violations (+ wrong-way, speed, red-light, triple riding, ANPR)
            ─► evidence snapshots · CSV reports · e-challans · live dashboard
```

## Features

| Area | What it does |
|---|---|
| **Detection** | Custom-trained YOLO26 model with 8 classes: Bike, Car, Jeep, Van, Helmet, No-Helmet, Seatbelt, No-Seatbelt. Global and per-class confidence thresholds. |
| **Tracking** | BoT-SORT with Re-ID gives persistent vehicle IDs. |
| **ID relinking** | When the tracker gives a vehicle a new ID after losing it, the new ID is matched back to the old one by predicted motion, IoU and colour-histogram appearance. Each physical vehicle keeps one ID, so counts stay exact. |
| **Anti-flicker** | A frame-hold buffer keeps drawing briefly missed vehicles, and EMA smoothing keeps boxes steady. |
| **Association** | Helmets are linked to bikes (search region extended above the bike for the rider's head). Seatbelts are linked to the upper part of cars, jeeps and vans. Each part goes to exactly one vehicle. |
| **Multi-frame voting** | Each check uses confidence-weighted votes over a sliding window. It needs a minimum number of votes and a minimum violation ratio. A confirmed violation is latched and reported once. |
| **Zones** | Polygon road zones; vehicles are judged only inside them, using the bottom-centre point of the box. |
| **Counting** | IN/OUT counting across a line, split by direction and class, with each vehicle counted at most once. |
| **Wrong-way** | Flags vehicles that cross the counting line against the allowed direction. |
| **Speed** | Time-of-flight between two calibrated lines, with an over-speed violation. |
| **Red-light** | Reads the signal state from a traffic-light ROI (HSV, smoothed) and flags vehicles crossing the stop line on red. |
| **Triple riding** | A COCO person detector counts riders per bike (median over recent frames). |
| **ANPR** | YOLO plate detector + EasyOCR. Plate reads are voted across frames and filled into reports. |
| **Evidence** | Stamped crop and full-frame context image for every violation. |
| **Reports** | `vehicle_report.csv`, `violations.csv`, `summary.json` |
| **e-Challans** | A draft challan per violation (JSON + printable HTML with evidence), status `PENDING_REVIEW`. |
| **Dashboard** | Side panel on the output video: totals, IN/OUT, class bars, violation counts, signal state, FPS, recent snapshots. |
| **Web panel** | Streamlit app: upload a video or enter an RTSP URL, tune thresholds, toggle modules, watch live, browse evidence, download reports, review challans. |
| **Sources** | Video files, webcams and RTSP/HTTP streams. |
| **Training** | Kaggle GPU notebook and script with CCTV-oriented augmentation; evaluation script for mAP, precision, recall and FPS. |

## Project structure

```
trafficeye/            core package
  detector.py          YOLO26 + BoT-SORT wrapper (per-class thresholds, class aliases)
  tracking.py          vehicle registry: relinking, hold buffer, smoothing
  association.py       helmet/seatbelt → vehicle, rider counting
  voting.py            multi-frame voting + latching
  zones.py             zones, line counter, speed estimator
  traffic_signal.py    signal state + red-light monitor
  riders.py            triple-riding detector
  anpr.py              number plate recognition
  evidence.py          snapshots, CSV reports, e-challans, summary
  dashboard.py         drawing + analytics panel
  pipeline.py          end-to-end orchestration
scripts/run.py         CLI
scripts/draw_zones.py  interactive zone / line / ROI editor
app/streamlit_app.py   web control panel
training/              data.yaml, train.py, evaluate.py, kaggle_train.ipynb
configs/               default.yaml, botsort_reid.yaml, coco_demo.yaml
tests/                 unit tests for the decision logic
```

## Quick start

```bash
pip install -r requirements.txt
```

1. **Train** with `training/kaggle_train.ipynb` on Kaggle, or locally:
   ```bash
   python training/train.py --data training/data.yaml --model yolo26s.pt --epochs 120
   ```
   Put the result in `weights/trafficeye_yolo26.pt`.
2. **Evaluate** to get the numbers to report:
   ```bash
   python training/evaluate.py --weights weights/trafficeye_yolo26.pt --split test
   ```
3. **Draw the site layout** (zones, counting line, speed lines, stop line, light ROI):
   ```bash
   python scripts/draw_zones.py --source data/videos/road.mp4 --out configs/site_road.yaml
   ```
4. **Run:**
   ```bash
   python scripts/run.py --source data/videos/road.mp4 --config configs/site_road.yaml
   python scripts/run.py --source road.mp4 --enable wrong_way speed anpr --show
   python scripts/run.py --source rtsp://camera/stream --max-frames 9000
   ```
5. **Web panel:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

No custom weights yet? `--config configs/coco_demo.yaml` runs the pipeline with a stock COCO YOLO26 model. Tracking, zones, counting, speed and the dashboard all work in this mode; the helmet and seatbelt checks don't, because COCO has no helmet or seatbelt classes.

## Outputs (`runs/<video>_<timestamp>/`)

| File | Contents |
|---|---|
| `annotated.mp4` | Tracked video with the dashboard panel |
| `snapshots/` | `<Violation>_ID<id>_F<frame>.jpg` crops + `_context.jpg` full frames |
| `vehicle_report.csv` | One row per vehicle: type, first/last seen, direction, helmet, seatbelt, riders, speed, plate, violations |
| `violations.csv` | One row per violation event with video time, wall time, confidence, plate, snapshot |
| `challans/` | Draft e-challans (`.json` + `.html`) |
| `summary.json` | Totals, class breakdown, IN/OUT, violations by type, processing FPS |

## Configuration

All settings are in `configs/default.yaml`. A site config only needs the keys it changes, because it is deep-merged over the defaults. The main tuning settings:

- `voting.min_votes` / `voting.violation_ratio`: raise them for fewer false positives, lower them for faster decisions.
- `tracker.relink.*`: how aggressively a lost vehicle is relinked.
- `tracker.hold_frames`, `tracker.smoothing_alpha`: how stable the boxes look.
- `association.*`: where helmets and seatbelts are searched for relative to the vehicle box.

## Tests

```bash
python -m pytest -q tests
```

## Notes

- Fine amounts in `evidence.py` are placeholders; edit them for your jurisdiction.
- Challans are drafts. An authorised officer must verify each one before it is issued.
- Speed accuracy depends on correct line calibration (`speed.distance_m`).
