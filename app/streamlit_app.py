"""TrafficEye AI web control panel.

    streamlit run app/streamlit_app.py

Upload a video (or enter an RTSP URL), tune thresholds and modules, watch the analysed stream live,
then browse violation evidence, download reports and review draft e-challans.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trafficeye.config import load_config  # noqa: E402
from trafficeye.pipeline import TrafficEyePipeline  # noqa: E402

st.set_page_config(page_title="TrafficEye AI", page_icon="🚦", layout="wide")
st.title("🚦 TrafficEye AI")
st.caption("Helmet & seatbelt violation monitoring — YOLO26 · BoT-SORT + Re-ID · multi-frame voting")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Source")
    mode = st.radio("Input", ["Upload video", "Stream URL / webcam"], horizontal=True)
    upload = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"]) if mode == "Upload video" else None
    url = st.text_input("RTSP / HTTP URL or webcam index", "0") if mode != "Upload video" else ""
    site_cfg = st.text_input("Site config YAML (optional)", "")

    st.header("Model")
    weights = st.text_input("Weights", "weights/trafficeye_yolo26.pt")
    conf = st.slider("Confidence", 0.05, 0.95, 0.35, 0.05)
    imgsz = st.select_slider("Image size", [640, 768, 960, 1280], 960)
    device = st.selectbox("Device", ["auto", "cpu", "0"])

    st.header("Decision logic")
    min_votes = st.slider("Min votes before decision", 1, 30, 6)
    ratio = st.slider("Violation vote ratio", 0.3, 1.0, 0.6, 0.05)
    hold = st.slider("Anti-flicker hold (frames)", 0, 30, 8)

    st.header("Modules")
    m_wrong = st.checkbox("Wrong-way detection")
    m_speed = st.checkbox("Speed estimation")
    m_red = st.checkbox("Red-light violation")
    m_triple = st.checkbox("Triple riding")
    m_anpr = st.checkbox("Number plate recognition (ANPR)")
    max_frames = st.number_input("Max frames (0 = all)", 0, 10_000_000, 0, step=500)
    run = st.button("▶ Run analysis", type="primary", use_container_width=True)

if "summary" not in st.session_state:
    st.session_state.summary = None

# --------------------------------------------------------------------- run
if run:
    if mode == "Upload video" and upload is None:
        st.error("Upload a video first.")
        st.stop()
    if upload is not None:
        tmp = Path(tempfile.mkdtemp()) / upload.name
        tmp.write_bytes(upload.getbuffer())
        source = str(tmp)
    else:
        source = url.strip()

    overrides = {
        "model": {"weights": weights, "conf": conf, "imgsz": imgsz, "device": "" if device == "auto" else device},
        "voting": {"min_votes": min_votes, "violation_ratio": ratio},
        "tracker": {"hold_frames": hold},
        "wrong_way": {"enabled": m_wrong}, "speed": {"enabled": m_speed}, "red_light": {"enabled": m_red},
        "triple_riding": {"enabled": m_triple}, "anpr": {"enabled": m_anpr},
    }
    cfg = load_config(site_cfg or None, overrides)

    frame_box = st.empty()
    bar = st.progress(0.0, text="Starting…")
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [c.empty() for c in (k1, k2, k3, k4, k5)]
    stop = st.button("⏹ Stop")

    def progress(i, total, frame, s):
        if i % 3 == 0:
            frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            kpis[0].metric("Vehicles", s["total_vehicles"])
            kpis[1].metric("Violations", s["total_violations"])
            kpis[2].metric(s["in_label"], s["in"])
            kpis[3].metric(s["out_label"], s["out"])
            kpis[4].metric("FPS", f"{s['fps']:.1f}")
        if total:
            bar.progress(min((i + 1) / total, 1.0), text=f"Frame {i + 1}/{total}")
        return not stop

    with st.spinner("Analysing video…"):
        try:
            pipe = TrafficEyePipeline(cfg, source, max_frames=max_frames or None, progress=progress)
            st.session_state.summary = pipe.run()
        except Exception as exc:  # surface model/source errors in the UI
            st.exception(exc)
            st.stop()
    bar.progress(1.0, text="Done")

# ----------------------------------------------------------------- results
summary = st.session_state.summary
if summary:
    out = summary["outputs"]
    run_dir = Path(out["summary"]).parent
    st.success(f"Processed {summary['frames_processed']} frames in {summary['processing_seconds']} s "
               f"({summary['avg_fps']} FPS). Results: {run_dir}")

    c = st.columns(4)
    c[0].metric("Unique vehicles", summary["total_vehicles"])
    c[1].metric("Violations", summary["total_violations"])
    c[2].metric("IDs relinked", summary["relinked_ids"])
    c[3].metric("Duration", summary["video_duration"])

    tab_overview, tab_evidence, tab_vehicles, tab_challans, tab_raw = st.tabs(
        ["Overview", "Evidence", "Vehicles", "e-Challans", "Summary JSON"])

    with tab_overview:
        a, b = st.columns(2)
        a.subheader("Vehicles by class")
        a.bar_chart(pd.Series(summary["vehicles_by_class"], name="vehicles"))
        b.subheader("Violations by type")
        b.bar_chart(pd.Series(summary["violations_by_type"], name="violations"))
        if out.get("video") and Path(out["video"]).exists():
            st.subheader("Annotated video")
            st.download_button("Download annotated video", Path(out["video"]).read_bytes(), "annotated.mp4")

    with tab_evidence:
        snaps = sorted(p for p in Path(out["snapshots"]).glob("*.jpg") if "_context" not in p.name)
        if not snaps:
            st.info("No violations recorded.")
        cols = st.columns(4)
        for i, p in enumerate(snaps):
            cols[i % 4].image(str(p), caption=p.stem, use_container_width=True)

    with tab_vehicles:
        if out.get("vehicle_report"):
            df = pd.read_csv(out["vehicle_report"])
            only_bad = st.toggle("Only vehicles with violations")
            st.dataframe(df[df["violations"].notna()] if only_bad else df, use_container_width=True)
            st.download_button("Download vehicle_report.csv", Path(out["vehicle_report"]).read_bytes(),
                               "vehicle_report.csv")
            st.download_button("Download violations.csv", Path(out["violation_log"]).read_bytes(),
                               "violations.csv")

    with tab_challans:
        files = sorted(Path(out["challans"]).glob("*.json")) if out.get("challans") else []
        if not files:
            st.info("No challans generated.")
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            with st.expander(f"{d['challan_id']} — {d['violation']} — vehicle {d['vehicle_id']} — ₹{d['fine_amount']}"):
                left, right = st.columns([2, 1])
                left.json(d)
                if d.get("evidence"):
                    right.image(str(run_dir / "snapshots" / d["evidence"]))

    with tab_raw:
        st.json(summary)
