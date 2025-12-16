import streamlit as st
import cv2
import tempfile
import time

from gaze import analyze_gaze
from concentration import get_engagement_prob
from fusion import AttentionFusion, summarize_logs

# ---------- Session State Initialization ----------
if "running" not in st.session_state:
    st.session_state.running = False

if "metrics_log" not in st.session_state:
    st.session_state.metrics_log = []

st.set_page_config(page_title="Attentiveness Evaluation", layout="centered")

st.title("Evaluation of Participant Attention in Online Meetings Using Computer Vision")

mode = st.radio(
    "Choose Analysis Mode:",
    ["Live Video Feed", "Upload Video"]
)

def classify_attention(cas):
    if cas < 0.4:
        return "Inattentive"
    elif cas < 0.7:
        return "Partially Attentive"
    else:
        return "Attentive"


# ---------------------------
# LIVE VIDEO FEED
# ---------------------------
if mode == "Live Video Feed":
    st.info("Press START to begin. Press STOP to finish and view results.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start"):
            st.session_state.running = True
            st.session_state.metrics_log = []

    with col2:
        if st.button("⏹ Stop"):
            st.session_state.running = False

    frame_window = st.image([])

    if st.session_state.running:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        fusion = AttentionFusion(window_seconds=5, fps=30)

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            gaze_state = analyze_gaze(frame)
            engagement_prob = get_engagement_prob(frame)

            fusion.update(gaze_state, engagement_prob)
            metrics = fusion.compute_metrics()

            if metrics:
                st.session_state.metrics_log.append(metrics)
                label = f"{metrics['STATE']} | CAS: {metrics['CAS']}"
                cv2.putText(
                    frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)

        cap.release()

    if not st.session_state.running and len(st.session_state.metrics_log) > 0:
        st.subheader("📊 Session Summary")

        summary = summarize_logs(st.session_state.metrics_log)

        # ⬇️ ADD classification WITHOUT touching OGR or EP
        avg_cas = summary.get("Average CAS")
        if avg_cas is not None:
            summary["Overall Attention Classification"] = classify_attention(avg_cas)

        st.json(summary)




# ---------------------------
# VIDEO UPLOAD
# ---------------------------
if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a meeting video", type=["mp4", "avi"])

    st.session_state.metrics_log = []

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_window = st.image([])

        fusion = AttentionFusion(window_seconds=5, fps=30)

        st.info("Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gaze_state = analyze_gaze(frame)
            engagement_prob = get_engagement_prob(frame)

            fusion.update(gaze_state, engagement_prob)
            metrics = fusion.compute_metrics()

            if metrics:
                st.session_state.metrics_log.append(metrics)
                label = f"{metrics['STATE']} | CAS: {metrics['CAS']}"
                cv2.putText(
                    frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)

        cap.release()

        st.subheader("📊 Video Analysis Summary")
        summary = summarize_logs(st.session_state.metrics_log)
        st.json(summary)