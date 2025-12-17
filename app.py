import streamlit as st
import cv2
import tempfile
import numpy as np
import time
import os
import pandas as pd

from gaze import analyze_gaze
from concentration import get_engagement_prob
from fusion import AttentionFusion, summarize_session

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="Student Engagement Analysis", layout="wide")
st.title("Evaluation of Participant Attention in Online Meetings Using Computer Vision")

# ------------------------------------------------
# SESSION STATE INIT
# ------------------------------------------------
if "fusion" not in st.session_state:
    st.session_state.fusion = AttentionFusion()

if "logs" not in st.session_state:
    st.session_state.logs = []

if "behavior_history" not in st.session_state:
    st.session_state.behavior_history = []

if "just_started" not in st.session_state:
    st.session_state.just_started = True

# ------------------------------------------------
# VISUAL HELPERS
# ------------------------------------------------
def determine_state_and_color(gaze, cas):
    if gaze in ["LEFT", "RIGHT", "DOWN"]:
        return f"DISTRACTED ({gaze})", (0, 0, 255)
    elif gaze == "UP":
        return "THINKING", (255, 165, 0)
    elif cas >= 0.7:
        return "ATTENTIVE", (0, 255, 0)
    elif cas >= 0.4:
        return "PARTIALLY ATTENTIVE", (255, 255, 0)
    else:
        return "INATTENTIVE", (128, 0, 128)


def draw_indicator(frame, box, label, color):
    (x, y, w, h) = box
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(
        frame, label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, color, 2
    )


def draw_metrics_overlay(frame, metrics, gaze):
    y = 30
    dy = 28

    def draw(text):
        nonlocal y
        cv2.putText(
            frame, text, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (255, 255, 255), 2
        )
        y += dy

    draw(f"Gaze Direction: {gaze}")
    draw(f"State: {metrics['STATE']}")
    draw(f"CAS: {metrics['CAS']}")
    draw(f"Look-Away Count: {metrics['Looking_Away_Events']}")
    draw(f"Max Away: {metrics['Max_Off_Duration_sec']} sec")


# ------------------------------------------------
# MODE SELECTION
# ------------------------------------------------
mode = st.sidebar.radio("Mode", ["Live Camera", "Upload Video"])

# ==============================================================
# LIVE CAMERA MODE
# ==============================================================
if mode == "Live Camera":
    st.subheader("🎥 Live Engagement Analysis")
    run_live = st.checkbox("Start / Stop Camera")

    frame_window = st.image([])
    metrics_box = st.empty()

    if run_live:
        if st.session_state.just_started:
            st.session_state.logs = []
            st.session_state.behavior_history = []
            st.session_state.fusion = AttentionFusion()
            st.session_state.just_started = False

        cap = cv2.VideoCapture(0)

        while run_live:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible.")
                break

            gaze = analyze_gaze(frame)
            prob, box = get_engagement_prob(frame)

            if box is not None:
                st.session_state.fusion.update(gaze, prob)
            else:
                st.session_state.fusion.update("ABSENT", None)

            metrics = st.session_state.fusion.compute_metrics()

            if metrics:
                st.session_state.logs.append(metrics)
                st.session_state.behavior_history = st.session_state.fusion.behavior_history

                label, color = determine_state_and_color(gaze, metrics["CAS"])

                if box is not None:
                    draw_indicator(frame, box, label, color)

                draw_metrics_overlay(frame, metrics, gaze)

                metrics_box.markdown(
                    f"""
                    **State:** {metrics['STATE']}  
                    **CAS:** {metrics['CAS']}  
                    **Look-Away Events:** {metrics['Looking_Away_Events']}  
                    **Max Look-Away:** {metrics['Max_Off_Duration_sec']} sec
                    """
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)

        cap.release()

    else:
        if len(st.session_state.logs) > 0:
            st.divider()
            st.success("Session Ended — Generating Report")

            summary = summarize_session(
                st.session_state.logs,
                st.session_state.behavior_history
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Average CAS", summary["Average_CAS"])
                st.metric("Final Verdict", summary["Final_Verdict"])

            with col2:
                st.subheader("Behavior Breakdown")
                st.dataframe(summary["Behavior_Breakdown"])

            df = pd.DataFrame(st.session_state.logs)
            st.line_chart(df["CAS"])

            st.session_state.just_started = True

# ==============================================================
# UPLOAD VIDEO MODE
# ==============================================================
elif mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload Lecture Video", type=["mp4", "avi"])

    if uploaded_file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(uploaded_file.read())

        if st.button("Analyze Video"):
            cap = cv2.VideoCapture(temp.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"avc1"),
                fps,
                (width, height)
            )

            fusion = AttentionFusion(fps=fps)
            logs = []

            progress = st.progress(0)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                gaze = analyze_gaze(frame)
                prob, box = get_engagement_prob(frame)

                if box is not None:
                    fusion.update(gaze, prob)
                else:
                    fusion.update("ABSENT", None)

                metrics = fusion.compute_metrics()
                if metrics:
                    logs.append(metrics)

                    label, color = determine_state_and_color(gaze, metrics["CAS"])
                    if box:
                        draw_indicator(frame, box, label, color)

                    draw_metrics_overlay(frame, metrics, gaze)

                out.write(frame)
                frame_count += 1
                if frame_count % 20 == 0:
                    progress.progress(frame_count / total_frames)

            cap.release()
            out.release()

            st.success("Processing Complete")
            st.video(output_path)

            summary = summarize_session(logs, fusion.behavior_history)

            st.metric("Average CAS", summary["Average_CAS"])
            st.metric("Final Verdict", summary["Final_Verdict"])
            st.dataframe(summary["Behavior_Breakdown"])
            st.line_chart([l["CAS"] for l in logs])
