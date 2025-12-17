# ------------------------------------------------
# FUSION MODULE
# Quantified Attentiveness via Time + Frequency
# ------------------------------------------------

from collections import deque, Counter
import numpy as np

HIGH_ATTENTION = 0.7
MID_ATTENTION = 0.4
DROWSY_THRESHOLD = 0.5

MIN_LOOKAWAY_SEC = 5          # tolerate thinking glances
MAX_LOOKAWAY_SEC = 15         # cap penalty
MAX_LOOKAWAY_EVENTS = 10      # cap frequency penalty


class AttentionFusion:
    def __init__(self, fps=30, window_seconds=5):
        self.fps = fps
        self.window_size = fps * window_seconds

        self.gaze_buffer = deque(maxlen=self.window_size)
        self.engagement_buffer = deque(maxlen=self.window_size)

        self.current_off_frames = 0
        self.off_durations = []
        self.lookaway_events = 0

        self.behavior_history = []

    def update(self, gaze_direction, engagement_prob):
        """
        gaze_direction: ON_SCREEN | OFF_SCREEN | ABSENT
        """

        is_off = gaze_direction == "OFF_SCREEN"

        # ---- GAZE BUFFER ----
        self.gaze_buffer.append(0 if is_off else 1)

        ep = engagement_prob if engagement_prob is not None else 0.0
        self.engagement_buffer.append(ep)

        # ---- LOOK-AWAY TRACKING ----
        if is_off:
            self.current_off_frames += 1
            if self.current_off_frames == 1:
                self.lookaway_events += 1
        else:
            if self.current_off_frames > 0:
                duration_sec = self.current_off_frames / self.fps
                self.off_durations.append(duration_sec)
                self.current_off_frames = 0

        # ---- BEHAVIOR LABEL ----
        off_sec = self.current_off_frames / self.fps

        if engagement_prob is None:
            behavior = "ABSENT"
        elif is_off and off_sec >= MIN_LOOKAWAY_SEC:
            behavior = "DISTRACTED"
        elif ep < DROWSY_THRESHOLD:
            behavior = "DROWSY / BORED"
        else:
            behavior = "FOCUSED"

        self.behavior_history.append(behavior)

    def compute_metrics(self):
        if not self.gaze_buffer:
            return None

        ogr = float(np.mean(self.gaze_buffer))
        ep = float(np.mean(self.engagement_buffer))

        # ---- NORMALIZED PENALTIES ----
        max_off = max(self.off_durations) if self.off_durations else 0
        norm_off_duration = min(max_off / MAX_LOOKAWAY_SEC, 1.0)
        norm_off_events = min(self.lookaway_events / MAX_LOOKAWAY_EVENTS, 1.0)

        # ---- COMPOSITE ATTENTIVENESS SCORE ----
        cas = (
            0.4 * ep +
            0.3 * ogr +
            0.2 * (1 - norm_off_duration) +
            0.1 * (1 - norm_off_events)
        )

        # ---- STATE ----
        if cas >= HIGH_ATTENTION:
            state = "ATTENTIVE"
        elif cas >= MID_ATTENTION:
            state = "PARTIALLY ATTENTIVE"
        else:
            state = "INATTENTIVE"

        current_off_sec = round(self.current_off_frames / self.fps, 2)

        return {
            "OGR": round(ogr, 2),
            "EP": round(ep, 2),
            "CAS": round(cas, 2),
            "STATE": state,
            "Looking_Away_Events": self.lookaway_events,
            "Current_Off_Duration_sec": current_off_sec,
            "Max_Off_Duration_sec": round(max_off, 2)
        }


def summarize_session(logs, behavior_history):
    if not logs:
        return None

    avg_cas = round(np.mean([l["CAS"] for l in logs]), 2)

    counts = Counter(behavior_history)
    total = sum(counts.values())

    breakdown = {
        k: round((v / total) * 100, 1)
        for k, v in counts.items()
    }

    if breakdown.get("FOCUSED", 0) >= 60:
        verdict = "HIGH ENGAGEMENT"
    elif breakdown.get("FOCUSED", 0) >= 30:
        verdict = "MODERATE ENGAGEMENT"
    else:
        verdict = "LOW ENGAGEMENT"

    return {
        "Average_CAS": avg_cas,
        "Final_Verdict": verdict,
        "Behavior_Breakdown": breakdown
    }
