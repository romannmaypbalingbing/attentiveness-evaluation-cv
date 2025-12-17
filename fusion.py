# --- START OF FILE fusion.py ---
from collections import deque
import numpy as np

THRESH_HIGH_ATTENTION = 0.70
THRESH_LOW_ATTENTION = 0.40
THRESH_DROWSY_MODEL = 0.50

class AttentionFusion:
    def __init__(self, window_seconds=5, fps=30):
        self.fps = fps
        self.window_size = window_seconds * fps

        # Buffers
        self.gaze_buffer = deque(maxlen=self.window_size)
        self.engagement_buffer = deque(maxlen=self.window_size)

        # Behavior history
        self.behavior_history = []

        # NEW: temporal tracking
        self.current_off_start = None
        self.off_durations = []
        self.frame_index = 0

    def update(self, gaze_state, engagement_prob):
        self.frame_index += 1

        # Normalize gaze
        g_val = 1.0 if gaze_state == "ON_SCREEN" else 0.0
        e_val = float(engagement_prob) if engagement_prob is not None else 0.0

        self.gaze_buffer.append(g_val)
        self.engagement_buffer.append(e_val)

        # ---- Track OFF-SCREEN duration ----
        if gaze_state == "OFF_SCREEN":
            if self.current_off_start is None:
                self.current_off_start = self.frame_index
        else:
            if self.current_off_start is not None:
                duration = (self.frame_index - self.current_off_start) / self.fps
                self.off_durations.append(duration)
                self.current_off_start = None

        # ---- Behavior classification ----
        if engagement_prob is None:
            behavior = "ABSENT"
        elif gaze_state == "OFF_SCREEN":
            behavior = "DISTRACTED"
        elif e_val < THRESH_DROWSY_MODEL:
            behavior = "DROWSY/BORED"
        else:
            behavior = "FOCUSED"

        self.behavior_history.append(behavior)

    def compute_metrics(self):
        if not self.gaze_buffer:
            return None

        ogr = np.mean(self.gaze_buffer)
        ep = np.mean(self.engagement_buffer)
        cas = (0.6 * ep) + (0.4 * ogr)

        # Derived metrics
        off_ratio = 1.0 - ogr
        avg_off_duration = np.mean(self.off_durations) if self.off_durations else 0
        max_off_duration = max(self.off_durations) if self.off_durations else 0
        looking_away_events = len(self.off_durations)

        if cas >= THRESH_HIGH_ATTENTION:
            state = "HIGHLY ENGAGED"
        elif cas >= THRESH_LOW_ATTENTION:
            state = "PARTIALLY ENGAGED"
        else:
            state = "DISENGAGED"

        return {
            "OGR": round(ogr, 2),
            "EP": round(ep, 2),
            "CAS": round(cas, 2),
            "Off_Screen_Ratio": round(off_ratio, 2),
            "Avg_Off_Duration_sec": round(avg_off_duration, 2),
            "Max_Off_Duration_sec": round(max_off_duration, 2),
            "Looking_Away_Events": looking_away_events,
            "STATE": state,
            "BEHAVIOR": self.behavior_history[-1]
        }
    
def summarize_session(logs, behavior_history):
    if not logs:
        return None

    avg_cas = np.mean([l["CAS"] for l in logs])
    avg_ogr = np.mean([l["OGR"] for l in logs])

    distracted = behavior_history.count("DISTRACTED")
    focused = behavior_history.count("FOCUSED")

    # Interpretation logic
    if avg_ogr > 0.7 and avg_cas > 0.7:
        verdict = "SUSTAINED ATTENTIVENESS"
    elif distracted > focused:
        verdict = "FREQUENT DISTRACTION DETECTED"
    else:
        verdict = "INTERMITTENT ATTENTION (NORMAL THINKING PATTERNS)"

    return {
        "Average_CAS": round(avg_cas, 2),
        "Average_OGR": round(avg_ogr, 2),
        "Total_Looking_Away_Events": distracted,
        "Final_Verdict": verdict
    }
