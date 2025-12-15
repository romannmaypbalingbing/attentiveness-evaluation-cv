# ---------------------------
# FUSION MODULE
# ---------------------------
# fuses gaze detection and student engagement probability over time to compute an attentiveness score
from collections import deque
import numpy as np


class AttentionFusion:
    def __init__(self, window_seconds=5, fps=30):
        """
        Temporal fusion over a sliding window
        """
        self.window_size = window_seconds * fps

        # buffer storing gaze states (1 = ON_SCREEN, 0 = OFF_SCREEN)
        self.gaze_buffer = deque(maxlen=self.window_size)
        # buffer storing engagement probabilities (0.0 - 1.0)
        self.engagement_buffer = deque(maxlen=self.window_size)

    def update(self, gaze_state, engagement_prob):
        """
        gaze_state: "ON_SCREEN" or "OFF_SCREEN"
        engagement_prob: float in [0,1] or None
        """

        # ---- Gaze ----
        self.gaze_buffer.append(1 if gaze_state == "ON_SCREEN" else 0)

        # ---- Engagement ----
        if engagement_prob is None:
            engagement_prob = 0.0

        self.engagement_buffer.append(float(engagement_prob))

    # computes attentiveness metrics
    def compute_metrics(self):
        """
            OGR:    On-Gaze Ratio
            EP:     Engagement Probability
            CAS:    Composite Attentiveness Score
            STATE:  Attentiveness Classification
        """
        if len(self.gaze_buffer) == 0:
            return None

        ogr = np.mean(self.gaze_buffer)
        ep = np.mean(self.engagement_buffer)

        # we defined the weights as this.
        cas = 0.6 * ep + 0.4 * ogr

        
        if cas >= 0.7:
            state = "ATTENTIVE"
        elif cas >= 0.4:
            state = "PARTIALLY_ATTENTIVE"
        else:
            state = "INATTENTIVE"

        return {
            "OGR": round(ogr, 2),
            "EP": round(ep, 2),
            "CAS": round(cas, 2),
            "STATE": state
        }
    
def summarize_logs(logs):
    if len(logs) == 0:
        return None

    avg_ogr = sum(l["OGR"] for l in logs) / len(logs)
    avg_ep  = sum(l["EP"] for l in logs) / len(logs)
    avg_cas = sum(l["CAS"] for l in logs) / len(logs)

    return {
        "Average OGR": round(avg_ogr, 2),
        "Average EP": round(avg_ep, 2),
        "Average CAS": round(avg_cas, 2)
    }