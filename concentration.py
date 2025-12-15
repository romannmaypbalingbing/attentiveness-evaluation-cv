# --------------------------------
# STUDENT ENGAGEMENT PROBABILITY
# --------------------------------
# estimates student engagement by classifying facial engagement using MobileNetV2 Model
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_CASCADE_PATH = os.path.join(
    BASE_DIR, "haarcascade_frontalface_default.xml"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "concentration_model.keras"
)

# ---------- Load resources ONCE ----------
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

# ---------- Main function ----------
def get_engagement_prob(frame):
    """
    Input:
        frame (numpy array): BGR image from OpenCV

    Output:
        float in [0, 1] -> engagement probability
        None -> if no face detected
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return None

    # Use largest detected face (only detects one participant)
    (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
    face = frame[y:y+h, x:x+w]

    if face.size == 0:
        return None

    # Preprocess for MobileNetV2
    face = cv2.resize(face, (224, 224))
    face = face.astype("float32")
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    prob = model.predict(face, verbose=0)[0][0]
    return float(prob)