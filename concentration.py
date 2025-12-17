# -----------------------------------------
# FACIAL ENGAGEMENT MODULE
# -----------------------------------------
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "concentration_model.keras")
FACE_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def get_engagement_prob(frame):
    """
    Returns:
        (probability, face_box) OR (None, None)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
    face = frame[y:y+h, x:x+w]

    if face.size == 0:
        return None, None

    face = cv2.resize(face, (224, 224))
    face = preprocess_input(face.astype("float32"))
    face = np.expand_dims(face, axis=0)

    prob = model.predict(face, verbose=0)[0][0]
    return float(prob), (x, y, w, h)