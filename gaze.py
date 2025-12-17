# -----------------------------------------
# GAZE DIRECTION DETECTION MODULE
# -----------------------------------------

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier("haarcascade_eye.xml")

clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

prev_gray = None
pupil_point = None
last_eye_box = None


def classify_direction(cx, cy, ew, eh):
    """Returns gaze direction based on normalized pupil position."""
    nx = cx / ew
    ny = cy / eh

    horiz = "CENTER"
    vert  = "CENTER"

    if nx < 0.35:
        horiz = "LEFT"
    elif nx > 0.65:
        horiz = "RIGHT"

    if ny < 0.35:
        vert = "UP"
    elif ny > 0.65:
        vert = "DOWN"

    # Combine
    if horiz == "CENTER" and vert == "CENTER":
        return "CENTER"
    if vert != "CENTER":
        return vert
    return horiz


def analyze_gaze(frame):
    """
    Returns:
        "LEFT", "RIGHT", "UP", "DOWN", "CENTER", or "NO_FACE"
    """

    global prev_gray, pupil_point, last_eye_box

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    soft = cv2.GaussianBlur(gray, (3,3), 0)

    # Reset tracker if resolution changes
    if prev_gray is not None and prev_gray.shape != soft.shape:
        prev_gray = None
        pupil_point = None
        last_eye_box = None

    faces = face_cascade.detectMultiScale(soft, 1.2, 5)

    if len(faces) == 0:
        prev_gray = soft.copy()
        pupil_point = None
        last_eye_box = None
        return "NO_FACE"

    # Use largest face
    (x, y, w, h) = max(faces, key=lambda b: b[2]*b[3])
    roi_gray = soft[y:y+h, x:x+w]

    detected_point = None
    detected_eye = None

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.15, 5)

    for (ex, ey, ew, eh) in eyes:
        eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
        eye_eq = clahe.apply(cv2.GaussianBlur(eye_gray, (3,3), 0))

        _, thresh = cv2.threshold(
            eye_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)

            if ew * 0.05 < radius < ew * 0.45:
                detected_point = np.array([[cx + ex + x, cy + ey + y]], dtype=np.float32)
                detected_eye = (cx, cy, ew, eh)
                break

    # Optical flow tracking
    if prev_gray is not None and pupil_point is not None and last_eye_box is not None:
        try:
            new_point, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, soft, pupil_point, None, **lk_params
            )
            if status[0][0] == 1:
                pupil_point = new_point
                prev_gray = soft.copy()
                cx, cy, ew, eh = last_eye_box
                return classify_direction(cx, cy, ew, eh)
        except:
            pupil_point = None

    if detected_point is not None:
        pupil_point = detected_point
        prev_gray = soft.copy()
        last_eye_box = detected_eye
        cx, cy, ew, eh = detected_eye
        return classify_direction(cx, cy, ew, eh)

    prev_gray = soft.copy()
    return "CENTER"
