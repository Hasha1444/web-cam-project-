import cv2
import dlib
import time
import hashlib
import streamlit as st
import pandas as pd
from datetime import datetime

from detectors import drowsiness, eye_blink, gaze_tracking, head_pose, presence, mouth_movement
from utils.camera_checker import is_camera_on
from utils.time_tracker import AbsenceTimer
from backend.log_activity import log_event
from backend.notify_tutor import notify_tutor
from alerts.message_queue import WarningManager
from config import *

# Initialize
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
warn_manager = WarningManager()
absence_timer = AbsenceTimer()

# For dashboard storage
if "logs" not in st.session_state:
    st.session_state.logs = []

# For notification throttling
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = {}

ALERT_INTERVAL = 10  # seconds


def get_face_id(shape):
    landmarks = [(p.x, p.y) for p in shape.parts()]
    flat = [coord for point in landmarks for coord in point]
    data = bytearray()
    for num in flat:
        data.extend(num.to_bytes(2, byteorder='big', signed=True))
    return hashlib.md5(data).hexdigest()


def log_and_alert(face_id, event):
    now = time.time()
    last_time = st.session_state.last_alert_time.get(event, 0)

    if now - last_time >= ALERT_INTERVAL:
        warn_manager.issue_warning(event)
        log_event(face_id, event)
        notify_tutor(face_id)
        st.session_state.logs.append({
            "Face ID": face_id,
            "Event": event,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.session_state.last_alert_time[event] = now


def monitor_stream():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_placeholder = st.image([])
    status_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or not is_camera_on(cap):
            log_and_alert("unknown", "Camera turned off")
            time.sleep(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            absence_timer.update(absent=True)
            if absence_timer.exceeded_limit():
                log_and_alert("unknown", "User left screen")
            continue
        else:
            absence_timer.update(absent=False)

        alerts = []

        for face in faces:
            shape = predictor(gray, face)
            face_id = get_face_id(shape)

            drowsy = drowsiness.is_drowsy(shape)
            blinking = eye_blink.is_blinking(shape)
            looking_away = gaze_tracking.is_watching_away(shape)
            head_down = head_pose.is_head_down(shape)
            talking = mouth_movement.is_talking(shape)

            if drowsy:
                log_and_alert(face_id, "Drowsiness")
                alerts.append("😴 Drowsy")
            if blinking:
                log_event(face_id, "Blinking")
            if looking_away:
                log_and_alert(face_id, "Looking Away")
                alerts.append("👀 Looking Away")
            if head_down:
                log_and_alert(face_id, "Head Down")
                alerts.append("📱 Head Down")
            if talking:
                log_and_alert(face_id, "Talking")
                alerts.append("🗣️ Talking")

        # Display alert info
        status_placeholder.markdown("### Live Alerts: " + ", ".join(alerts) if alerts else "🟢 Monitoring")

        # Show webcam frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

        if st.session_state.get("stop", False):
            break

    cap.release()
    st.success("Monitoring stopped")
    st.session_state["stop"] = False


# --------- Streamlit UI ----------
st.set_page_config(page_title="EduGaze Attention Tracker", layout="wide")
st.title("📸 Webcam-Based Attention Tracker")

col1, col2 = st.columns(2)

with col1:
    if st.button("📷 Open Camera"):
        st.session_state.stop = False
        monitor_stream()

with col2:
    if st.button("📊 View Dashboard"):
        if st.session_state.logs:
            df = pd.DataFrame(st.session_state.logs)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No activity logged yet.")

st.markdown("---")
st.caption("Built with Dlib, OpenCV, and Streamlit")
