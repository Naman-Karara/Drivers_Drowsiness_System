import streamlit as st
import tensorflow as tf

import numpy as np
import cv2
import time
import os
import ssl
from typing import Optional
from esp32_camera import ESP32Camera

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# UI Setup
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("🚗 Real-Time Driver Drowsiness Detection (ESP32-CAM)")
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        padding: 10px;
        font-weight: bold;
    }
    .stAlert {
        padding: 20px;
        font-size: 18px;
    }
    .prediction-score {
        font-size: 16px;
        color: #ff4b4b;
        font-weight: bold;
    }
    .drowsy-alert {
        color: red;
        font-weight: bold;
        animation: blinker 1s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# Drowsiness threshold
DROWSINESS_THRESHOLD = 0.57

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.update({
        'running': False,
        'cap': None,
        'drowsy_start_time': None,
        'last_prediction': None,
        'face_detected': False,
        'alarm_triggered': False
    })

# UI Elements
col1, col2 = st.columns([3, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    status_placeholder = st.empty()
    st.markdown("### Detection Info")
    info_placeholder = st.empty()
    esp_ip = st.text_input("ESP32 Camera IP", "192.168.219.80")

# Load models with caching
@st.cache_resource
def load_models():
    PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    CAFFEMODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    # Download files if needed
    if not os.path.exists("deploy.prototxt"):
        import urllib.request
        urllib.request.urlretrieve(PROTOTXT_URL, "deploy.prototxt")
    if not os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"):
        import urllib.request
        urllib.request.urlretrieve(CAFFEMODEL_URL, "res10_300x300_ssd_iter_140000.caffemodel")
    
    try:
        model = tf.keras.models.load_model("drowsiness_model.h5")
        net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
        return model, net
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

model, face_net = load_models()

def detect_faces(frame: np.ndarray) -> list:
    """Detect faces using DNN face detector"""
    if frame is None:
        return []
        
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            faces.append(box.astype("int"))
    
    return faces

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Prepare image for model prediction"""
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def run_detection():
    """Main detection loop"""
    st.session_state.cap = ESP32Camera(ip_address=esp_ip)
    
    detection_fail_count = 0
    last_pred_time = time.time()
    
    while st.session_state.running:
        ret, frame = st.session_state.cap.capture_frame()
        
        if not ret:
            status_placeholder.error("Failed to get frame from ESP32 camera")
            time.sleep(0.1)
            continue
            
        # Face detection
        faces = detect_faces(frame)
        
        if len(faces) == 0:
            detection_fail_count += 1
            st.session_state.face_detected = False
            
            if detection_fail_count > 5:
                status_placeholder.warning("⚠ Face not detected - adjust your position")
            
            frame_placeholder.image(frame, channels="BGR")
            time.sleep(0.1)
            continue
        
        detection_fail_count = 0
        st.session_state.face_detected = True
        (startX, startY, endX, endY) = faces[0]
        
        try:
            # Extract and process face
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
                
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            processed_face = preprocess_image(face_rgb)
            
            # Predict drowsiness
            current_time = time.time()
            if current_time - last_pred_time > 0.5:
                prediction = model.predict(processed_face, verbose=0)[0][0]
                st.session_state.last_prediction = prediction
                last_pred_time = current_time
            
            # Drowsiness logic
            if st.session_state.last_prediction >= DROWSINESS_THRESHOLD:
                if st.session_state.drowsy_start_time is None:
                    st.session_state.drowsy_start_time = time.time()
                elif time.time() - st.session_state.drowsy_start_time >= 2:
                    if not st.session_state.alarm_triggered:
                        st.session_state.cap.trigger_alarm(True)
                        st.session_state.alarm_triggered = True
                    status_placeholder.markdown(
                        f"<div class='drowsy-alert'>🚨 DROWSINESS DETECTED! (Score: {st.session_state.last_prediction:.2f})</div>", 
                        unsafe_allow_html=True
                    )
            else:
                if st.session_state.alarm_triggered:
                    st.session_state.cap.trigger_alarm(False)
                    st.session_state.alarm_triggered = False
                st.session_state.drowsy_start_time = None
                status_placeholder.success("✅ Driver is alert")
            
            # Visual feedback
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            score_text = f"Drowsy: {st.session_state.last_prediction:.2f}" if st.session_state.last_prediction >= DROWSINESS_THRESHOLD else f"Alert: {st.session_state.last_prediction:.2f}"
            text_color = (0, 0, 255) if st.session_state.last_prediction >= DROWSINESS_THRESHOLD else (0, 255, 0)
            cv2.putText(frame, score_text, (startX, startY-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            # Update info panel
            drowsy_status = "<span class='drowsy-alert'>DROWSY</span>" if st.session_state.last_prediction >= DROWSINESS_THRESHOLD else "ALERT"
            info_text = f"""
            *Detection Status:*
            - Face Detected: ✅
            - Driver Status: {drowsy_status}
            - Drowsiness Score: <span class="prediction-score">{st.session_state.last_prediction:.2f}</span>
            - Threshold: {DROWSINESS_THRESHOLD}
            - Alarm State: {'ON' if st.session_state.alarm_triggered else 'OFF'}
            - ESP32 IP: {esp_ip}
            """
            info_placeholder.markdown(info_text, unsafe_allow_html=True)
            
        except Exception as e:
            status_placeholder.warning(f"Processing error: {str(e)}")
            continue
        
        frame_placeholder.image(frame, channels="BGR")
        time.sleep(0.05)
    
    # Cleanup
    if st.session_state.alarm_triggered:
        st.session_state.cap.trigger_alarm(False)
    if st.session_state.cap is not None:
        st.session_state.cap.release()

# Control buttons
with st.container():
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        if st.button("Start Detection", key="start") and not st.session_state.running:
            st.session_state.running = True
            run_detection()
    with col2:
        if st.button("Stop Detection", key="stop") and st.session_state.running:
            st.session_state.running = False
            status_placeholder.info("System stopped")
            info_placeholder.empty()

# Instructions
with st.sidebar:
    st.markdown("### 📋 Instructions")
    st.write("1. Enter ESP32 IP address")
    st.write("2. Click 'Start Detection'")
    st.write("3. Position your face in the camera view")
    st.write(f"4. System will alert when score ≥ {DROWSINESS_THRESHOLD}")
    st.write("5. Click 'Stop Detection' when finished")
    
    st.markdown("### ℹ About")
    st.write("Alert Threshold:", DROWSINESS_THRESHOLD)
    st.write("Scoring:")
    st.write(f"- Below {DROWSINESS_THRESHOLD}: Alert")
    st.write(f"- ≥ {DROWSINESS_THRESHOLD}: Drowsy")

if not st.session_state.running and not st.session_state.get('initialized', False):
    status_placeholder.info("Enter ESP32 IP and click 'Start Detection'")
    st.session_state.initialized = True