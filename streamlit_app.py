import sys
import subprocess

# Force headless OpenCV to prevent import crashes on Linux/Docker servers
try:
    import cv2
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
        import cv2
    except Exception as e:
        pass

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
import math
import wave
import struct
import base64
import shutil
import av
from collections import deque
import joblib

# ─────────────────────────────────────────────────────────────
# Setup Python paths and dependencies
# ─────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RF_APP_PATH = os.path.join(CURRENT_DIR, "rf_app")

# Ensure rf_app directory exists
os.makedirs(RF_APP_PATH, exist_ok=True)

# Ensure rf_app is treated as a package and prioritized in sys.path
init_file = os.path.join(RF_APP_PATH, "__init__.py")
if not os.path.exists(init_file):
    try:
        with open(init_file, "w") as f:
            pass
    except Exception:
        pass

if RF_APP_PATH not in sys.path:
    sys.path.insert(0, RF_APP_PATH)

# Ensure face_landmarker.task is in the rf_app folder as expected by rf_app modules
def setup_files():
    src_task = os.path.join(CURRENT_DIR, "face_landmarker.task")
    dst_task = os.path.join(RF_APP_PATH, "face_landmarker.task")
    
    if os.path.exists(src_task) and not os.path.exists(dst_task):
        try:
            shutil.copy(src_task, dst_task)
            print(f"[SYSTEM] Copied face_landmarker.task to {dst_task}")
        except Exception as e:
            print(f"[ERROR] Failed to copy task file: {e}")

setup_files()

# Import from rf_app
try:
    from feature_extraction import FeatureExtractor
    from detector import FaceLandmarkDetector
except ImportError as e:
    st.error(f"Failed to import from rf_app: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Automatic model training and sound generation on startup
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(CURRENT_DIR, "random_forest_model.pkl")
ALARM_PATH = os.path.join(CURRENT_DIR, "alarm.wav")

def generate_alarm_sound(file_path=ALARM_PATH):
    """Generates a standard two-tone 'PEE-PAW' warning alarm WAV file if missing."""
    if os.path.exists(file_path):
        return
        
    sample_rate = 8000
    duration_tone = 0.45  # ~450ms per tone (siren speed)
    num_tones = 6         # 3 cycles of PEE-PAW
    frequencies = [950, 700] * (num_tones // 2)  # High PEE (950Hz), Low PAW (700Hz)
    
    audio_data = []
    for freq in frequencies:
        num_samples = int(sample_rate * duration_tone)
        for i in range(num_samples):
            t = i / sample_rate
            # 16-bit PCM Sine Wave
            val = int(32767 * math.sin(2 * math.pi * freq * t))
            audio_data.append(struct.pack('<h', val))
            
    try:
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)      # Mono
            wav_file.setsampwidth(2)      # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b''.join(audio_data))
        print(f"[SYSTEM] Generated default alarm sound at '{file_path}'.")
    except Exception as e:
        print(f"[ERROR] Failed to generate alarm sound: {e}")


def train_fallback_model(model_path=MODEL_PATH):
    """Automatically trains a RandomForestClassifier on synthetic data if no model exists."""
    if os.path.exists(model_path):
        return
        
    st.info("🤖 RF Model not found in root. Automatically training a classifier on synthetic data...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    rng = np.random.RandomState(42)
    n_per_class = 1500
    rows = []
    
    # 1. Alert State
    for _ in range(n_per_class):
        ear = rng.uniform(0.25, 0.35)
        if rng.random() < 0.05:
            ear = rng.uniform(0.08, 0.19)  # brief blinks
        rows.append({
            "ear": ear + rng.normal(0, 0.008),
            "mar": rng.uniform(0.10, 0.30) + rng.normal(0, 0.01),
            "blink_count": int(rng.randint(5, 25)),
            "eye_closure_duration": rng.uniform(0, 0.12) if ear < 0.20 else 0.0,
            "yawn_duration": 0.0,
            "face_present": 1.0,
            "label": "Alert"
        })
        
    # 2. Drowsy State
    for _ in range(n_per_class):
        ear = rng.uniform(0.08, 0.21) + rng.normal(0, 0.008)
        closure = rng.uniform(1.2, 6.0) if ear < 0.20 else rng.uniform(0, 0.3)
        rows.append({
            "ear": max(0.03, ear),
            "mar": rng.uniform(0.10, 0.28) + rng.normal(0, 0.01),
            "blink_count": int(rng.randint(0, 10)),
            "eye_closure_duration": closure,
            "yawn_duration": rng.uniform(0, 0.5),
            "face_present": 1.0,
            "label": "Drowsy"
        })
        
    # 3. Yawning State
    for _ in range(n_per_class):
        mar = rng.uniform(0.60, 0.92) + rng.normal(0, 0.01)
        rows.append({
            "ear": rng.uniform(0.20, 0.32) + rng.normal(0, 0.008),
            "mar": max(0.30, mar),
            "blink_count": int(rng.randint(5, 25)),
            "eye_closure_duration": rng.uniform(0, 0.25),
            "yawn_duration": rng.uniform(1.2, 6.0),
            "face_present": 1.0,
            "label": "Yawning"
        })
        
    df = pd.DataFrame(rows)
    feature_cols = ["ear", "mar", "blink_count", "eye_closure_duration", "yawn_duration", "face_present"]
    df[feature_cols] = df[feature_cols].clip(lower=0.0)
    
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    payload = {
        "model": model,
        "label_encoder": le,
        "feature_names": feature_cols,
        "class_labels": ["Alert", "Drowsy", "Yawning"]
    }
    
    # Save to root and rf_app
    joblib.dump(payload, model_path, compress=3)
    joblib.dump(payload, os.path.join(RF_APP_PATH, "random_forest_model.pkl"), compress=3)
    st.success("🎉 Fallback Random Forest model trained and saved successfully!")

# Load model helper
@st.cache_resource
def get_rf_model(path=MODEL_PATH):
    import pandas as pd # needed inside cache for model unpickling
    if not os.path.exists(path):
        train_fallback_model(path)
    try:
        payload = joblib.load(path)
        return payload["model"], payload["label_encoder"]
    except Exception as e:
        st.error(f"Error loading Random Forest model: {e}")
        return None, None

# Run startup builders
generate_alarm_sound()
model_rf, le_rf = get_rf_model()

# ─────────────────────────────────────────────────────────────
# Streamlit App Configurations
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling block
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Outfit:wght@300;500;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
    color: #e5e7eb;
    font-family: 'Outfit', sans-serif;
}

/* Glassmorphism Containers */
.glass-panel {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.4);
}

/* Glow indicators */
.status-card {
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: all 0.3s ease;
}

.awake-glow {
    background: rgba(16, 185, 129, 0.08);
    border: 2px solid #10b981;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.25);
    color: #10b981;
}

.drowsy-glow {
    background: rgba(239, 68, 68, 0.08);
    border: 2px solid #ef4444;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
    color: #ef4444;
}

.yawn-glow {
    background: rgba(245, 158, 11, 0.08);
    border: 2px solid #f59e0b;
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.25);
    color: #f59e0b;
}

.glow-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    font-size: 1.8rem;
    letter-spacing: 1.5px;
    margin: 0;
}

.stat-label {
    font-size: 0.9rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stat-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
}

/* Scrollable log panel */
.log-panel {
    background: #030712;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 12px;
    height: 180px;
    overflow-y: scroll;
    font-family: monospace;
    font-size: 0.85rem;
    color: #34d399;
}
</style>
""", unsafe_allow_html=True)

# Helper function to play sound client-side
def trigger_html_alarm():
    if os.path.exists(ALARM_PATH):
        try:
            with open(ALARM_PATH, "rb") as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
                <audio autoplay loop id="alarm-audio">
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        except Exception:
            pass

# Log state management
if "event_log" not in st.session_state:
    st.session_state.event_log = ["System online."]

def log_event(msg):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.event_log.append(f"[{timestamp}] {msg}")
    if len(st.session_state.event_log) > 40:
        st.session_state.event_log.pop(0)

# WebRTC streaming state
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

class DrowsinessProcessor:
    def __init__(self):
        # Setup detector & extractor inside background thread
        setup_files()
        self.landmarker = FaceLandmarkDetector()
        self.extractor = FeatureExtractor()
        
        # Load local model reference
        self.model = None
        self.le = None
        
        # Latest frame parameters
        self.ear = 0.30
        self.mar = 0.15
        self.blink_count = 0
        self.eye_closure_duration = 0.0
        self.yawn_duration = 0.0
        self.prediction = "Alert"
        self.confidence = 1.0
        self.face_present = False
        
        # History queue for prediction smoothing
        self.pred_queue = deque(maxlen=7)
        self.prev_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Flip BGR image for selfie-view
        h, w, _ = img.shape
        now = time.time()
        
        # 1. Face landmarks detection
        landmarks = self.landmarker.detect(img)
        face_present = landmarks is not None
        
        # 2. Update feature extractor
        feat = self.extractor.update(landmarks, h, w, face_present)
        
        # 3. Classify frame using Random Forest or heuristic fallback
        prediction = "Alert"
        confidence = 1.0
        
        if face_present and self.model is not None:
            features = self.extractor.to_vector(feat)
            pred_idx = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]
            raw_pred = self.le.inverse_transform([pred_idx])[0]
            confidence = float(proba.max())
            
            # Smooth predictions
            self.pred_queue.append(raw_pred)
            prediction = max(set(self.pred_queue), key=self.pred_queue.count)
        elif face_present:
            # Simple threshold fallback if model not injected
            if feat["ear"] < 0.20 and feat["eye_closure_duration"] > 1.5:
                prediction = "Drowsy"
            elif feat["mar"] > 0.65:
                prediction = "Yawning"
            else:
                prediction = "Alert"
                
        # 4. Set state metrics
        self.ear = feat["ear"]
        self.mar = feat["mar"]
        self.blink_count = feat["blink_count"]
        self.eye_closure_duration = feat["eye_closure_duration"]
        self.yawn_duration = feat["yawn_duration"]
        self.prediction = prediction
        self.confidence = confidence
        self.face_present = face_present
        
        # Calculate local FPS
        dt = now - self.prev_time
        fps = 1.0 / dt if dt > 0 else 30.0
        self.prev_time = now
        
        # 5. Draw face landmark mesh overlay
        self.landmarker.annotate(img, landmarks, prediction=prediction)
        
        # 6. Render HUD elements directly onto the frame
        color_map = {
            "Alert": (80, 200, 0),     # Tealish Green
            "Drowsy": (0, 0, 255),     # Red
            "Yawning": (255, 165, 0)   # Orange
        }
        pred_color = color_map.get(prediction, (200, 200, 200))
        
        # Drowsiness alert banner
        if prediction == "Drowsy" and confidence >= 0.70:
            cv2.rectangle(img, (20, h // 2 - 40), (w - 20, h // 2 + 40), (0, 0, 200), -1)
            cv2.putText(img, "WARNING: DROWSINESS DETECTED!", (w // 2 - 250, h // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                        
        if prediction == "Yawning":
            cv2.putText(img, "YAWN DETECTED - COFFEE BREAK!", (30, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                        
        # Overlay metrics values on video feed
        cv2.putText(img, f"EAR: {self.ear:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"MAR: {self.mar:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"FPS: {int(fps)}", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────────────────────────
# Sidebar Elements
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 Detector Control")
    st.markdown("---")
    
    # Session state settings
    st.subheader("🛠️ Settings")
    mode_selection = st.selectbox(
        "Detection Model Mode",
        ["Random Forest (ML)", "Heuristic (Rule-Based)"]
    )
    
    alarm_thresh = st.slider(
        "Alarm Confidence Threshold",
        min_value=0.50, max_value=0.95, value=0.70, step=0.05,
        help="Alert triggers when the model predicts Drowsy above this confidence."
    )
    
    is_muted = st.checkbox("🔇 Mute Audio Siren", value=False)
    
    st.markdown("---")
    
    # Project Information
    st.subheader("ℹ️ Project Info")
    st.markdown(
        """
        This dashboard uses machine learning and deep learning to monitor driver status in real-time.
        
        - **MediaPipe FaceLandmarker**: Computes 3D face meshes at high frame rates to capture facial geometry.
        - **Eye Aspect Ratio (EAR)**: Monitors eye closure (blinks vs micro-sleeps).
        - **Mouth Aspect Ratio (MAR)**: Monitors yawn frequency.
        - **Random Forest**: Classifies driver state into **Alert**, **Drowsy**, or **Yawning**.
        """
    )
    st.info("Ready to scan driver face mesh.")

# ─────────────────────────────────────────────────────────────
# Main Application UI Layout
# ─────────────────────────────────────────────────────────────
st.title("🚗 Driver Drowsiness Detection System")
st.markdown("Monitor driver alertness in real-time using custom Computer Vision models.")

# Tabs for operating modes
tab_webrtc, tab_snapshot, tab_simulator = st.tabs([
    "🎥 Live WebRTC Camera Stream",
    "📸 Camera Snapshot Analyzer",
    "🎮 Interactive Face Simulator"
])

# ─────────────────────────────────────────────────────────────
# TAB 1: Live WebRTC Stream
# ─────────────────────────────────────────────────────────────
with tab_webrtc:
    st.markdown("### 🎥 Live Real-Time Driver Scan")
    st.write("This mode uses WebRTC to process your webcam stream directly in the web browser. Recommended for deployment.")
    
    col_video, col_metrics = st.columns([2, 1])
    
    with col_video:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        # WebRTC component
        ctx = webrtc_streamer(
            key="drowsiness-webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302", "stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                    {"urls": ["stun:stun.services.mozilla.com"]},
                    {"urls": ["stun:global.stun.twilio.com:3478"]}
                ]
            },
            video_processor_factory=DrowsinessProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_metrics:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.subheader("📊 Live Telemetry & Status")
        
        # Placeholders for dynamic content
        status_box = st.empty()
        confidence_box = st.empty()
        metrics_table_box = st.empty()
        
        st.markdown("---")
        st.subheader("📜 System Event Log")
        log_box = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Inject model into background processor if it starts playing
    if ctx.video_processor:
        if mode_selection == "Random Forest (ML)":
            ctx.video_processor.model = model_rf
            ctx.video_processor.le = le_rf
        else:
            ctx.video_processor.model = None
            ctx.video_processor.le = None
        ctx.video_processor.alarm_threshold = alarm_thresh
        
        # Run loop to pull metrics and update Streamlit UI in real-time
        last_prediction = "Alert"
        try:
            while ctx.state.playing:
                proc = ctx.video_processor
                if proc:
                    # Update dynamic UI metrics
                    prediction = proc.prediction
                    confidence = proc.confidence
                    ear = proc.ear
                    mar = proc.mar
                    blink_count = proc.blink_count
                    eye_duration = proc.eye_closure_duration
                    yawn_duration = proc.yawn_duration
                    face_present = proc.face_present
                    
                    # Update event log on state changes
                    if prediction != last_prediction:
                        if face_present:
                            log_event(f"State transitioned to: {prediction} (Conf: {confidence*100:.1f}%)")
                        else:
                            log_event("Face tracking lost.")
                        last_prediction = prediction
                        
                    # 1. Render Status card
                    if not face_present:
                        status_box.markdown(
                            '<div class="status-card drowsy-glow"><p class="glow-title">TRACKING LOST</p><p>No driver face detected</p></div>', 
                            unsafe_allow_html=True
                        )
                    elif prediction == "Drowsy" and confidence >= alarm_thresh:
                        status_box.markdown(
                            f'<div class="status-card drowsy-glow"><p class="glow-title">⚠️ DROWSY</p><p>Confidence: {confidence*100:.1f}%</p></div>', 
                            unsafe_allow_html=True
                        )
                    elif prediction == "Yawning":
                        status_box.markdown(
                            f'<div class="status-card yawn-glow"><p class="glow-title">🥱 YAWNING</p><p>Confidence: {confidence*100:.1f}%</p></div>', 
                            unsafe_allow_html=True
                        )
                    else:
                        status_box.markdown(
                            f'<div class="status-card awake-glow"><p class="glow-title">✅ ALERT (AWAKE)</p><p>Confidence: {confidence*100:.1f}%</p></div>', 
                            unsafe_allow_html=True
                        )
                        
                    # 2. Render Confidence Score bar
                    conf_val = int(confidence * 100) if face_present else 0
                    confidence_box.markdown(f"**Classification Confidence: {conf_val}%**")
                    
                    # 3. Play audio alarm if drowsy
                    if face_present and prediction == "Drowsy" and confidence >= alarm_thresh and not is_muted:
                        trigger_html_alarm()
                        
                    # 4. Render metrics
                    metrics_table_box.markdown(f"""
                    | Metric | Current Value | Standard Range |
                    | :--- | :---: | :---: |
                    | **EAR** (Eye Aspect) | `{ear:.3f}` | `0.22 - 0.35` |
                    | **MAR** (Mouth Aspect) | `{mar:.3f}` | `< 0.60` |
                    | **Blinks Count** | `{blink_count}` | `-` |
                    | **Closure Duration** | `{eye_duration:.2f}s` | `< 1.0s` |
                    | **Yawn Duration** | `{yawn_duration:.2f}s` | `< 1.5s` |
                    """)
                    
                    # 5. Render Log entries
                    log_html = "".join(f"<div>{log}</div>" for log in reversed(st.session_state.event_log))
                    log_box.markdown(f'<div class="log-panel">{log_html}</div>', unsafe_allow_html=True)
                    
                time.sleep(0.18)
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────
# TAB 2: Camera Snapshot Analyzer
# ─────────────────────────────────────────────────────────────
with tab_snapshot:
    st.markdown("### 📸 Driver Frame Snapshot Analyzer")
    st.write("Capture a single snapshot using your webcam to verify face landmarker values and classifier predictions.")
    
    col_snap_ctrl, col_snap_res = st.columns([1, 1])
    
    with col_snap_ctrl:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        img_file = st.camera_input("Capture Driver Photo")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_snap_res:
        if img_file is not None:
            # Convert uploaded image bytes to BGR frame
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            h, w, _ = img.shape
            
            # Process frame using detector
            detector = FaceLandmarkDetector()
            extractor = FeatureExtractor()
            
            landmarks = detector.detect(img)
            face_present = landmarks is not None
            
            feat = extractor.update(landmarks, h, w, face_present)
            
            # Predict
            prediction = "Alert"
            confidence = 1.0
            if face_present and model_rf is not None and mode_selection == "Random Forest (ML)":
                features = extractor.to_vector(feat)
                pred_idx = model_rf.predict(features)[0]
                proba = model_rf.predict_proba(features)[0]
                prediction = le_rf.inverse_transform([pred_idx])[0]
                confidence = float(proba.max())
            elif face_present:
                if feat["ear"] < 0.20:
                    prediction = "Drowsy"
                elif feat["mar"] > 0.65:
                    prediction = "Yawning"
            else:
                prediction = "Unknown"
                confidence = 0.0
                
            # Draw mesh
            annotated_img = detector.annotate(img.copy(), landmarks, prediction=prediction)
            
            # Close resources
            detector.close()
            
            # Display results
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.subheader("🔍 Analysis Results")
            
            if not face_present:
                st.error("❌ No face detected in the snapshot. Try adjusting your position or lighting.")
            else:
                if prediction == "Drowsy" and confidence >= alarm_thresh:
                    st.error(f"⚠️ Drowsiness Detected! Status: {prediction} ({confidence*100:.1f}%)")
                    if not is_muted:
                        trigger_html_alarm()
                elif prediction == "Yawning":
                    st.warning(f"🥱 Yawn Detected. Status: {prediction} ({confidence*100:.1f}%)")
                else:
                    st.success(f"✅ Driver is Awake and Alert! Status: {prediction} ({confidence*100:.1f}%)")
                    
                # Show annotated image
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Landmark Mesh Overlay", use_container_width=True)
                
                # Show features
                st.write("**Feature Details:**")
                st.write(f"- Eye Aspect Ratio (EAR): `{feat['ear']:.3f}`")
                st.write(f"- Mouth Aspect Ratio (MAR): `{feat['mar']:.3f}`")
                st.write(f"- Eye Closure Duration: `{feat['eye_closure_duration']:.2f}s`")
                st.write(f"- Yawn Duration: `{feat['yawn_duration']:.2f}s`")
            st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 3: Interactive Face Simulator
# ─────────────────────────────────────────────────────────────
with tab_simulator:
    st.markdown("### 🎮 Driver Face Geometric Simulator")
    st.write("Simulate facial landmarks and feature outputs to test the Random Forest model. Perfect for testing when a camera is unavailable.")
    
    col_sim_ctrl, col_sim_view = st.columns([1, 1])
    
    # Simulator parameters
    with col_sim_ctrl:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.subheader("🕹️ Simulation Inputs")
        
        sim_scenario = st.selectbox(
            "Quick Scenario Presets",
            ["Awake / Normal", "Blinking", "Drowsy (Eyes Closed)", "Yawning (Mouth Open)", "Manual Configuration"]
        )
        
        # Auto-configure based on preset
        if sim_scenario == "Awake / Normal":
            sim_ear = 0.31
            sim_mar = 0.15
            sim_closure = 0.0
            sim_yawn = 0.0
            sim_blinks = 12
        elif sim_scenario == "Blinking":
            sim_ear = 0.12
            sim_mar = 0.18
            sim_closure = 0.10
            sim_yawn = 0.0
            sim_blinks = 13
        elif sim_scenario == "Drowsy (Eyes Closed)":
            sim_ear = 0.09
            sim_mar = 0.14
            sim_closure = 2.40
            sim_yawn = 0.0
            sim_blinks = 6
        elif sim_scenario == "Yawning (Mouth Open)":
            sim_ear = 0.28
            sim_mar = 0.78
            sim_closure = 0.0
            sim_yawn = 3.20
            sim_blinks = 14
        else: # Manual control
            sim_ear = st.slider("Simulated EAR (Eye Opening)", 0.05, 0.40, 0.30, 0.01)
            sim_mar = st.slider("Simulated MAR (Mouth Opening)", 0.05, 1.00, 0.18, 0.01)
            sim_closure = st.slider("Simulated Closure Duration (sec)", 0.0, 5.0, 0.0, 0.1)
            sim_yawn = st.slider("Simulated Yawn Duration (sec)", 0.0, 5.0, 0.0, 0.1)
            sim_blinks = st.number_input("Simulated Blinks Count", min_value=0, max_value=100, value=12)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_sim_view:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.subheader("🖥️ Simulator Status & Rendering")
        
        # Predict on simulated inputs
        sim_features = np.array([[
            sim_ear,
            sim_mar,
            sim_blinks,
            sim_closure,
            sim_yawn,
            1.0  # face_present
        ]], dtype=np.float32)
        
        if model_rf is not None and mode_selection == "Random Forest (ML)":
            pred_idx = model_rf.predict(sim_features)[0]
            proba = model_rf.predict_proba(sim_features)[0]
            sim_pred = le_rf.inverse_transform([pred_idx])[0]
            sim_conf = float(proba.max())
        else:
            # Fallback heuristic
            if sim_ear < 0.20 and sim_closure > 1.2:
                sim_pred = "Drowsy"
            elif sim_mar > 0.65:
                sim_pred = "Yawning"
            else:
                sim_pred = "Alert"
            sim_conf = 1.0
            
        # Draw simulated face vectors
        sim_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Background Grid lines
        for x in range(0, 640, 30):
            cv2.line(sim_frame, (x, 0), (x, 480), (16, 22, 36), 1)
        for y in range(0, 480, 30):
            cv2.line(sim_frame, (0, y), (640, y), (16, 22, 36), 1)
            
        cx, cy = 320, 220
        # Draw face oval
        cv2.ellipse(sim_frame, (cx, cy), (110, 150), 0, 0, 360, (75, 85, 99), 2)
        # Nose bridge line
        cv2.line(sim_frame, (cx, cy - 35), (cx, cy + 30), (107, 114, 128), 2)
        
        # Draw simulated eyes
        eye_y_offset = -20
        eye_x_offset = 45
        rx = 22
        # Map EAR to vertical radius of eyes (between 2 and 12 pixels)
        ry = int(max(2, (sim_ear / 0.40) * 12))
        
        eye_col = (16, 185, 129) if sim_ear >= 0.20 else (239, 68, 68) # BGR
        # Draw left and right eyes
        cv2.ellipse(sim_frame, (cx - eye_x_offset, cy + eye_y_offset), (rx, ry), 0, 0, 360, eye_col, 2)
        cv2.ellipse(sim_frame, (cx + eye_x_offset, cy + eye_y_offset), (rx, ry), 0, 0, 360, eye_col, 2)
        
        # Draw simulated mouth
        # Map MAR to vertical radius of mouth
        my = int(max(3, (sim_mar / 1.0) * 35))
        mx = int(35 - (sim_mar / 1.0) * 5)
        mou_col = (245, 158, 11) if sim_mar > 0.65 else (59, 130, 246)
        cv2.ellipse(sim_frame, (cx, cy + 55), (mx, my), 0, 0, 360, mou_col, 3)
        
        # Render status alert
        if sim_pred == "Drowsy" and sim_conf >= alarm_thresh:
            cv2.rectangle(sim_frame, (20, 180), (620, 260), (0, 0, 239), -1)
            cv2.putText(sim_frame, "DROWSINESS DETECTED!", (60, 232),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            if not is_muted:
                trigger_html_alarm()
                
        # Status Card Layout
        if sim_pred == "Drowsy" and sim_conf >= alarm_thresh:
            st.markdown(
                f'<div class="status-card drowsy-glow"><p class="glow-title">⚠️ DROWSY</p><p>Confidence: {sim_conf*100:.1f}%</p></div>', 
                unsafe_allow_html=True
            )
        elif sim_pred == "Yawning":
            st.markdown(
                f'<div class="status-card yawn-glow"><p class="glow-title">🥱 YAWNING</p><p>Confidence: {sim_conf*100:.1f}%</p></div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="status-card awake-glow"><p class="glow-title">✅ ALERT (AWAKE)</p><p>Confidence: {sim_conf*100:.1f}%</p></div>', 
                unsafe_allow_html=True
            )
            
        # Draw face canvas
        st.image(cv2.cvtColor(sim_frame, cv2.COLOR_BGR2RGB), caption="Simulated Driver Face Vector", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
