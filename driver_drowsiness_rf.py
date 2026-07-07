"""
driver_drowsiness_rf.py
=======================
Driver Drowsiness Detection — Random Forest Real-Time Predictor
---------------------------------------------------------------
Loads the pre-trained Random Forest model (random_forest_model.pkl),
extracts live EAR / MAR / blink / eye-closure / yawn features from
the webcam via MediaPipe Face Landmarker, and calls model.predict()
every frame to classify the driver state as:

    Alert | Drowsy | Yawning

Alarm fires ONLY when the model predicts "Drowsy" with confidence
above ALARM_CONFIDENCE_THRESHOLD.

Usage:
    python driver_drowsiness_rf.py

Prerequisites:
    1. Run collect_dataset.py to collect training data.
    2. Run train_model.py to produce random_forest_model.pkl.
    3. Then run this script for live detection.

Press  q  in the video window to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import threading
import winsound
import urllib.request
import joblib
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
MODEL_PATH   = "face_landmarker.task"
MODEL_URL    = ("https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
RF_MODEL_PATH = "random_forest_model.pkl"

# Alarm fires only when "Drowsy" confidence exceeds this value (0.0–1.0)
ALARM_CONFIDENCE_THRESHOLD = 0.70

# EAR / MAR thresholds — used ONLY for blink/closure/yawn tracking (not for decisions)
EAR_CLOSED_THRESHOLD = 0.20
MAR_YAWN_THRESHOLD   = 0.65

# MediaPipe landmark indices (same as original detector)
LEFT_EYE    = [33, 160, 158, 133, 153, 144]
RIGHT_EYE   = [263, 385, 387, 362, 373, 380]
MOUTH_OUTER = [61, 291, 37, 84, 0, 17, 267, 314]

# Smoothing window — prediction is the majority vote over last N frames
SMOOTH_WINDOW = 7


# ─────────────────────────────────────────────────────────────
# Audio alarms (same as original detector — Windows winsound)
# ─────────────────────────────────────────────────────────────
is_beeping          = False
is_tracking_beeping = False


def play_alarm_sound():
    """Plays Indian ambulance PEE-PAW siren in a daemon thread (3 cycles)."""
    global is_beeping
    if is_beeping:
        return
    is_beeping = True

    def _thread():
        global is_beeping
        try:
            for _ in range(3):
                winsound.Beep(950, 450)   # PEE — high tone
                winsound.Beep(700, 450)   # PAW — low tone
        except Exception:
            pass
        finally:
            is_beeping = False

    threading.Thread(target=_thread, daemon=True).start()


def play_tracking_lost_alarm():
    """Plays 3 sharp warning pips (2400 Hz) when face tracking is lost."""
    global is_tracking_beeping
    if is_tracking_beeping:
        return
    is_tracking_beeping = True

    def _thread():
        global is_tracking_beeping
        try:
            for _ in range(3):
                winsound.Beep(2400, 80)
                time.sleep(0.05)
        except Exception:
            pass
        finally:
            is_tracking_beeping = False

    threading.Thread(target=_thread, daemon=True).start()


# ─────────────────────────────────────────────────────────────
# Geometry helpers (identical to original detector)
# ─────────────────────────────────────────────────────────────
def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def calculate_ear(lms, indices, h, w):
    """Eye Aspect Ratio: (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)"""
    pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in indices]
    v1 = euclidean(pts[1], pts[5])
    v2 = euclidean(pts[2], pts[4])
    ho = euclidean(pts[0], pts[3])
    return (v1 + v2) / (2.0 * ho) if ho > 0 else 0.0


def calculate_mar(lms, indices, h, w):
    """Mouth Aspect Ratio: (||v1||+||v2||+||v3||) / (2 * ||h||)"""
    pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in indices]
    v1 = euclidean(pts[2], pts[3])
    v2 = euclidean(pts[4], pts[5])
    v3 = euclidean(pts[6], pts[7])
    ho = euclidean(pts[0], pts[1])
    return (v1 + v2 + v3) / (2.0 * ho) if ho > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading face landmarker model...")
        def _progress(count, block, total):
            print(f"\r  {int(count*block*100/total)}% complete", end="", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, _progress)
        print("\n[INFO] Download complete.")


def load_rf_model(path):
    """Loads the Random Forest payload dict from disk."""
    if not os.path.isfile(path):
        print(f"[ERROR] RF model not found: '{path}'")
        print("        Run train_model.py first.")
        return None, None
    payload = joblib.load(path)
    model   = payload["model"]
    le      = payload["label_encoder"]
    print(f"[INFO] Loaded RF model from '{path}'")
    print(f"       Classes: {list(le.classes_)}")
    return model, le


# ─────────────────────────────────────────────────────────────
# Simulation fallback (kept from original detector)
# ─────────────────────────────────────────────────────────────
def run_simulation(rf_model, le):
    """
    Runs the vector-based face simulator with RF prediction overlay.
    Mirrors the original drowsiness_detector.py simulation exactly,
    but uses the RF model for classification instead of thresholds.
    """
    print("[SYSTEM] No camera found — Starting Python Face Simulation Mode...")
    print("[SYSTEM] Press 'q' in the window to quit.")

    if rf_model is None:
        print("[WARNING] No RF model loaded — simulation uses threshold fallback.")

    eye_closed_counter   = 0
    blink_count          = 0
    eye_closure_start    = None
    eye_closure_duration = 0.0
    yawn_start           = None
    yawn_duration        = 0.0
    was_eye_closed       = False
    was_yawning          = False

    pred_queue = deque(maxlen=SMOOTH_WINDOW)
    prev_time  = 0
    sim_timer  = 0

    while True:
        frame     = np.zeros((480, 640, 3), dtype=np.uint8)
        sim_timer += 1

        # Grid background
        for x in range(0, 640, 30):
            cv2.line(frame, (x, 0), (x, 480), (15, 20, 10), 1)
        for y in range(0, 480, 30):
            cv2.line(frame, (0, y), (640, y), (15, 20, 10), 1)

        phase = sim_timer % 600
        if phase < 200:
            scenario = "ALERT"
            sim_eye_open  = 100
            sim_mouth_open = 5
            if (phase % 120) < 5:
                sim_eye_open = 10   # blink
        elif phase < 400:
            scenario = "DROWSY"
            dp = phase - 200
            if dp < 50:        sim_eye_open = 100
            elif dp < 80:      sim_eye_open = max(25, 100 - (dp-50)*3)
            elif dp < 170:     sim_eye_open = 5
            else:              sim_eye_open = 100
            sim_mouth_open = 5
        else:
            scenario = "YAWNING"
            yp = phase - 400
            sim_eye_open = 95
            if yp < 30:    sim_mouth_open = 5
            elif yp < 80:  sim_mouth_open = int((yp-30)*1.8)
            elif yp < 150: sim_mouth_open = 90; sim_eye_open = 20
            else:          sim_mouth_open = max(5, 90-(yp-150)*1.8)

        ear = 0.08 + (sim_eye_open  / 100.0) * 0.24 + (np.random.rand()-0.5)*0.012
        mar = 0.12 + (sim_mouth_open / 100.0) * 0.66 + (np.random.rand()-0.5)*0.012
        ear = max(0.02, ear)
        mar = max(0.02, mar)

        # Blink / closure tracking
        now            = time.time()
        eye_closed_now = ear < EAR_CLOSED_THRESHOLD
        if eye_closed_now and not was_eye_closed:
            blink_count      += 1
            eye_closure_start = now
        if eye_closed_now and eye_closure_start:
            eye_closure_duration = now - eye_closure_start
        elif not eye_closed_now:
            eye_closure_duration = 0.0
            eye_closure_start    = None
        was_eye_closed = eye_closed_now

        yawning_now = mar > MAR_YAWN_THRESHOLD
        if yawning_now and not was_yawning:
            yawn_start = now
        if yawning_now and yawn_start:
            yawn_duration = now - yawn_start
        elif not yawning_now:
            yawn_duration = 0.0
            yawn_start    = None
        was_yawning = yawning_now

        # Build feature vector and predict
        features   = np.array([[ear, mar, blink_count,
                                 eye_closure_duration, yawn_duration, 1.0]])
        prediction = scenario   # fallback
        confidence = 1.0

        if rf_model is not None:
            pred_idx   = rf_model.predict(features)[0]
            proba      = rf_model.predict_proba(features)[0]
            confidence = float(proba.max())
            prediction = le.inverse_transform([pred_idx])[0]
            pred_queue.append(prediction)
            # Smoothed majority vote
            if len(pred_queue) == SMOOTH_WINDOW:
                prediction = max(set(pred_queue), key=pred_queue.count)

        # Draw simulated face (same as original)
        cx, cy = 320, 220
        cv2.ellipse(frame, (cx, cy), (110, 150), 0, 0, 360, (60, 60, 60), 2)
        cv2.line(frame, (cx, cy-35), (cx, cy+35), (90, 90, 90), 2)

        eye_x, eye_y = 45, -25
        rx = 22
        ry = int(1.5 + (sim_eye_open/100.0)*8.5)
        eye_col = (0, 255, 242) if ear >= EAR_CLOSED_THRESHOLD else (0, 0, 255)
        for ex in [cx-eye_x, cx+eye_x]:
            cv2.ellipse(frame, (ex, cy+eye_y), (rx, ry), 0, 0, 360, eye_col, 2)
            if ry > 3:
                cv2.circle(frame, (ex, cy+eye_y), min(ry-1, 6), (154,195,2), -1)
                cv2.circle(frame, (ex, cy+eye_y), min(ry-2, 2), (0,0,0), -1)

        mx  = int(35 - (sim_mouth_open/100.0)*8)
        my  = int(3  + (sim_mouth_open/100.0)*32)
        mou_col = (0,158,245) if mar < MAR_YAWN_THRESHOLD else (245,158,0)
        cv2.ellipse(frame, (cx, cy+55), (mx, my), 0, 0, 360, mou_col, 3)

        # FPS
        current_time = time.time()
        fps = 1.0/(current_time-prev_time) if prev_time > 0 else 30.0
        prev_time = current_time

        _draw_hud(frame, ear, mar, fps, prediction, confidence,
                  eye_closure_duration, yawn_duration, blink_count,
                  mode_label=f"SIMULATOR ({scenario})")

        if prediction == "Drowsy" and confidence >= ALARM_CONFIDENCE_THRESHOLD:
            play_alarm_sound()

        cv2.imshow("AI Driver Drowsiness (RF)", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────
# HUD drawing helper
# ─────────────────────────────────────────────────────────────
def _draw_hud(frame, ear, mar, fps, prediction, confidence,
              eye_closure_duration, yawn_duration, blink_count,
              mode_label="CAMERA"):
    """
    Draws all on-screen overlays:
      - Prediction banner with colour coding
      - EAR / MAR / FPS stats
      - Eye closure and yawn duration counters
      - Confidence score bar
      - Mode label at the bottom
    """
    h, w = frame.shape[:2]

    # ── Prediction colour ────────────────────────────────────────────
    color_map = {
        "Alert":   (0, 200, 80),
        "Drowsy":  (0, 0, 255),
        "Yawning": (0, 165, 255)
    }
    pred_color = color_map.get(prediction, (200, 200, 200))

    # ── Drowsiness alarm banner ──────────────────────────────────────
    if prediction == "Drowsy" and confidence >= ALARM_CONFIDENCE_THRESHOLD:
        cv2.rectangle(frame, (20, h//2 - 45), (w-20, h//2 + 45), (0, 0, 200), -1)
        cv2.putText(frame, "DROWSINESS DETECTED!", (40, h//2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    # ── Yawn banner ──────────────────────────────────────────────────
    if prediction == "Yawning":
        cv2.putText(frame, "YAWN DETECTED — COFFEE BREAK!", (30, h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # ── Top-left stats ───────────────────────────────────────────────
    cv2.putText(frame, f"EAR: {ear:.3f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)
    cv2.putText(frame, f"MAR: {mar:.3f}", (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)
    cv2.putText(frame, f"Eye closed: {eye_closure_duration:.2f}s", (20, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(frame, f"Yawn dur:   {yawn_duration:.2f}s", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(frame, f"Blinks: {blink_count}", (20, 134),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    # ── FPS (top-right) ──────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {int(fps)}", (w-110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,208), 2)

    # ── Prediction pill (top-centre) ─────────────────────────────────
    label_text = f"RF: {prediction}  ({confidence*100:.1f}%)"
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    px = (w - tw) // 2
    cv2.rectangle(frame, (px-8, 8), (px+tw+8, 42), (30,30,30), -1)
    cv2.putText(frame, label_text, (px, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, pred_color, 2)

    # ── Confidence bar ───────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = px-8, 46, tw+16, 6
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
    fill = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), pred_color, -1)

    # ── Mode label (bottom-left) ─────────────────────────────────────
    cv2.putText(frame, f"MODE: {mode_label}", (20, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,208), 1)


# ─────────────────────────────────────────────────────────────
# Main real-time detection loop
# ─────────────────────────────────────────────────────────────
def main():
    download_model_if_missing()

    # Load Random Forest model
    rf_model, le = load_rf_model(RF_MODEL_PATH)

    # ── Open webcam ─────────────────────────────────────────────────
    cap = None
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[SYSTEM] Camera opened at index {idx} (DirectShow).")
            break
        cap.release()
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[SYSTEM] Camera opened at index {idx} (default backend).")
            break
        cap.release()
        cap = None

    if cap is None or not cap.isOpened():
        print("[SYSTEM] No webcam — reverting to face simulation mode.")
        run_simulation(rf_model, le)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[SYSTEM] Press 'q' to quit.")

    # ── Build MediaPipe FaceLandmarker ──────────────────────────────
    base_opts = python.BaseOptions(model_asset_path=MODEL_PATH)
    lm_opts   = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )

    # ── State variables ─────────────────────────────────────────────
    blink_count          = 0
    eye_closure_start    = None
    eye_closure_duration = 0.0
    yawn_start           = None
    yawn_duration        = 0.0
    was_eye_closed       = False
    was_yawning          = False
    face_lost_counter    = 0
    empty_frame_count    = 0

    pred_queue = deque(maxlen=SMOOTH_WINDOW)   # smoothing buffer
    prev_time  = 0

    with vision.FaceLandmarker.create_from_options(lm_opts) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()

            # Handle dropped frames
            if not ret:
                empty_frame_count += 1
                if empty_frame_count > 30:
                    print("\n[ERROR] Camera stream lost. Reverting to simulation.")
                    cap.release()
                    cv2.destroyAllWindows()
                    run_simulation(rf_model, le)
                    return
                time.sleep(0.03)
                continue

            empty_frame_count = 0
            frame = cv2.flip(frame, 1)        # mirror (natural selfie view)
            h, w, _ = frame.shape
            now = time.time()

            # ── MediaPipe inference ─────────────────────────────────
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = landmarker.detect(mp_img)

            ear           = 0.30
            mar           = 0.15
            face_detected = 0

            if res.face_landmarks and len(res.face_landmarks) > 0:
                face_detected = 1
                face_lost_counter = 0
                lms = res.face_landmarks[0]

                ear = (calculate_ear(lms, LEFT_EYE,  h, w) +
                       calculate_ear(lms, RIGHT_EYE, h, w)) / 2.0
                mar = calculate_mar(lms, MOUTH_OUTER, h, w)

                # ── Draw face mesh (identical to original detector) ─────
                for lm in lms:
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 1, (0,255,208), -1)
                for i in LEFT_EYE + RIGHT_EYE:
                    cv2.circle(frame, (int(lms[i].x*w), int(lms[i].y*h)), 2, (255,242,0), -1)
                for i in MOUTH_OUTER:
                    cv2.circle(frame, (int(lms[i].x*w), int(lms[i].y*h)), 2, (0,158,245), -1)
            else:
                # Face lost
                face_lost_counter += 1
                cv2.putText(frame, "TRACKING LOST — NO FACE DETECTED",
                            (30, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                if face_lost_counter >= 30:
                    play_tracking_lost_alarm()
                # Reset transient state when face disappears
                eye_closure_duration = 0.0
                eye_closure_start    = None
                yawn_duration        = 0.0
                yawn_start           = None

            # ── Blink / closure tracking ────────────────────────────
            eye_closed_now = (ear < EAR_CLOSED_THRESHOLD) and face_detected

            if eye_closed_now and not was_eye_closed:
                blink_count      += 1
                eye_closure_start = now

            if eye_closed_now and eye_closure_start:
                eye_closure_duration = now - eye_closure_start
            elif not eye_closed_now:
                eye_closure_duration = 0.0
                eye_closure_start    = None

            was_eye_closed = eye_closed_now

            # ── Yawn tracking ───────────────────────────────────────
            yawning_now = (mar > MAR_YAWN_THRESHOLD) and face_detected

            if yawning_now and not was_yawning:
                yawn_start = now

            if yawning_now and yawn_start:
                yawn_duration = now - yawn_start
            elif not yawning_now:
                yawn_duration = 0.0
                yawn_start    = None

            was_yawning = yawning_now

            # ── Random Forest prediction ────────────────────────────
            prediction = "Alert"
            confidence = 1.0

            if face_detected and rf_model is not None:
                features = np.array([[
                    ear,
                    mar,
                    blink_count,
                    eye_closure_duration,
                    yawn_duration,
                    float(face_detected)
                ]])
                pred_idx   = rf_model.predict(features)[0]
                proba      = rf_model.predict_proba(features)[0]
                raw_pred   = le.inverse_transform([pred_idx])[0]
                confidence = float(proba.max())

                # Push to smoothing queue and take majority vote
                pred_queue.append(raw_pred)
                prediction = max(set(pred_queue), key=pred_queue.count)

            elif rf_model is None and face_detected:
                # No model — simple threshold fallback with label
                if ear < EAR_CLOSED_THRESHOLD and eye_closure_duration > 1.5:
                    prediction = "Drowsy"
                elif mar > MAR_YAWN_THRESHOLD:
                    prediction = "Yawning"
                else:
                    prediction = "Alert"
                confidence = 1.0

            # ── Trigger alarm on Drowsy ─────────────────────────────
            if prediction == "Drowsy" and confidence >= ALARM_CONFIDENCE_THRESHOLD:
                play_alarm_sound()

            # ── FPS ─────────────────────────────────────────────────
            dt        = now - prev_time
            fps       = 1.0 / dt if dt > 0 else 30.0
            prev_time = now

            # ── Overlay all HUD elements ────────────────────────────
            _draw_hud(frame, ear, mar, fps, prediction, confidence,
                      eye_closure_duration, yawn_duration, blink_count,
                      mode_label="LIVE CAMERA")

            cv2.imshow("AI Driver Drowsiness (RF)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Camera closed. Execution complete.")


if __name__ == "__main__":
    main()

