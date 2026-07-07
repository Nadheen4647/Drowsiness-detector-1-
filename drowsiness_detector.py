import cv2
import mediapipe as mp
import numpy as np
import time
import math
import winsound
import os
import urllib.request
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration Thresholds
EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.65
CONSECUTIVE_FRAMES_LIMIT = 45 # ~1.5 seconds at 30 FPS
YAWN_FRAMES_LIMIT = 90         # ~3.0 seconds at 30 FPS

# MediaPipe Landmark Indices for Eye Aspect Ratio (EAR)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 385, 387, 362, 373, 380]

# Mouth indices for Mouth Aspect Ratio (MAR)
MOUTH_OUTER = [61, 291, 37, 84, 0, 17, 267, 314]

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def download_model_if_missing():
    """Downloads the face landmarker task model file if it is not present in the workspace."""
    if not os.path.exists(MODEL_PATH):
        print(f"[SYSTEM] Face landmarker model file '{MODEL_PATH}' not found.")
        print(f"[SYSTEM] Downloading from Google Storage ({MODEL_URL})...")
        try:
            # Simple stream download with progress bar
            def report_progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                print(f"\rDownloading model: {percent}% completed", end="")
                
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, report_progress)
            print("\n[SYSTEM] Model download complete.")
        except Exception as e:
            print(f"\n[ERROR] Failed to download model: {e}")
            raise e

is_beeping = False
is_tracking_beeping = False

def play_alarm_sound():
    """Plays an Indian ambulance 'PEE-PAW' two-tone siren in the background."""
    global is_beeping
    if is_beeping:
        return
    is_beeping = True
    
    def beep_thread():
        global is_beeping
        try:
            # Indian ambulance PEE-PAW:
            #   PEE = high tone 950Hz for 450ms
            #   PAW = low tone  700Hz for 450ms
            # Repeat 3 full cycles = ~2.7 seconds of ambulance sound
            for _ in range(3):
                winsound.Beep(950, 450)   # PEE  (high tone, slightly rising feel)
                winsound.Beep(700, 450)   # PAW  (low tone, slightly falling feel)
        except Exception:
            pass
        finally:
            is_beeping = False
            
    threading.Thread(target=beep_thread, daemon=True).start()


def play_tracking_lost_alarm():
    """Plays a rapid, sharp warning beep sequence (3 quick pip-pip-pip beeps) for tracking lost."""
    global is_tracking_beeping
    if is_tracking_beeping:
        return
    is_tracking_beeping = True
    
    def tracking_thread():
        global is_tracking_beeping
        try:
            # 3 sharp warning pips (2400Hz)
            for _ in range(3):
                winsound.Beep(2400, 80)
                time.sleep(0.05)
        except Exception:
            pass
        finally:
            is_tracking_beeping = False
            
    threading.Thread(target=tracking_thread, daemon=True).start()


def get_distance(p1, p2):
    """Calculates Euclidean distance between two points in 2D space."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(eye_landmarks, shape):
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    h, w = shape
    # Convert normalized landmarks to pixel coordinates
    pts = [(int(pt.x * w), int(pt.y * h)) for pt in eye_landmarks]
    
    d_v1 = get_distance(pts[1], pts[5])
    d_v2 = get_distance(pts[2], pts[4])
    d_h = get_distance(pts[0], pts[3])
    
    ear = (d_v1 + d_v2) / (2.0 * d_h)
    return ear

def calculate_mar(mouth_landmarks, shape):
    """Calculates the Mouth Aspect Ratio (MAR) for yawn detection."""
    h, w = shape
    pts = [(int(pt.x * w), int(pt.y * h)) for pt in mouth_landmarks]
    
    d_v1 = get_distance(pts[2], pts[3]) # Index 37 to 84
    d_v2 = get_distance(pts[4], pts[5]) # Index 0 to 17
    d_v3 = get_distance(pts[6], pts[7]) # Index 267 to 314
    d_h = get_distance(pts[0], pts[1])  # Index 61 to 291
    
    mar = (d_v1 + d_v2 + d_v3) / (2.0 * d_h)
    return mar

def run_simulation():
    """Generates a simulated face using OpenCV draw utilities, calculating EAR/MAR and testing alarms."""
    print("[SYSTEM] Starting Python Face Simulation Mode...")
    print("[SYSTEM] Press 'q' in the window to exit.")
    
    eye_closed_counter = 0
    yawn_counter = 0
    
    prev_time = 0
    sim_timer = 0
    
    while True:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sim_timer += 1
        
        # Cybernetic grid background
        for x in range(0, 640, 30):
            cv2.line(frame, (x, 0), (x, 480), (15, 20, 10), 1)
        for y in range(0, 480, 30):
            cv2.line(frame, (0, y), (640, y), (15, 20, 10), 1)
            
        phase = sim_timer % 600
        
        # Preset cycles: Normal (0-200), Drowsy (200-400), Yawning (400-600)
        if phase < 200:
            scenario = "ALERT"
            sim_eye_open = 100
            sim_mouth_open = 5
            # blinking
            if (phase % 120) < 5:
                sim_eye_open = 10
        elif phase < 400:
            scenario = "DROWSY"
            drowsy_phase = phase - 200
            if drowsy_phase < 50:
                sim_eye_open = 100
            elif drowsy_phase < 80:
                sim_eye_open = max(25, 100 - (drowsy_phase - 50) * 3)
            elif drowsy_phase < 170:
                sim_eye_open = 5
            else:
                sim_eye_open = 100
            sim_mouth_open = 5
        else:
            scenario = "YAWNING"
            yawn_phase = phase - 400
            sim_eye_open = 95
            if yawn_phase < 30:
                sim_mouth_open = 5
            elif yawn_phase < 80:
                sim_mouth_open = int((yawn_phase - 30) * 1.8)
            elif yawn_phase < 150:
                sim_mouth_open = 90
                sim_eye_open = 20
            else:
                sim_mouth_open = max(5, 90 - (yawn_phase - 150) * 1.8)
                
        # Map percentages to aspect ratios
        ear = 0.08 + (sim_eye_open / 100.0) * 0.24
        mar = 0.12 + (sim_mouth_open / 100.0) * 0.66
        
        # Add micro-noise
        ear += (np.random.rand() - 0.5) * 0.012
        mar += (np.random.rand() - 0.5) * 0.012
        ear = max(0.02, ear)
        mar = max(0.02, mar)
        
        cx, cy = 320, 220
        # Draw face contours
        cv2.ellipse(frame, (cx, cy), (110, 150), 0, 0, 360, (60, 60, 60), 2)
        cv2.line(frame, (cx, cy - 35), (cx, cy + 35), (90, 90, 90), 2)
        
        # Draw Eyes
        eye_x = 45
        eye_y = -25
        rx = 22
        ry = int(1.5 + (sim_eye_open / 100.0) * 8.5)
        
        # Left eye
        cv2.ellipse(frame, (cx - eye_x, cy + eye_y), (rx, ry), 0, 0, 360, (255, 242, 0) if ear < EAR_THRESHOLD else (0, 255, 208), 2)
        if ry > 3:
            cv2.circle(frame, (cx - eye_x, cy + eye_y), min(ry - 1, 6), (154, 195, 2), -1)
            cv2.circle(frame, (cx - eye_x, cy + eye_y), min(ry - 2, 2), (0, 0, 0), -1)
            
        # Right eye
        cv2.ellipse(frame, (cx + eye_x, cy + eye_y), (rx, ry), 0, 0, 360, (255, 242, 0) if ear < EAR_THRESHOLD else (0, 255, 208), 2)
        if ry > 3:
            cv2.circle(frame, (cx + eye_x, cy + eye_y), min(ry - 1, 6), (154, 195, 2), -1)
            cv2.circle(frame, (cx + eye_x, cy + eye_y), min(ry - 2, 2), (0, 0, 0), -1)
            
        # Draw Mouth
        mouth_y = 55
        mx = int(35 - (sim_mouth_open / 100.0) * 8)
        my = int(3 + (sim_mouth_open / 100.0) * 32)
        cv2.ellipse(frame, (cx, cy + mouth_y), (mx, my), 0, 0, 360, (0, 158, 245) if mar < MAR_THRESHOLD else (245, 158, 11), 3)
        
        # Decision logic
        if ear < EAR_THRESHOLD:
            eye_closed_counter += 1
            if eye_closed_counter >= CONSECUTIVE_FRAMES_LIMIT:
                cv2.rectangle(frame, (30, 180), (610, 260), (0, 0, 255), -1)
                cv2.putText(frame, "DROWSINESS DETECTED!", (50, 235),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                play_alarm_sound()
        else:
            eye_closed_counter = 0
            
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= YAWN_FRAMES_LIMIT:
                cv2.putText(frame, "YAWN DETECTED - COFFEE BREAK!", (50, 420),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 158, 245), 2)
        else:
            yawn_counter = 0
            
        # FPS Calculation
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 30.0
        prev_time = current_time
        
        # UI overlays
        cv2.putText(frame, f"EAR: {ear:.2f} (Thresh: {EAR_THRESHOLD})", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f} (Thresh: {MAR_THRESHOLD})", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (540, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 208), 2)
        cv2.putText(frame, f"MODE: PYTHON SIMULATOR ({scenario})", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 208), 2)
        
        cv2.imshow("AI Driver Drowsiness Detector", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

def main():
    try:
        download_model_if_missing()
    except Exception:
        print("[FATAL] Cannot start drowsiness detector because model file is missing.")
        return
        
    print("[SYSTEM] Starting Real-Time Driver Drowsiness Detection...")
    print("[SYSTEM] Press 'q' in the camera window to exit.")
    
    # Configure options for the FaceLandmarker Tasks API
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    
    # Initialize Video Capture (Sweep camera indices and backends)
    cap = None
    for index in [0, 1, 2]:
        # DirectShow is highly reliable on Windows and boots faster
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[SYSTEM] Successfully opened camera index {index} via DirectShow.")
            break
        cap.release()
        
        # Fallback to default backend
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"[SYSTEM] Successfully opened camera index {index} via default backend.")
            break
        cap.release()
        cap = None
        
    if cap is None or not cap.isOpened():
        print("[ERROR] Could not open webcam source. Please verify:")
        print(" 1. A physical webcam is connected to your computer.")
        print(" 2. The camera is not currently being used by Zoom, Teams, or another application.")
        print(" 3. Windows Camera privacy settings allow desktop applications to access the camera.")
        print("\n[SYSTEM] Reverting to Python face simulation mode...")
        run_simulation()
        return
    
    # Set camera resolution properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # State tracking variables
    eye_closed_counter = 0
    yawn_counter = 0
    face_lost_counter = 0
    
    prev_time = 0
    empty_frame_count = 0
    
    # Use context manager to instantiate FaceLandmarker
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                empty_frame_count += 1
                if empty_frame_count > 30:
                    print("\n[ERROR] Camera stream failed or returned empty frames repeatedly.")
                    cap.release()
                    cv2.destroyAllWindows()
                    print("[SYSTEM] Reverting to Python face simulation mode...")
                    run_simulation()
                    return
                time.sleep(0.03)
                continue
            
            empty_frame_count = 0
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert BGR frame to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Run FaceLandmarker inference
            results = landmarker.detect(mp_image)
            
            ear = 0.30
            mar = 0.15
            face_detected = False
            
            if results.face_landmarks and len(results.face_landmarks) > 0:
                face_detected = True
                landmarks = results.face_landmarks[0]
                
                # Extract Eye Landmarks
                left_eye_lms = [landmarks[i] for i in LEFT_EYE]
                right_eye_lms = [landmarks[i] for i in RIGHT_EYE]
                
                # Calculate EARs and average
                ear_left = calculate_ear(left_eye_lms, (h, w))
                ear_right = calculate_ear(right_eye_lms, (h, w))
                ear = (ear_left + ear_right) / 2.0
                
                # Extract Mouth Landmarks
                mouth_lms = [landmarks[i] for i in MOUTH_OUTER]
                mar = calculate_mar(mouth_lms, (h, w))
                
                # Draw Face landmarks overlay
                for landmark in landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 208), -1)
                    
                # Outline eyes in cyan
                for i in LEFT_EYE + RIGHT_EYE:
                    pt = landmarks[i]
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (255, 242, 0), -1)
                    
                # Outline mouth in blue
                for i in MOUTH_OUTER:
                    pt = landmarks[i]
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 158, 245), -1)
            
            # Drowsiness Decision Logic
            if face_detected:
                face_lost_counter = 0
                if ear < EAR_THRESHOLD:
                    eye_closed_counter += 1
                    if eye_closed_counter >= CONSECUTIVE_FRAMES_LIMIT:
                        cv2.rectangle(frame, (30, 200), (610, 280), (0, 0, 255), -1)
                        cv2.putText(frame, "DROWSINESS DETECTED!", (50, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                        play_alarm_sound()
                else:
                    eye_closed_counter = 0
                    
                if mar > MAR_THRESHOLD:
                    yawn_counter += 1
                    if yawn_counter >= YAWN_FRAMES_LIMIT:
                        cv2.putText(frame, "YAWN DETECTED - COFFEE BREAK!", (50, 420),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 158, 245), 2)
                else:
                    yawn_counter = 0
            else:
                face_lost_counter += 1
                cv2.putText(frame, "TRACKING LOST - NO FACE DETECTED", (30, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                eye_closed_counter = 0
                yawn_counter = 0
                
                # If face is lost/blocked for more than ~1 second
                if face_lost_counter >= 30:
                    play_tracking_lost_alarm()

            # Calculate Frame Rate (FPS)
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 30.0
            prev_time = current_time
            
            # Overlay HUD Statistics
            cv2.putText(frame, f"EAR: {ear:.2f} (Thresh: {EAR_THRESHOLD})", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f} (Thresh: {MAR_THRESHOLD})", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (540, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 208), 2)
            
            # Display window
            cv2.imshow("AI Driver Drowsiness Detector", frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Camera closed. Execution completed.")

if __name__ == "__main__":
    main()
