"""
collect_dataset.py
==================
Driver Drowsiness Detection — Dataset Collection Tool
------------------------------------------------------
Captures webcam frames, computes EAR / MAR / blink count /
eye-closure duration / yawn duration via MediaPipe, and saves
labeled rows to  drowsiness_dataset.csv.

Controls:
    1  ->  Label = "Alert"
    2  ->  Label = "Drowsy"
    3  ->  Label = "Yawning"
    q  ->  Quit and save

Usage:
    python collect_dataset.py
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import csv
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
MODEL_PATH   = "face_landmarker.task"
MODEL_URL    = ("https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
DATASET_FILE = "drowsiness_dataset.csv"

EAR_CLOSED_THRESHOLD = 0.20   # eyes considered closed below this EAR
MAR_YAWN_THRESHOLD   = 0.65   # mouth considered yawning above this MAR

# MediaPipe facial landmark indices
LEFT_EYE    = [33, 160, 158, 133, 153, 144]
RIGHT_EYE   = [263, 385, 387, 362, 373, 380]
MOUTH_OUTER = [61, 291, 37, 84, 0, 17, 267, 314]

LABEL_MAP = {ord('1'): 'Alert', ord('2'): 'Drowsy', ord('3'): 'Yawning'}


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def download_model_if_missing():
    """Downloads the MediaPipe face landmarker model if not present locally."""
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading face landmarker model...")
        def _progress(count, block, total):
            print(f"\r  {int(count*block*100/total)}% complete", end="", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, _progress)
        print("\n[INFO] Download complete.")


def euclidean(p1, p2):
    """Returns Euclidean distance between two 2-D pixel points."""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def calculate_ear(lms, indices, h, w):
    """
    Eye Aspect Ratio (EAR):
        EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
    """
    pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in indices]
    v1 = euclidean(pts[1], pts[5])
    v2 = euclidean(pts[2], pts[4])
    ho = euclidean(pts[0], pts[3])
    return (v1 + v2) / (2.0 * ho) if ho > 0 else 0.0


def calculate_mar(lms, indices, h, w):
    """
    Mouth Aspect Ratio (MAR):
        MAR = (||v1|| + ||v2|| + ||v3||) / (2 * ||h||)
    """
    pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in indices]
    v1 = euclidean(pts[2], pts[3])
    v2 = euclidean(pts[4], pts[5])
    v3 = euclidean(pts[6], pts[7])
    ho = euclidean(pts[0], pts[1])
    return (v1 + v2 + v3) / (2.0 * ho) if ho > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────
def collect():
    download_model_if_missing()

    # Open webcam (try DirectShow first for Windows reliability)
    cap = None
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[INFO] Camera opened at index {idx} (DirectShow).")
            break
        cap.release()
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[INFO] Camera opened at index {idx} (default backend).")
            break
        cap.release()
        cap = None

    if cap is None or not cap.isOpened():
        print("[ERROR] No webcam found. Exiting.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Open CSV for appending
    file_exists = os.path.isfile(DATASET_FILE)
    csv_file    = open(DATASET_FILE, "a", newline="")
    writer      = csv.writer(csv_file)
    if not file_exists:
        # Write header only once
        writer.writerow(["ear", "mar", "blink_count",
                         "eye_closure_duration", "yawn_duration",
                         "face_detected", "label"])
        print(f"[INFO] Created new dataset: '{DATASET_FILE}'")
    else:
        print(f"[INFO] Appending to existing dataset: '{DATASET_FILE}'")

    # Build MediaPipe FaceLandmarker in IMAGE mode (synchronous per-frame)
    base_opts = python.BaseOptions(model_asset_path=MODEL_PATH)
    lm_opts   = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )

    # Stateful counters reset each session
    blink_count          = 0
    eye_closure_start    = None
    eye_closure_duration = 0.0
    yawn_start           = None
    yawn_duration        = 0.0
    was_eye_closed       = False
    was_yawning          = False

    current_label = "Alert"
    sample_count  = 0
    prev_time     = time.time()

    print("\n  CONTROLS:  1=Alert   2=Drowsy   3=Yawning   q=Quit\n")

    with vision.FaceLandmarker.create_from_options(lm_opts) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame    = cv2.flip(frame, 1)   # mirror so it feels natural
            h, w, _ = frame.shape
            now      = time.time()

            # ── MediaPipe inference ─────────────────────────────────
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = landmarker.detect(mp_img)

            ear           = 0.30
            mar           = 0.15
            face_detected = 0

            if res.face_landmarks and len(res.face_landmarks) > 0:
                face_detected = 1
                lms = res.face_landmarks[0]

                ear = (calculate_ear(lms, LEFT_EYE,  h, w) +
                       calculate_ear(lms, RIGHT_EYE, h, w)) / 2.0
                mar = calculate_mar(lms, MOUTH_OUTER, h, w)

                # Draw full face mesh (green)
                for lm in lms:
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 1, (0,255,208), -1)
                # Eye landmarks (yellow)
                for i in LEFT_EYE + RIGHT_EYE:
                    cv2.circle(frame, (int(lms[i].x*w), int(lms[i].y*h)), 2, (0,242,255), -1)
                # Mouth landmarks (orange)
                for i in MOUTH_OUTER:
                    cv2.circle(frame, (int(lms[i].x*w), int(lms[i].y*h)), 2, (245,158,0), -1)

            # ── Blink / eye-closure tracking ────────────────────────
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

            # ── FPS ─────────────────────────────────────────────────
            dt        = now - prev_time
            fps       = 1.0 / dt if dt > 0 else 30.0
            prev_time = now

            # ── HUD overlay ─────────────────────────────────────────
            label_color = {"Alert":(0,200,80), "Drowsy":(0,0,255),
                           "Yawning":(0,165,255)}.get(current_label,(200,200,200))

            cv2.rectangle(frame, (0,0), (640,42), (18,18,18), -1)
            cv2.putText(frame, f"LABEL: {current_label}   Samples: {sample_count}",
                        (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, label_color, 2)

            for n, txt in enumerate([
                f"EAR: {ear:.3f}",
                f"MAR: {mar:.3f}",
                f"Blinks: {blink_count}",
                f"Eye closed: {eye_closure_duration:.2f}s",
                f"Yawn dur:   {yawn_duration:.2f}s",
                f"Face: {'YES' if face_detected else 'NO'}"
            ]):
                cv2.putText(frame, txt, (10, 62+n*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)

            cv2.putText(frame, f"FPS: {int(fps)}", (560,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,208), 2)
            cv2.putText(frame, "1=Alert  2=Drowsy  3=Yawning  q=Quit",
                        (10,460), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160,160,160), 1)

            cv2.imshow("Dataset Collection — Driver Drowsiness", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in LABEL_MAP:
                current_label = LABEL_MAP[key]
                print(f"[LABEL] -> {current_label}")

            # Write a sample row for every face-detected frame
            if face_detected:
                writer.writerow([
                    round(ear,                4),
                    round(mar,                4),
                    blink_count,
                    round(eye_closure_duration, 4),
                    round(yawn_duration,        4),
                    face_detected,
                    current_label
                ])
                sample_count += 1

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"\n[DONE] Saved {sample_count} samples -> '{DATASET_FILE}'")


if __name__ == "__main__":
    collect()

