# Driver Drowsiness Detection System with Streamlit

A modern, high-performance web application designed for real-time driver drowsiness monitoring, built using Streamlit, MediaPipe, OpenCV, and Scikit-Learn.

This project uses facial landmarker geometry (Eye Aspect Ratio and Mouth Aspect Ratio) combined with a Random Forest machine learning classifier to detect if a driver is awake, drowsy, or yawning.

---

## 📁 Project Structure

```text
aiml/
│── streamlit_app.py          # Streamlit UI dashboard and app entrypoint
│── requirements.txt          # Python dependency specifications for deployment
│── random_forest_model.pkl   # Serialized Random Forest classifier [AUTO-GENERATED ON START]
│── alarm.wav                 # Wave audio warning siren sound [AUTO-GENERATED ON START]
│── face_landmarker.task      # MediaPipe FaceLandmarker model file
│── drowsiness_detector.py    # Standard heuristic detection script
│── driver_drowsiness_rf.py   # Standard Random Forest detection script
│── collect_dataset.py        # Dataset collector script
│── train_model.py            # Model training script
│── README.md                 # Project documentation
└── rf_app/                   # Modular sub-package for Random Forest classifier
    ├── detector.py           # Face Landmark Detector class wrapper
    ├── feature_extraction.py # Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) computations
    ├── utils.py              # Configuration constants, alarms, and download utilities
    └── train_model.py        # Sub-package model trainer (with synthetic fallback)
```

---

## 🌟 Key Features

1. **Live WebRTC Camera Stream**: Runs real-time camera inference inside the browser using WebRTC, making it compatible with Streamlit Community Cloud and cloud container environments.
2. **Camera Snapshot Analyzer**: Allows capturing a single driver photo to run static landmarker scans and display predictions.
3. **Interactive Face Simulator**: Simulates driver parameters (EAR, MAR, yawn duration, etc.) using custom sliders and presets (Awake, Blinking, Drowsy, Yawning) to render a vector face showing the classifier output.
4. **Auto-Training & Auto-Generation on Start**:
   - Automatically checks for `random_forest_model.pkl`. If missing, it immediately trains a classifier on synthetic data (incorporating alert, yawning, and closed-eye features) so the app is instantly usable.
   - Programmatically generates `alarm.wav` containing a standard ambulance two-tone PEE-PAW siren if it is missing.
5. **No-Stutter Browser Siren**: Uses a base64-encoded looping HTML5 `<audio>` player to play the warning siren directly in the client's browser, preventing stutters and working in container environments.
6. **Mute Control**: Easily mute the alarm audio via the sidebar checkbox.

---

## 🚀 Setup & Execution

### Prerequisites

Ensure you have Python 3.8+ installed.

### 1. Install Dependencies

Install all required libraries specified in the `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Note: The requirements specify `opencv-python-headless` instead of `opencv-python` to prevent GUI link errors in containerized Linux hosting environments (such as Streamlit Community Cloud).*

### 2. Run the Streamlit App

Run the following command in your terminal to start the local development server:

```bash
streamlit run streamlit_app.py
```

The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 🧠 Detection Logic & Model Details

The classifier evaluates the driver's state based on a 6-dimensional feature vector:
1. **EAR (Eye Aspect Ratio)**: Measure of eye opening (decreases when eyes close).
2. **MAR (Mouth Aspect Ratio)**: Measure of mouth opening (increases during yawns).
3. **Blinks Count**: Running tally of eye closures.
4. **Eye-Closure Duration**: Consecutive seconds the eyes remain closed.
5. **Yawn Duration**: Consecutive seconds the mouth remains open.
6. **Face Present**: Binary flag indicating if face detection tracking is successful.
