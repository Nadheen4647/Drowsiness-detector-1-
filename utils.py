"""
utils.py
========
Driver Drowsiness Detection (RF Edition) — Shared Utilities
------------------------------------------------------------
Provides all global constants, file-path definitions, model
serialisation helpers, face-model auto-download, and a
cross-platform alarm function used by the rest of the project.

PEP 8 compliant. Docstrings follow Google style.
"""

import os
import platform
import threading
import urllib.request
import logging
import joblib

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# File paths
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MediaPipe face-landmarker task model
FACE_MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")

# Trained Random Forest model (produced by train_model.py)
RF_MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")

# CSV dataset produced by collect_dataset.py (optional — synthetic data used if absent)
DATASET_PATH = os.path.join(BASE_DIR, "drowsiness_data.csv")

# ─────────────────────────────────────────────────────────────
# Remote asset URL
# ─────────────────────────────────────────────────────────────
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ─────────────────────────────────────────────────────────────
# MediaPipe landmark index groups
# ─────────────────────────────────────────────────────────────
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 385, 387, 362, 373, 380]
MOUTH_IDX = [61, 291, 37, 84, 0, 17, 267, 314]

# ─────────────────────────────────────────────────────────────
# Feature configuration
# ─────────────────────────────────────────────────────────────
# Column order that MUST match between training and inference
FEATURE_NAMES = [
    "ear",
    "mar",
    "blink_count",
    "eye_closure_duration",
    "yawn_duration",
    "face_present",
]

# ─────────────────────────────────────────────────────────────
# Thresholds (used ONLY for feature computation, NOT decisions)
# ─────────────────────────────────────────────────────────────
EAR_CLOSED_THRESH = 0.20   # EAR below this → eye closed
MAR_OPEN_THRESH = 0.65     # MAR above this → mouth open / yawning

# ─────────────────────────────────────────────────────────────
# Classification settings
# ─────────────────────────────────────────────────────────────
CLASS_LABELS = ["Alert", "Drowsy", "Yawning"]

# Minimum model confidence to trigger the drowsy alarm
ALARM_CONFIDENCE = 0.70

# Majority-vote smoothing window (frames)
SMOOTH_WINDOW = 7

# ─────────────────────────────────────────────────────────────
# Chart / history settings
# ─────────────────────────────────────────────────────────────
CHART_HISTORY = 120   # Number of frames stored in live EAR/MAR history


# ─────────────────────────────────────────────────────────────
# Face model download
# ─────────────────────────────────────────────────────────────
def download_face_model() -> None:
    """Downloads the MediaPipe face-landmarker .task file if absent.

    The file is ~3.7 MB and is saved to ``FACE_MODEL_PATH``.
    Progress is printed to stdout.  Safe to call multiple times —
    exits immediately if the file already exists.
    """
    if os.path.exists(FACE_MODEL_PATH):
        return

    logger.info("face_landmarker.task not found — downloading from Google Storage …")

    def _progress(block: int, block_size: int, total: int) -> None:
        downloaded = min(block * block_size, total)
        pct = int(downloaded * 100 / total) if total > 0 else 0
        print(f"\r  Downloading face_landmarker.task: {pct}%", end="", flush=True)

    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH, _progress)
    print()
    logger.info("Download complete.")


# ─────────────────────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────────────────────
def save_model(model, label_encoder, path: str = RF_MODEL_PATH) -> None:
    """Serialises the trained model and encoder to disk with joblib.

    Args:
        model: Fitted ``sklearn.ensemble.RandomForestClassifier``.
        label_encoder: Fitted ``sklearn.preprocessing.LabelEncoder``.
        path: Destination ``.pkl`` file path.
    """
    payload = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_names": FEATURE_NAMES,
        "class_labels": CLASS_LABELS,
    }
    joblib.dump(payload, path, compress=3)
    size_kb = os.path.getsize(path) // 1024
    logger.info(f"Model saved → '{path}'  ({size_kb} KB)")


def load_model(path: str = RF_MODEL_PATH):
    """Loads a serialised model payload from disk.

    Args:
        path: Path to the ``.pkl`` file created by :func:`save_model`.

    Returns:
        tuple: ``(RandomForestClassifier, LabelEncoder)``

    Raises:
        FileNotFoundError: If the file is absent at *path*.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: '{path}'.  "
            "Run train_model.py to generate it."
        )
    payload = joblib.load(path)
    logger.info(f"Model loaded ← '{path}'")
    return payload["model"], payload["label_encoder"]


# ─────────────────────────────────────────────────────────────
# Alarm
# ─────────────────────────────────────────────────────────────
_alarm_running = [False]   # mutable flag shared across threads
_alarm_lock = threading.Lock()


def play_alarm() -> None:
    """Plays a two-tone PEE-PAW siren in a background daemon thread.

    Behaviour:
        * **Windows** — uses ``winsound.Beep`` (950 Hz / 700 Hz, 3 cycles).
        * **Other platforms / cloud** — silently skipped; a visual banner
          in the UI serves as the alert instead.

    The function is re-entrant safe: a second call while the siren is
    already playing is ignored.
    """
    with _alarm_lock:
        if _alarm_running[0]:
            return
        _alarm_running[0] = True

    def _siren() -> None:
        try:
            if platform.system() == "Windows":
                import winsound  # noqa: PLC0415  (Windows-only import)
                for _ in range(3):
                    winsound.Beep(950, 400)   # PEE — high tone
                    winsound.Beep(700, 400)   # PAW — low tone
        except Exception:
            pass  # Silently ignore in non-Windows / sandboxed environments
        finally:
            with _alarm_lock:
                _alarm_running[0] = False

    threading.Thread(target=_siren, daemon=True).start()
