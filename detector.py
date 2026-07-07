"""
detector.py
===========
Driver Drowsiness Detection (RF Edition) — MediaPipe Face Landmark Detector
---------------------------------------------------------------------------
Wraps MediaPipe Tasks FaceLandmarker into a clean, reusable class that
provides per-frame synchronous detection and colour-coded annotation.

PEP 8 compliant.  Docstrings follow Google style.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from typing import Optional, List, Any

from utils import (
    FACE_MODEL_PATH,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    MOUTH_IDX,
    download_face_model,
)


# ─────────────────────────────────────────────────────────────
# BGR colour palette
# ─────────────────────────────────────────────────────────────
_MESH_COLOR   = (0, 255, 200)      # teal — general face mesh dots
_EYE_COLOR    = (0, 230, 255)      # yellow-cyan — eye key-points
_MOUTH_COLOR  = (245, 165, 0)      # orange — mouth key-points
_ALERT_COLOR  = (0, 200, 80)       # green
_DROWSY_COLOR = (0, 0, 255)        # red
_YAWN_COLOR   = (0, 165, 255)      # amber

_PRED_COLORS = {
    "Alert":   _ALERT_COLOR,
    "Drowsy":  _DROWSY_COLOR,
    "Yawning": _YAWN_COLOR,
}


class FaceLandmarkDetector:
    """Synchronous, single-face MediaPipe FaceLandmarker wrapper.

    Uses ``RunningMode.IMAGE`` so every call to :meth:`detect` is
    independent (no timing / frame-ID requirement), making it suitable
    for both live-camera and simulated-frame scenarios.

    Example::

        detector = FaceLandmarkDetector()
        with detector:
            for frame in frames:
                landmarks = detector.detect(frame)
                detector.annotate(frame, landmarks, prediction="Drowsy")

    Can also be used without a context manager — call :meth:`close`
    manually when done.
    """

    def __init__(self) -> None:
        # Ensure the model file is present before building the landmarker
        download_face_model()

        base_options = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.50,
            min_face_presence_score=0.50,
            min_tracking_confidence=0.50,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    def detect(self, bgr_frame: Any) -> Optional[List[Any]]:
        """Runs face landmark detection on a single BGR OpenCV frame.

        Args:
            bgr_frame: ``numpy.ndarray`` in BGR format (standard OpenCV output).

        Returns:
            Flat list of 478 ``NormalizedLandmark`` objects if a face is
            found, otherwise ``None``.
        """
        # MediaPipe expects RGB input
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._landmarker.detect(mp_image)

        if result.face_landmarks and len(result.face_landmarks) > 0:
            return result.face_landmarks[0]
        return None

    # ------------------------------------------------------------------
    def annotate(
        self,
        frame: Any,
        landmarks: Optional[List[Any]],
        prediction: str = "Alert",
    ) -> Any:
        """Draws face mesh and key-point overlays on *frame* in-place.

        Overlay layers (back → front):
            1. Small teal dots at every one of the 478 face mesh points.
            2. Larger prediction-coloured dots at eye key-points.
            3. Orange dots at mouth key-points.

        Args:
            frame:      BGR ``numpy.ndarray`` — modified **in place**.
            landmarks:  Output of :meth:`detect`, or ``None``.
            prediction: Current class label used to choose the eye
                        highlight colour (``"Alert"``, ``"Drowsy"``,
                        ``"Yawning"``).

        Returns:
            The same *frame* array (for chaining convenience).
        """
        if landmarks is None:
            return frame

        h, w = frame.shape[:2]
        accent = _PRED_COLORS.get(prediction, _ALERT_COLOR)

        # Layer 1 — full face mesh (small 1-px dots)
        for lm in landmarks:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(frame, (cx, cy), 1, _MESH_COLOR, -1)

        # Layer 2 — eye key-points (2-px, accent colour)
        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            px = int(landmarks[idx].x * w)
            py = int(landmarks[idx].y * h)
            cv2.circle(frame, (px, py), 2, accent, -1)

        # Layer 3 — mouth key-points (2-px, orange)
        for idx in MOUTH_IDX:
            px = int(landmarks[idx].x * w)
            py = int(landmarks[idx].y * h)
            cv2.circle(frame, (px, py), 2, _MOUTH_COLOR, -1)

        return frame

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Releases MediaPipe landmarker resources.

        Safe to call even if the landmarker was never initialised.
        """
        try:
            self._landmarker.close()
        except Exception:
            pass

    # ── Context manager support ────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
