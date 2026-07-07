"""
feature_extraction.py
=====================
Driver Drowsiness Detection (RF Edition) — Feature Extraction Module
----------------------------------------------------------------------
Provides stateless geometry functions (EAR, MAR) and a stateful
``FeatureExtractor`` class that tracks blink count, eye-closure
duration, and yawn duration across frames.

All computations are pure Python / NumPy; no OpenCV dependency.

PEP 8 compliant.  Docstrings follow Google style.
"""

import math
import time
import numpy as np
from typing import Optional, List, Any

from utils import (
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    MOUTH_IDX,
    EAR_CLOSED_THRESH,
    MAR_OPEN_THRESH,
    FEATURE_NAMES,
)


# ─────────────────────────────────────────────────────────────
# Stateless geometry helpers
# ─────────────────────────────────────────────────────────────

def landmark_distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two 2-D pixel points.

    Args:
        p1: ``(x, y)`` tuple (integer pixel coordinates).
        p2: ``(x, y)`` tuple (integer pixel coordinates).

    Returns:
        Non-negative float distance.
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_ear(landmarks: List[Any], indices: List[int],
                frame_h: int, frame_w: int) -> float:
    """Computes the Eye Aspect Ratio for a single eye.

    Formula::

        EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)

    where p0–p5 are the six eye key-points in order:
    outer corner, upper-inner, upper-outer,
    inner corner, lower-outer, lower-inner.

    Args:
        landmarks: Flat list of MediaPipe ``NormalizedLandmark`` objects.
        indices:   Six landmark indices for the target eye.
        frame_h:   Frame height in pixels (for de-normalisation).
        frame_w:   Frame width  in pixels (for de-normalisation).

    Returns:
        EAR value (float).  Returns ``0.0`` if the horizontal
        distance is zero (degenerate case).
    """
    pts = [
        (int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h))
        for i in indices
    ]
    vert_a = landmark_distance(pts[1], pts[5])
    vert_b = landmark_distance(pts[2], pts[4])
    horiz  = landmark_distance(pts[0], pts[3])
    return (vert_a + vert_b) / (2.0 * horiz) if horiz > 0 else 0.0


def compute_mar(landmarks: List[Any], indices: List[int],
                frame_h: int, frame_w: int) -> float:
    """Computes the Mouth Aspect Ratio for yawn detection.

    Formula::

        MAR = (||p2-p3|| + ||p4-p5|| + ||p6-p7||) / (2 * ||p0-p1||)

    where p0 = left corner, p1 = right corner, and p2-p7 are three
    vertical pairs along the mouth opening.

    Args:
        landmarks: Flat list of MediaPipe ``NormalizedLandmark`` objects.
        indices:   Eight landmark indices for the outer mouth contour.
        frame_h:   Frame height in pixels.
        frame_w:   Frame width  in pixels.

    Returns:
        MAR value (float).  Returns ``0.0`` if the horizontal
        distance is zero.
    """
    pts = [
        (int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h))
        for i in indices
    ]
    vert_a = landmark_distance(pts[2], pts[3])
    vert_b = landmark_distance(pts[4], pts[5])
    vert_c = landmark_distance(pts[6], pts[7])
    horiz  = landmark_distance(pts[0], pts[1])
    return (vert_a + vert_b + vert_c) / (2.0 * horiz) if horiz > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# Stateful feature extractor
# ─────────────────────────────────────────────────────────────

class FeatureExtractor:
    """Maintains per-session state and builds the 6-dimensional feature vector.

    State tracked across frames:
        * Blink count — increments on each rising edge of eye closure.
        * Eye-closure duration — cumulative seconds eyes stay closed.
        * Yawn duration — cumulative seconds mouth stays wide open.

    Example::

        extractor = FeatureExtractor()
        for frame in video_stream:
            landmarks, face_found = detect(frame)
            feat = extractor.update(landmarks, h, w, face_found)
            vec  = extractor.to_vector(feat)   # shape (1, 6)
    """

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Resets all state counters and timers to zero.

        Call this when switching between camera and simulation modes,
        or when the driver session restarts.
        """
        self.blink_count: int          = 0
        self.eye_closure_duration: float = 0.0
        self.yawn_duration: float       = 0.0

        self._eye_closed_start: Optional[float] = None
        self._yawn_start: Optional[float]        = None
        self._prev_eye_closed: bool              = False
        self._prev_yawning: bool                 = False

    # ------------------------------------------------------------------
    def update(
        self,
        landmarks: Optional[List[Any]],
        frame_h: int,
        frame_w: int,
        face_present: bool,
    ) -> dict:
        """Processes one frame and returns the feature dictionary.

        Args:
            landmarks:    MediaPipe landmark list from ``FaceLandmarkDetector``,
                          or ``None`` if no face was detected.
            frame_h:      Frame height in pixels.
            frame_w:      Frame width  in pixels.
            face_present: Whether a face was found in this frame.

        Returns:
            dict: Keys are ``FEATURE_NAMES``; values are Python scalars.
        """
        now = time.monotonic()

        # ── Compute EAR and MAR ───────────────────────────────────────
        if face_present and landmarks is not None:
            ear_left  = compute_ear(landmarks, LEFT_EYE_IDX,  frame_h, frame_w)
            ear_right = compute_ear(landmarks, RIGHT_EYE_IDX, frame_h, frame_w)
            ear = (ear_left + ear_right) / 2.0
            mar = compute_mar(landmarks, MOUTH_IDX, frame_h, frame_w)
        else:
            # Neutral fallback values when face is absent
            ear = 0.30
            mar = 0.15

        # ── Blink / eye-closure tracking ─────────────────────────────
        eye_closed_now = face_present and (ear < EAR_CLOSED_THRESH)

        if eye_closed_now and not self._prev_eye_closed:
            # Rising edge — new blink started
            self.blink_count += 1
            self._eye_closed_start = now

        if eye_closed_now and self._eye_closed_start is not None:
            self.eye_closure_duration = now - self._eye_closed_start
        elif not eye_closed_now:
            self.eye_closure_duration = 0.0
            self._eye_closed_start    = None

        self._prev_eye_closed = eye_closed_now

        # ── Yawn-duration tracking ────────────────────────────────────
        yawning_now = face_present and (mar > MAR_OPEN_THRESH)

        if yawning_now and not self._prev_yawning:
            # Rising edge — yawn started
            self._yawn_start = now

        if yawning_now and self._yawn_start is not None:
            self.yawn_duration = now - self._yawn_start
        elif not yawning_now:
            self.yawn_duration = 0.0
            self._yawn_start   = None

        self._prev_yawning = yawning_now

        # ── Assemble feature dictionary ───────────────────────────────
        return {
            "ear":                  round(ear, 5),
            "mar":                  round(mar, 5),
            "blink_count":          self.blink_count,
            "eye_closure_duration": round(self.eye_closure_duration, 4),
            "yawn_duration":        round(self.yawn_duration, 4),
            "face_present":         int(face_present),
        }

    # ------------------------------------------------------------------
    def to_vector(self, feature_dict: dict) -> np.ndarray:
        """Converts a feature dictionary to a model-ready 2-D NumPy array.

        Args:
            feature_dict: Dict returned by :meth:`update`.

        Returns:
            NumPy array of shape ``(1, 6)`` and dtype ``float32``,
            with columns in ``FEATURE_NAMES`` order.
        """
        return np.array(
            [[feature_dict[name] for name in FEATURE_NAMES]],
            dtype=np.float32,
        )
