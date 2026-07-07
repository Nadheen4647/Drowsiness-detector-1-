"""
train_model.py
==============
Driver Drowsiness Detection (RF Edition) — Random Forest Training Script
-------------------------------------------------------------------------
Workflow
--------
1. Attempt to load a real labelled dataset from ``drowsiness_data.csv``
   (created by a separate data-collection tool).
2. If no real dataset is found, generate a realistic synthetic one so
   the rest of the project is immediately usable for testing.
3. Pre-process: encode labels, optionally balance with SMOTE.
4. Train a ``RandomForestClassifier`` with stratified cross-validation.
5. Evaluate on a held-out test split and print a full report.
6. Save the model as ``random_forest_model.pkl`` via :func:`utils.save_model`.

Usage::

    python train_model.py                  # uses CSV if present, else synthetic
    python train_model.py --synthetic      # forces synthetic data generation
    python train_model.py --csv my_data.csv

PEP 8 compliant.  Docstrings follow Google style.
"""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
)
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (
    accuracy_score, classification_report, confusion_matrix,
)
from sklearn.utils           import shuffle

# Optional SMOTE for class balancing
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_OK = True
except ImportError:
    _SMOTE_OK = False

from utils import (
    DATASET_PATH, RF_MODEL_PATH,
    FEATURE_NAMES, CLASS_LABELS,
    save_model,
)

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
# Hyper-parameters
# ─────────────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":      200,
    "max_depth":         None,       # grow fully (pure leaves)
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "max_features":      "sqrt",
    "class_weight":      "balanced",
    "random_state":      42,
    "n_jobs":            -1,         # use all CPU cores
}

TEST_RATIO  = 0.20    # 20 % held-out test split
CV_FOLDS    = 5       # stratified k-fold cross-validation
N_SYNTHETIC = 1500    # synthetic samples per class


# ─────────────────────────────────────────────────────────────
# Synthetic dataset generator
# ─────────────────────────────────────────────────────────────
def generate_synthetic_dataset(n_per_class: int = N_SYNTHETIC) -> pd.DataFrame:
    """Generates a realistic synthetic drowsiness dataset.

    Each class is characterised by physiologically plausible ranges:

    * **Alert** — normal EAR (0.25–0.35), closed mouth, minimal closure.
    * **Drowsy** — low EAR (0.08–0.21), prolonged closure > 1 s.
    * **Yawning** — normal EAR, high MAR (0.60–0.90), long yawn duration.

    Gaussian noise is added to every sample to simulate sensor variation.

    Args:
        n_per_class: Number of samples to generate per class.

    Returns:
        ``pandas.DataFrame`` with ``FEATURE_NAMES`` columns plus ``label``.
    """
    rng = np.random.RandomState(42)
    rows: list = []

    # ── Alert ──────────────────────────────────────────────────────────
    for _ in range(n_per_class):
        ear = rng.uniform(0.25, 0.35)
        # Simulate brief blinks (5 % of frames)
        if rng.random() < 0.05:
            ear = rng.uniform(0.08, 0.19)
        rows.append({
            "ear":                  ear + rng.normal(0, 0.008),
            "mar":                  rng.uniform(0.10, 0.30) + rng.normal(0, 0.01),
            "blink_count":          int(rng.randint(5, 25)),
            "eye_closure_duration": rng.uniform(0, 0.12) if ear < 0.20 else 0.0,
            "yawn_duration":        0.0,
            "face_present":         1,
            "label":                "Alert",
        })

    # ── Drowsy ─────────────────────────────────────────────────────────
    for _ in range(n_per_class):
        ear     = rng.uniform(0.08, 0.21) + rng.normal(0, 0.008)
        closure = rng.uniform(1.2, 6.0) if ear < 0.20 else rng.uniform(0, 0.3)
        rows.append({
            "ear":                  max(0.03, ear),
            "mar":                  rng.uniform(0.10, 0.28) + rng.normal(0, 0.01),
            "blink_count":          int(rng.randint(0, 10)),
            "eye_closure_duration": closure,
            "yawn_duration":        rng.uniform(0, 0.5),
            "face_present":         1,
            "label":                "Drowsy",
        })

    # ── Yawning ────────────────────────────────────────────────────────
    for _ in range(n_per_class):
        mar = rng.uniform(0.60, 0.92) + rng.normal(0, 0.01)
        rows.append({
            "ear":                  rng.uniform(0.20, 0.32) + rng.normal(0, 0.008),
            "mar":                  max(0.30, mar),
            "blink_count":          int(rng.randint(5, 25)),
            "eye_closure_duration": rng.uniform(0, 0.25),
            "yawn_duration":        rng.uniform(1.2, 6.0),
            "face_present":         1,
            "label":                "Yawning",
        })

    df = pd.DataFrame(rows)
    df[FEATURE_NAMES] = df[FEATURE_NAMES].clip(lower=0.0)   # no negative values
    return df


# ─────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────
def load_dataset(csv_path: str) -> pd.DataFrame:
    """Loads a CSV dataset and validates its schema.

    Args:
        csv_path: Path to the ``.csv`` file.

    Returns:
        Cleaned ``DataFrame`` with required columns.

    Raises:
        SystemExit: If required columns are missing.
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset: {len(df)} rows from '{csv_path}'")

    required = FEATURE_NAMES + ["label"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Dataset is missing columns: {missing}")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=required)
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with NaN values.")

    return df


# ─────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """Encodes labels and optionally applies SMOTE.

    Args:
        df: DataFrame with ``FEATURE_NAMES`` columns and a ``label`` column.

    Returns:
        tuple: ``(X, y, label_encoder)``
            * ``X`` — float32 NumPy array shape ``(n, 6)``.
            * ``y`` — int NumPy array of encoded class indices.
            * ``label_encoder`` — fitted ``LabelEncoder``.
    """
    X_raw = df[FEATURE_NAMES].values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    logger.info("Class distribution before balancing:")
    for cls, cnt in zip(le.classes_, np.bincount(y)):
        logger.info(f"  {cls:10s}: {cnt:>6d} samples")

    # SMOTE — balance minority classes if available
    if _SMOTE_OK:
        min_count = min(np.bincount(y))
        k_neighbors = min(5, min_count - 1)
        if k_neighbors >= 1:
            try:
                sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_raw, y = sm.fit_resample(X_raw, y)
                logger.info("SMOTE applied. New class distribution:")
                for cls, cnt in zip(le.classes_, np.bincount(y)):
                    logger.info(f"  {cls:10s}: {cnt:>6d} samples")
            except ValueError as exc:
                logger.warning(f"SMOTE skipped ({exc}). Using class_weight='balanced'.")
    else:
        logger.info("imbalanced-learn not installed — skipping SMOTE.")

    X_raw, y = shuffle(X_raw, y, random_state=42)
    return X_raw, y, le


# ─────────────────────────────────────────────────────────────
# Training and evaluation
# ─────────────────────────────────────────────────────────────
def train_and_evaluate(X: np.ndarray, y: np.ndarray, le: LabelEncoder):
    """Trains a Random Forest and evaluates it on a held-out test split.

    Prints cross-validation scores, test accuracy, classification report,
    confusion matrix, and feature importances.

    Args:
        X:  Feature matrix ``(n, 6)`` float32.
        y:  Integer label array ``(n,)``.
        le: Fitted ``LabelEncoder`` (used for display).

    Returns:
        Fitted ``RandomForestClassifier``.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=42, stratify=y
    )
    logger.info(f"Split — train: {len(X_train)}, test: {len(X_test)}")

    # ── Stratified cross-validation ───────────────────────────────────
    logger.info(f"Running {CV_FOLDS}-fold stratified CV …")
    cv_clf    = RandomForestClassifier(**RF_PARAMS)
    skf       = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_clf, X_train, y_train, cv=skf, scoring="accuracy")
    logger.info(
        f"CV Accuracy: {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})"
    )

    # ── Final training ─────────────────────────────────────────────────
    logger.info("Training final Random Forest …")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  TEST ACCURACY : {accuracy * 100:.2f}%")
    print(f"{sep}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("Confusion Matrix  (rows = actual, cols = predicted):")
    cm     = confusion_matrix(y_test, y_pred)
    header = "          " + "  ".join(f"{c:8s}" for c in le.classes_)
    print(header)
    for row_label, row in zip(le.classes_, cm):
        print(f"  {row_label:8s}" + "  ".join(f"{v:8d}" for v in row))

    print("\nFeature Importances:")
    importance_pairs = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda t: t[1],
        reverse=True,
    )
    for feat, imp in importance_pairs:
        bar = "#" * int(imp * 50)
        print(f"  {feat:25s}: {imp:.4f}  |{bar}")

    return model


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Driver Drowsiness Random Forest classifier."
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Force synthetic data generation (ignores any CSV file).",
    )
    parser.add_argument(
        "--csv", metavar="PATH", default=DATASET_PATH,
        help=f"Path to labelled CSV dataset (default: {DATASET_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point — orchestrates dataset loading, training, and saving."""
    args = _parse_args()

    print("=" * 56)
    print("  Driver Drowsiness RF — Model Training")
    print("=" * 56 + "\n")

    # ── Decide data source ─────────────────────────────────────────────
    if args.synthetic or not os.path.exists(args.csv):
        if not args.synthetic:
            logger.warning(
                f"Dataset not found at '{args.csv}'. "
                "Generating synthetic data instead."
            )
        logger.info(f"Generating synthetic dataset ({N_SYNTHETIC} samples / class) …")
        df = generate_synthetic_dataset(N_SYNTHETIC)
    else:
        df = load_dataset(args.csv)

    logger.info(f"Total samples: {len(df)}")

    # ── Pre-process → train → evaluate → save ─────────────────────────
    X, y, le = preprocess(df)
    model     = train_and_evaluate(X, y, le)
    save_model(model, le)

    print(f"\n[DONE] Model saved to '{RF_MODEL_PATH}'")
    print("       Run  streamlit run app.py  to launch the dashboard.\n")


if __name__ == "__main__":
    main()
