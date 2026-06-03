"""
accuracy.py — Quick CNN Accuracy Check
=======================================
Loads the trained MobileNetV2 model and reports test accuracy, loss,
and a full classification report in a single command.

Usage
-----
    python accuracy.py

    # Override paths via environment variables:
    CNN_MODEL_PATH=weights/model.h5 TEST_DIR=Testing python accuracy.py

Author : Manan Pal  (B.Tech CSE, KIIT University)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CNN_MODEL_PATH: str = os.environ.get("CNN_MODEL_PATH", "my_brain_tumor_mobilenetv2.h5")
TEST_DIR:       str = os.environ.get("TEST_DIR",       "Testing")
IMG_SIZE: tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32

# Canonical label order (alphabetical — matches flow_from_directory output)
CLASS_NAMES: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]


def main() -> None:
    # ── Validate paths ───────────────────────────────────────────────────
    model_path = Path(CNN_MODEL_PATH)
    test_path  = Path(TEST_DIR)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {CNN_MODEL_PATH}\n"
            "Set the CNN_MODEL_PATH environment variable."
        )
    if not test_path.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}\n"
            "Set the TEST_DIR environment variable."
        )

    # ── Load model ───────────────────────────────────────────────────────
    logger.info("Loading model from %s …", model_path)
    model = load_model(str(model_path), compile=False)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Test data generator ──────────────────────────────────────────────
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_data    = test_datagen.flow_from_directory(
        str(test_path),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,     # must stay False for correct label alignment
    )

    # Confirm class order matches expected
    detected_classes = list(test_data.class_indices.keys())
    if detected_classes != CLASS_NAMES:
        logger.warning(
            "Class order mismatch!\n  Detected : %s\n  Expected : %s",
            detected_classes, CLASS_NAMES,
        )

    # ── Evaluate ─────────────────────────────────────────────────────────
    logger.info("Evaluating on %d images …", test_data.samples)
    loss, accuracy = model.evaluate(test_data, verbose=1)

    logger.info("=" * 40)
    logger.info("Test Loss     : %.4f", loss)
    logger.info("Test Accuracy : %.2f%%", accuracy * 100)
    logger.info("=" * 40)

    # ── Classification report ─────────────────────────────────────────────
    y_pred_proba = model.predict(test_data, verbose=1)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    y_true       = test_data.classes

    report = classification_report(
        y_true, y_pred,
        target_names=list(test_data.class_indices.keys()),
    )
    logger.info("\nClassification Report:\n%s", report)


if __name__ == "__main__":
    main()
