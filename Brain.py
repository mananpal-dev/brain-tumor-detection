"""
Brain.py — MobileNetV2 CNN Training Script
==========================================
Trains a 4-class brain tumour classifier (Glioma / Meningioma / No Tumor / Pituitary)
using MobileNetV2 transfer learning on the Kaggle Brain MRI dataset.

Dataset  : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Author   : Manan Pal  (B.Tech CSE, KIIT University)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
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
# Configuration  — edit these paths before running
# ---------------------------------------------------------------------------
TRAIN_DIR: str = os.environ.get("TRAIN_DIR", "Training")
TEST_DIR:  str = os.environ.get("TEST_DIR",  "Testing")

IMG_SIZE:    int   = 224
BATCH_SIZE:  int   = 32
NUM_EPOCHS:  int   = 40
NUM_CLASSES: int   = 4
LR:          float = 1e-4
DROPOUT:     float = 0.4

MODEL_PATH: str = "my_brain_tumor_mobilenetv2.h5"

# Canonical label order produced by flow_from_directory (alphabetical)
CLASS_LABELS: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
def build_generators(
    train_dir: str,
    test_dir: str,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
) -> tuple:
    """Return (train_generator, test_generator)."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.25,
        width_shift_range=0.18,
        height_shift_range=0.18,
        shear_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.2],
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,          # must stay False for correct label alignment
    )
    return train_gen, test_gen


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE) -> tf.keras.Model:
    """Build MobileNetV2 transfer-learning classifier.

    Architecture:
        MobileNetV2 (ImageNet weights, frozen)
        → GlobalAveragePooling2D
        → Dense(128, relu)
        → Dropout(0.4)
        → Dense(num_classes, softmax)
    """
    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False  # freeze pre-trained weights

    model = Sequential(
        [
            base,
            GlobalAveragePooling2D(),
            Dense(128, activation="relu"),
            Dropout(DROPOUT),
            Dense(num_classes, activation="softmax"),
        ],
        name="brain_tumor_mobilenetv2",
    )
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)
    return model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def build_callbacks(model_path: str = MODEL_PATH) -> list:
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
    ]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_history(history: tf.keras.callbacks.History, save_path: str = "training_curves.png") -> None:
    """Save accuracy and loss curves to disk."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    logger.info("Training curves saved → %s", save_path)
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str = "confusion_matrix.png",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info("Confusion matrix saved → %s", save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Validate paths
    for path in (TRAIN_DIR, TEST_DIR):
        if not Path(path).is_dir():
            raise FileNotFoundError(
                f"Directory not found: {path}\n"
                "Set TRAIN_DIR / TEST_DIR environment variables or edit the script."
            )

    logger.info("Building data generators …")
    train_gen, test_gen = build_generators(TRAIN_DIR, TEST_DIR)

    logger.info("Class → index mapping: %s", train_gen.class_indices)

    logger.info("Building model …")
    model = build_model()

    logger.info("Training …")
    history = model.fit(
        train_gen,
        epochs=NUM_EPOCHS,
        validation_data=test_gen,
        callbacks=build_callbacks(),
    )

    # Evaluation
    loss, accuracy = model.evaluate(test_gen)
    logger.info("Test  loss     : %.4f", loss)
    logger.info("Test  accuracy : %.2f%%", accuracy * 100)

    # Plots
    plot_history(history)

    y_pred_proba = model.predict(test_gen)
    y_pred  = np.argmax(y_pred_proba, axis=1)
    y_true  = test_gen.classes
    labels  = list(test_gen.class_indices.keys())

    logger.info("\n%s", classification_report(y_true, y_pred, target_names=labels))
    plot_confusion_matrix(y_true, y_pred, labels)


if __name__ == "__main__":
    main()
