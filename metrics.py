"""
metrics.py — CNN vs ViT Evaluation & Comparison
================================================
Loads both trained models and evaluates them on the held-out test set.
Produces:
  • Per-model accuracy
  • Full classification report (precision / recall / F1)
  • Confusion matrices
  • Side-by-side bar chart saved to ``model_comparison.png``

Usage
-----
    # Paths default to files in the current directory.
    # Override via environment variables:
    CNN_MODEL_PATH=<path> VIT_MODEL_PATH=<path> TEST_DIR=<path> python metrics.py

Author : Manan Pal  (B.Tech CSE, KIIT University)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

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
VIT_MODEL_PATH: str = os.environ.get("VIT_MODEL_PATH", "best_vit_model.pth")
TEST_DIR:       str = os.environ.get("TEST_DIR",       "Testing")

IMG_SIZE:   tuple[int, int] = (224, 224)
BATCH_SIZE: int             = 32

# Class labels — alphabetical order (matches flow_from_directory / ImageFolder)
CLASS_NAMES: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", device)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def build_cnn_test_generator(test_dir: str) -> tuple:
    """Return (test_generator, y_true) for the CNN evaluation."""
    datagen  = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    logger.info("CNN test generator — classes: %s", test_gen.class_indices)
    return test_gen, test_gen.classes


def build_vit_test_loader(test_dir: str) -> tuple:
    """Return (DataLoader, targets) for the ViT evaluation.

    NOTE: normalisation (mean=0.5, std=0.5) must be identical to
    training transforms in vit_model.py.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=min(4, os.cpu_count() or 1))
    logger.info("ViT test loader — classes: %s", dataset.classes)
    return loader, dataset.targets


# ---------------------------------------------------------------------------
# CNN evaluation
# ---------------------------------------------------------------------------
def evaluate_cnn(
    model_path: str,
    test_dir: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate CNN. Returns (accuracy, y_true, y_pred)."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CNN model not found: {model_path}\n"
            "Set CNN_MODEL_PATH environment variable."
        )

    logger.info("Loading CNN from %s …", path)
    model = load_model(str(path), compile=False)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    test_gen, y_true = build_cnn_test_generator(test_dir)
    loss, acc = model.evaluate(test_gen, verbose=1)
    logger.info("CNN  — loss: %.4f  accuracy: %.2f%%", loss, acc * 100)

    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    return acc, np.array(y_true), y_pred


# ---------------------------------------------------------------------------
# ViT evaluation
# ---------------------------------------------------------------------------
def evaluate_vit(
    model_path: str,
    test_dir: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate ViT. Returns (accuracy, y_true, y_pred).

    PREPROCESSING NOTE
    ------------------
    Images are normalised to [-1, 1] via (x - 0.5) / 0.5  (per-channel).
    This is the same normalisation used during training in vit_model.py and
    inference in app.py.  Any deviation would silently degrade accuracy.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"ViT model not found: {model_path}\n"
            "Set VIT_MODEL_PATH environment variable."
        )

    logger.info("Loading ViT from %s …", path)
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=4,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(str(path), map_location=device))
    model.to(device)
    model.eval()

    loader, targets = build_vit_test_loader(test_dir)
    y_true          = np.array(targets)
    all_preds: list[int] = []

    with torch.no_grad():
        for images, _ in loader:
            images  = images.to(device)
            logits  = model(images).logits
            preds   = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())

    y_pred = np.array(all_preds)
    acc    = float(np.mean(y_pred == y_true))
    logger.info("ViT  — accuracy: %.2f%%", acc * 100)
    return acc, y_true, y_pred


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    ax: plt.Axes,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)


def plot_comparison(cnn_acc: float, vit_acc: float, save_path: str = "model_comparison.png") -> None:
    """Save a bar chart comparing CNN and ViT accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # ── Accuracy bar chart ───────────────────────────────────────────────
    bars = axes[0].bar(
        ["CNN\n(MobileNetV2)", "ViT\n(ViT-Base/16)"],
        [cnn_acc * 100, vit_acc * 100],
        color=["#1976D2", "#E53935"],
        width=0.5,
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, acc in zip(bars, [cnn_acc, vit_acc]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{acc*100:.2f}%",
            ha="center", va="bottom", fontweight="bold",
        )
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("CNN vs ViT — Test Accuracy")

    # Confusion matrices filled in by caller
    axes[1].set_title("CNN Confusion Matrix")
    axes[2].set_title("ViT Confusion Matrix")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info("Comparison chart saved → %s", save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not Path(TEST_DIR).is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}\n"
            "Set the TEST_DIR environment variable."
        )

    results: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}

    # ── CNN ──────────────────────────────────────────────────────────────
    cnn_available = Path(CNN_MODEL_PATH).is_file()
    if cnn_available:
        cnn_acc, cnn_y_true, cnn_y_pred = evaluate_cnn(CNN_MODEL_PATH, TEST_DIR)
        results["CNN"] = (cnn_acc, cnn_y_true, cnn_y_pred)
        logger.info("\n%s", classification_report(
            cnn_y_true, cnn_y_pred, target_names=CLASS_NAMES
        ))
    else:
        logger.warning("CNN model not found at %s — skipping.", CNN_MODEL_PATH)

    # ── ViT ──────────────────────────────────────────────────────────────
    vit_available = Path(VIT_MODEL_PATH).is_file()
    if vit_available:
        vit_acc, vit_y_true, vit_y_pred = evaluate_vit(VIT_MODEL_PATH, TEST_DIR)
        results["ViT"] = (vit_acc, vit_y_true, vit_y_pred)
        logger.info("\n%s", classification_report(
            vit_y_true, vit_y_pred, target_names=CLASS_NAMES
        ))
    else:
        logger.warning("ViT model not found at %s — skipping.", VIT_MODEL_PATH)

    if not results:
        logger.error("No models were evaluated. Check model paths.")
        return

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 50)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 50)
    for name, (acc, _, _) in results.items():
        logger.info("%-6s accuracy : %.2f%%", name, acc * 100)

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 1 + len(results), figsize=(6 + 5 * len(results), 5))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Accuracy bar
    names = list(results.keys())
    accs  = [results[n][0] * 100 for n in names]
    colours = {"CNN": "#1976D2", "ViT": "#E53935"}

    bars = axes[0].bar(
        names, accs,
        color=[colours.get(n, "#4CAF50") for n in names],
        width=0.45, edgecolor="white", linewidth=1.2,
    )
    for bar, acc in zip(bars, accs):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontweight="bold",
        )
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Model Comparison")

    # Confusion matrices
    for i, (name, (_, y_true, y_pred)) in enumerate(results.items(), start=1):
        plot_confusion_matrix(y_true, y_pred, f"{name} Confusion Matrix", axes[i])

    plt.tight_layout()
    save_path = "model_comparison.png"
    plt.savefig(save_path, dpi=150)
    logger.info("Saved → %s", save_path)
    plt.show()


if __name__ == "__main__":
    main()
