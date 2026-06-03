"""
vit_model.py — Vision Transformer (ViT) Training Script
========================================================
Fine-tunes google/vit-base-patch16-224 on the 4-class Brain MRI dataset.

Dataset  : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Author   : Manan Pal  (B.Tech CSE, KIIT University)

Notes
-----
* Preprocessing: resize to 224×224, normalize to [-1, 1]  (mean=0.5, std=0.5)
  — must be replicated identically in inference (app.py / metrics.py).
* Class ordering is determined by torchvision.datasets.ImageFolder (alphabetical):
  0=glioma  1=meningioma  2=notumor  3=pituitary
* Mixed-precision AMP is only enabled when a CUDA GPU is available.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
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
# Configuration  — edit these paths before running
# ---------------------------------------------------------------------------
TRAIN_DIR: str = os.environ.get("TRAIN_DIR", "Training")
TEST_DIR:  str = os.environ.get("TEST_DIR",  "Testing")

BATCH_SIZE:  int   = 16
NUM_EPOCHS:  int   = 15
NUM_CLASSES: int   = 4
LR:          float = 3e-5
MODEL_PATH:  str   = "best_vit_model.pth"

# Canonical class order (must match ImageFolder alphabetical order)
CLASS_NAMES: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
# ViT expects pixel values in [-1, 1] (mean=0.5, std=0.5 per channel).
# This MUST be identical to the preprocessing in app.py and metrics.py.
vit_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),                              # [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5],         # → [-1, 1]
                             std=[0.5, 0.5, 0.5]),
    ]
)

# Augmented transforms for training
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def build_loaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, datasets.ImageFolder]:
    """Return (train_loader, test_loader, test_dataset)."""
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=vit_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=device.type == "cuda",
    )
    return train_loader, test_loader, test_dataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int = NUM_CLASSES) -> ViTForImageClassification:
    """Load pretrained ViT-Base/16 and replace the classification head."""
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: ViTForImageClassification,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images).logits
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images).logits
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct      += (logits.argmax(1) == labels).sum().item()
        total        += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: ViTForImageClassification,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, np.ndarray]:
    """Evaluate model. Returns (avg_loss, accuracy, predictions_array)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0
    all_preds: list[int] = []

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images).logits
        loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
        all_preds.extend(preds.cpu().tolist())

    return running_loss / total, correct / total, np.array(all_preds)


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

    logger.info("Building data loaders …")
    train_loader, test_loader, test_dataset = build_loaders(TRAIN_DIR, TEST_DIR)
    logger.info("Classes: %s", train_loader.dataset.classes)

    logger.info("Building ViT model …")
    model = build_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info("Epoch %d / %d", epoch, NUM_EPOCHS)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, use_amp
        )
        val_loss, val_acc, val_preds = evaluate(model, test_loader, criterion)

        logger.info(
            "  train loss=%.4f  acc=%.4f | val loss=%.4f  acc=%.4f",
            train_loss, train_acc, val_loss, val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info("  ✅ Best model saved (val_acc=%.4f)", best_val_acc)

        scheduler.step()

    # Final evaluation with best checkpoint
    logger.info("\nLoading best checkpoint for final evaluation …")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    _, final_acc, final_preds = evaluate(model, test_loader, criterion)
    logger.info("Final test accuracy: %.2f%%", final_acc * 100)

    y_true = np.array(test_dataset.targets)
    logger.info(
        "\n%s",
        classification_report(y_true, final_preds, target_names=CLASS_NAMES),
    )


if __name__ == "__main__":
    main()
