import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import classification_report
from tqdm import tqdm

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# PATHS
# ======================
train_dir = r"C:\AD Project\brain_tumor_detection\Training"
test_dir  = r"C:\AD Project\brain_tumor_detection\Testing"

batch_size = 16   # Increase if GPU strong
num_epochs = 15
num_classes = 4

# ======================
# TRANSFORMS (VERY IMPORTANT FOR ViT)
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # ViT normalization
])

# ======================
# DATA LOADERS
# ======================
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ======================
# MODEL (PRETRAINED ViT)
# ======================
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

model.to(device)

# ======================
# LOSS & OPTIMIZER
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# Scheduler (IMPORTANT)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Mixed precision (FASTER)
scaler = torch.cuda.amp.GradScaler()

# ======================
# TRAINING
# ======================
best_acc = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images).logits
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()

        loop.set_postfix(loss=loss.item())

    train_acc = correct / len(train_dataset)

    # ======================
    # VALIDATION
    # ======================
    model.eval()
    val_correct = 0
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1)

            val_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())

    val_acc = val_correct / len(test_dataset)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_vit_model.pth")
        print("✅ Best model saved!")

    scheduler.step()

# ======================
# FINAL EVALUATION
# ======================
print("\nFinal Evaluation:\n")

print(classification_report(
    test_dataset.targets,
    all_preds,
    target_names=train_dataset.classes
))