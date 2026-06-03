# 🧠 Brain Tumor Detection — Multimodal AI System

> **MobileNetV2 CNN · Vision Transformer (ViT) · DCGAN Synthetic Augmentation · NLP Symptom Analyser**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-brightgreen?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Educational-lightgrey)](#license)

---

## Overview

End-to-end deep learning pipeline for **4-class brain tumour classification** from MRI scans.

The system combines three independent models for robust, cross-validated predictions:

| Component | Implementation | Role |
|-----------|---------------|------|
| **CNN** | MobileNetV2 (ImageNet transfer learning) | Primary classifier |
| **ViT** | `google/vit-base-patch16-224` (fine-tuned) | Independent validator |
| **GAN** | DCGAN per class, latent dim = 128 | Synthetic data augmentation |
| **NLP** | TF-IDF + Logistic Regression | Supplementary symptom analysis |

---

## Classes Detected

| Class | Description |
|-------|-------------|
| **Glioma** | Tumour arising from glial cells (~30% of brain tumours) |
| **Meningioma** | Grows from the meninges; usually benign |
| **Pituitary** | Affects the pituitary gland; most are benign adenomas |
| **No Tumor** | Normal MRI scan |

---

## Repository Structure

```
brain-tumor-detection/
│
├── Brain.py                       # CNN (MobileNetV2) training script
├── vit_model.py                   # Vision Transformer training script
├── mri_gan_generator.py           # DCGAN training & synthetic generation (local)
├── mri_gan_generator_colab.py     # DCGAN training & synthetic generation (Colab)
│
├── app.py                         # Streamlit web application
├── BrainGui.py                    # Tkinter desktop GUI (offline use)
│
├── metrics.py                     # Full CNN vs ViT evaluation & comparison
├── accuracy.py                    # Quick standalone CNN accuracy check
│
├── my_brain_tumor_mobilenetv2.h5  # Trained CNN weights
├── best_vit_model.pth             # Trained ViT weights
│
├── requirements.txt
├── .gitignore
└── README.md
```

Generated at runtime (not tracked in git):
```
synthetic_dataset/    # GAN-generated MRI images (500 per class)
gan_models/           # Saved GAN generators (.h5)
gan_preview/          # Training preview grids
gan_checkpoints/      # GAN training checkpoints
```

---

## Installation

```bash
git clone https://github.com/mananpal-dev/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt
```

Python **3.10 or 3.11** recommended.

---

## Dataset

**Source:** [Kaggle — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Split    | Folder     |
|----------|------------|
| Training | `Training/` |
| Testing  | `Testing/`  |

> The dataset is not included due to size. Download and place under the project root.

Expected directory layout after download:
```
Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

---

## Usage

All scripts resolve paths from **environment variables** first, then fall back to defaults in the current directory.

### 1 — GAN: Synthetic Data Generation

**Local:**
```bash
DATASET_PATH=Training python mri_gan_generator.py
```

**Google Colab** (edit `DATASET_PATH` inside the file to your Drive path):
```bash
python mri_gan_generator_colab.py
```

Outputs:
- `synthetic_dataset/<class>/synthetic_XXXX.png` — 500 × 224×224 images per class
- `gan_models/<class>_generator_final.h5`
- `gan_preview/<class>_epoch_XXXX.png`

### 2 — CNN Training

```bash
TRAIN_DIR=Training TEST_DIR=Testing python Brain.py
```

Key hyperparameters (edit at top of `Brain.py`):

| Parameter | Default |
|-----------|---------|
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 40 (early stopping) |
| Learning rate | 1e-4 |
| Augmentation | rotation, zoom, flip, brightness |

Best model saved to `my_brain_tumor_mobilenetv2.h5`.

### 3 — ViT Training

```bash
TRAIN_DIR=Training TEST_DIR=Testing python vit_model.py
```

Key hyperparameters:

| Parameter | Default |
|-----------|---------|
| Image size | 224 × 224 |
| Batch size | 16 |
| Epochs | 15 |
| Learning rate | 3e-5 (AdamW + CosineAnnealingLR) |
| Normalisation | mean=0.5, std=0.5 → [-1, 1] |

Best model saved to `best_vit_model.pth`.

### 4 — Evaluation

**Full comparison (CNN + ViT):**
```bash
TEST_DIR=Testing python metrics.py
```

Outputs: per-class classification report, confusion matrices, `model_comparison.png`.

**Quick CNN accuracy:**
```bash
TEST_DIR=Testing python accuracy.py
```

### 5 — Streamlit Web App

```bash
streamlit run app.py
```

Model paths can be set via environment:
```bash
CNN_MODEL_PATH=my_brain_tumor_mobilenetv2.h5 \
VIT_MODEL_PATH=best_vit_model.pth \
streamlit run app.py
```

Features:
- Upload MRI scan (JPG / PNG)
- CNN and ViT predictions with per-class confidence bars
- Grad-CAM heatmap overlay (CNN)
- Optional symptom-based NLP analysis
- Model agreement / disagreement indicator
- Tumour type information cards

### 6 — Desktop GUI (Tkinter)

```bash
CNN_MODEL_PATH=my_brain_tumor_mobilenetv2.h5 python BrainGui.py
```

Offline, no browser required.

---

## Model Architecture

### CNN — MobileNetV2

```
MobileNetV2 (ImageNet weights, frozen)
→ GlobalAveragePooling2D
→ Dense(128, relu)
→ Dropout(0.4)
→ Dense(4, softmax)
```

Trained with: EarlyStopping · ReduceLROnPlateau · ModelCheckpoint

### ViT — Vision Transformer

`google/vit-base-patch16-224` pretrained on ImageNet-21k, classification head replaced with `Linear(768 → 4)`.

Trained with: AdamW (lr=3e-5) · CosineAnnealingLR · mixed-precision AMP (when GPU available)

### GAN — DCGAN

| Component | Architecture |
|-----------|-------------|
| **Generator** | Dense → Reshape(8×8×512) → 3× Conv2DTranspose (256→128→64) → Conv2D(1, tanh) |
| **Discriminator** | 4× Conv2D (64→128→256→512) + Dropout(0.4) → Dense(1, sigmoid) |
| **Latent dim** | 128 |
| **Native resolution** | 64×64 grayscale → upsampled to 224×224 |
| **Label smoothing** | Real labels = 0.9 |

---

## Preprocessing Consistency

Critical: all three entry points use the same normalisation.

| Model | Script | Normalisation |
|-------|--------|--------------|
| CNN | `Brain.py`, `accuracy.py`, `metrics.py`, `app.py` | `/ 255.0` → [0, 1] |
| ViT | `vit_model.py`, `metrics.py`, `app.py` | `(/ 255.0 - 0.5) / 0.5` → [-1, 1] |

Any deviation between training and inference will silently degrade accuracy.

---

## Technical Notes on the ViT Result

The ViT model was trained via fine-tuning `google/vit-base-patch16-224` (pretrained on ImageNet-21k and ImageNet-1k). Several factors contribute to high performance on this dataset:

**Factors supporting high accuracy:**
- Strong ImageNet-21k pretraining provides rich low-level and mid-level features
- The Kaggle Brain MRI dataset is relatively clean and well-labelled
- 4-class classification with visually distinct MRI patterns is a tractable problem for ViT

**Factors that require scrutiny:**
- The Kaggle dataset's train/test split was provided by the dataset author; if any test images are visually similar to training images (common in MRI slice datasets), accuracy may be inflated
- 15 epochs of fine-tuning on a fixed test set (used as validation) could lead to implicit test-set overfitting during `best_acc` model selection
- The NLP symptom model uses a minimal 8-sample corpus and should not be treated as clinically meaningful

**Conclusion:** Results are plausible for a well-tuned ViT on this dataset, but should be validated on an independent held-out set before any claims of clinical relevance.

---

## Disclaimer

> ⚠️ This project is for **educational and research purposes only**.
> It must not be used as a substitute for professional medical diagnosis.
> Always consult a qualified healthcare professional for medical decisions.

---

## Author

**Manan Pal**  
B.Tech Computer Science & Engineering  
KIIT University

---

## License

This repository is provided for educational and portfolio purposes.
