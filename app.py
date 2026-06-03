"""
app.py — Brain Tumor AI Diagnosis System (Streamlit)
=====================================================
Multimodal inference pipeline:
  1. CNN  (MobileNetV2)  — primary classification from MRI image
  2. ViT  (ViT-Base/16)  — independent classification for model comparison
  3. NLP  (TF-IDF + LR)  — supplementary symptom-based prediction
  4. Grad-CAM             — visual explanation of CNN focus regions

Run
---
    streamlit run app.py

Model files expected (relative paths, same directory as app.py):
    my_brain_tumor_mobilenetv2.h5
    best_vit_model.pth

Override via environment variables:
    CNN_MODEL_PATH=<path> VIT_MODEL_PATH=<path> streamlit run app.py

Author : Manan Pal  (B.Tech CSE, KIIT University)
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
from transformers import ViTForImageClassification

# ---------------------------------------------------------------------------
# Logging (Streamlit captures stdout, so use standard logging)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor AI System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE: tuple[int, int] = (224, 224)

# Class labels — MUST match the alphabetical order produced by
# flow_from_directory / ImageFolder:  glioma(0) meningioma(1) notumor(2) pituitary(3)
CLASS_LABELS: list[str] = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

CLASS_INFO: dict[str, str] = {
    "Glioma":     "Gliomas arise from glial cells and account for ~30% of all brain tumours.",
    "Meningioma": "Meningiomas grow from the meninges and are usually benign (~85%).",
    "No Tumor":   "No tumour detected in the scan.",
    "Pituitary":  "Pituitary tumours affect the pituitary gland; most are benign adenomas.",
}

# Model paths — resolved from environment variables with sensible defaults
CNN_MODEL_PATH: str = os.environ.get("CNN_MODEL_PATH", "my_brain_tumor_mobilenetv2.h5")
VIT_MODEL_PATH: str = os.environ.get("VIT_MODEL_PATH", "best_vit_model.pth")

device = torch.device("cpu")   # CPU-safe for Streamlit Cloud deployment


# ---------------------------------------------------------------------------
# Model loading  (cached so models load once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading CNN model…")
def load_cnn() -> tf.keras.Model | None:
    """Load MobileNetV2 classifier.  Returns None if file is missing."""
    path = Path(CNN_MODEL_PATH)
    if not path.is_file():
        st.warning(
            f"CNN model not found at `{CNN_MODEL_PATH}`.  "
            "Set the `CNN_MODEL_PATH` environment variable to point to "
            "`my_brain_tumor_mobilenetv2.h5`."
        )
        return None
    try:
        model = load_model(str(path), compile=False)
        logger.info("CNN model loaded from %s", path)
        return model
    except Exception as exc:
        st.error(f"Failed to load CNN model: {exc}")
        logger.exception("CNN load error")
        return None


@st.cache_resource(show_spinner="Loading ViT model…")
def load_vit() -> ViTForImageClassification | None:
    """Load ViT-Base/16 classifier.  Returns None if file is missing."""
    path = Path(VIT_MODEL_PATH)
    if not path.is_file():
        st.warning(
            f"ViT model not found at `{VIT_MODEL_PATH}`.  "
            "Set the `VIT_MODEL_PATH` environment variable to point to "
            "`best_vit_model.pth`."
        )
        return None
    try:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=4,
            ignore_mismatched_sizes=True,
        )
        state = torch.load(str(path), map_location=device)
        model.load_state_dict(state)
        model.eval()
        logger.info("ViT model loaded from %s", path)
        return model
    except Exception as exc:
        st.error(f"Failed to load ViT model: {exc}")
        logger.exception("ViT load error")
        return None


@st.cache_resource(show_spinner="Building NLP model…")
def load_nlp() -> tuple[TfidfVectorizer, LogisticRegression]:
    """Train a lightweight TF-IDF + Logistic Regression symptom classifier."""
    texts = [
        "headache dizziness blurred vision",
        "severe headache memory loss confusion seizures",
        "nausea seizures vision problem weakness",
        "seizures vomiting nausea focal deficit",
        "hormone issues weight gain fatigue infertility",
        "growth problems infertility hormonal imbalance",
        "no headache no symptoms healthy",
        "healthy normal no symptoms",
    ]
    labels = [
        "Glioma", "Glioma",
        "Meningioma", "Meningioma",
        "Pituitary", "Pituitary",
        "No Tumor", "No Tumor",
    ]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X     = tfidf.fit_transform(texts)
    clf   = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X, labels)
    return tfidf, clf


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_for_cnn(pil_img: Image.Image) -> np.ndarray:
    """Resize, convert to RGB, normalise to [0, 1].

    Returns shape (1, 224, 224, 3).
    """
    img = pil_img.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_for_vit(pil_img: Image.Image) -> torch.Tensor:
    """Resize, convert to RGB, normalise to [-1, 1] (ViT standard).

    Returns shape (1, 3, 224, 224) float32 tensor.
    NOTE: must be identical to the transform used in vit_model.py training.
    """
    img = pil_img.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1]
    arr = (arr - 0.5) / 0.5                          # → [-1, 1]
    arr = np.transpose(arr, (2, 0, 1))               # HWC → CHW
    return torch.from_numpy(arr).unsqueeze(0).float()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_cnn(
    model: tf.keras.Model,
    arr: np.ndarray,
) -> tuple[str, float, np.ndarray]:
    """Run CNN inference.

    Returns (predicted_label, confidence_pct, all_probabilities).
    """
    probs = model.predict(arr, verbose=0)[0]          # shape (4,)
    idx   = int(np.argmax(probs))
    return CLASS_LABELS[idx], float(probs[idx]) * 100, probs


def run_vit(
    model: ViTForImageClassification,
    tensor: torch.Tensor,
) -> tuple[str, float, np.ndarray]:
    """Run ViT inference.

    Returns (predicted_label, confidence_pct, all_probabilities).
    """
    with torch.no_grad():
        logits = model(tensor).logits                 # shape (1, 4)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx   = int(np.argmax(probs))
    return CLASS_LABELS[idx], float(probs[idx]) * 100, probs


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def compute_gradcam(
    model: tf.keras.Model,
    arr: np.ndarray,
    layer_name: str = "Conv_1",
) -> np.ndarray | None:
    """Compute Grad-CAM heatmap for the top predicted class.

    Returns a 2-D float array in [0, 1], or None on failure.
    """
    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(arr, training=False)
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]

        grads     = tape.gradient(loss, conv_out)          # (1, h, w, c)
        pooled    = tf.reduce_mean(grads, axis=(0, 1, 2))  # (c,)
        heatmap   = tf.reduce_mean(conv_out[0] * pooled, axis=-1)  # (h, w)
        heatmap   = np.maximum(heatmap.numpy(), 0)
        max_val   = heatmap.max()
        if max_val > 0:
            heatmap /= max_val
        return heatmap
    except Exception as exc:
        logger.warning("Grad-CAM failed: %s", exc)
        return None


def overlay_gradcam(heatmap: np.ndarray, pil_img: Image.Image) -> Image.Image:
    """Overlay Grad-CAM heatmap on *pil_img* and return composite PIL image."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    # Jet colormap → RGB
    jet_rgb = (plt.cm.jet(heatmap_uint8)[:, :, :3] * 255).astype(np.uint8)
    original = np.array(pil_img.resize(heatmap_uint8.shape[::-1]).convert("RGB"))
    blended  = (jet_rgb * 0.45 + original * 0.55).astype(np.uint8)
    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# Matplotlib confidence chart
# ---------------------------------------------------------------------------
def build_bar_chart(cnn_probs: np.ndarray, vit_probs: np.ndarray) -> plt.Figure:
    """Return a horizontal bar chart comparing CNN vs ViT confidences."""
    fig, ax = plt.subplots(figsize=(6, 3))
    y      = np.arange(len(CLASS_LABELS))
    height = 0.35

    ax.barh(y + height / 2, cnn_probs * 100, height, label="CNN",  alpha=0.82, color="#1976D2")
    ax.barh(y - height / 2, vit_probs * 100, height, label="ViT",  alpha=0.82, color="#E53935")
    ax.set_yticks(y)
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("CNN vs ViT Confidence Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------
cnn_model            = load_cnn()
vit_model            = load_vit()
tfidf, symptom_clf   = load_nlp()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🧠 Brain Tumor AI Diagnosis System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#666;'>"
    "CNN (MobileNetV2) · Vision Transformer · Grad-CAM · Symptom NLP"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file  = st.file_uploader(
        "Upload MRI Scan", type=["jpg", "jpeg", "png"]
    )
    symptoms_input = st.text_area(
        "Symptoms (optional)",
        placeholder="e.g. headache, blurred vision, nausea",
        height=80,
    )
    show_gradcam = st.checkbox("Show Grad-CAM overlay", value=True)

    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer**: This tool is for research and educational purposes only "
        "and must NOT be used for clinical diagnosis."
    )

    # Model status indicators
    st.markdown("**Model Status**")
    st.markdown(
        f"{'✅' if cnn_model else '❌'} CNN (MobileNetV2)\n\n"
        f"{'✅' if vit_model else '❌'} ViT-Base/16\n\n"
        "✅ NLP Symptom Classifier"
    )

# ── Main layout ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([5, 4], gap="large")

if uploaded_file is not None:
    try:
        pil_img = Image.open(uploaded_file)

        with st.spinner("Analysing MRI scan…"):
            # ── CNN ──────────────────────────────────────────────────────
            cnn_label = cnn_conf = cnn_probs = None
            if cnn_model is not None:
                cnn_arr                    = preprocess_for_cnn(pil_img)
                cnn_label, cnn_conf, cnn_probs = run_cnn(cnn_model, cnn_arr)
            else:
                cnn_arr = preprocess_for_cnn(pil_img)   # still needed for Grad-CAM

            # ── ViT ──────────────────────────────────────────────────────
            vit_label = vit_conf = vit_probs = None
            if vit_model is not None:
                vit_tensor                     = preprocess_for_vit(pil_img)
                vit_label, vit_conf, vit_probs = run_vit(vit_model, vit_tensor)

            # ── NLP ──────────────────────────────────────────────────────
            sym_pred = sym_probs = None
            if symptoms_input.strip():
                X_sym    = tfidf.transform([symptoms_input.lower()])
                sym_pred = symptom_clf.predict(X_sym)[0]
                sym_probs_arr = symptom_clf.predict_proba(X_sym)[0]
                sym_conf = float(max(sym_probs_arr)) * 100

        # ── Left column ──────────────────────────────────────────────────
        with col_left:
            display_img = pil_img.resize(IMG_SIZE).convert("RGB")
            st.image(display_img, caption="Uploaded MRI", use_container_width=True)

            if show_gradcam and cnn_model is not None:
                heatmap = compute_gradcam(cnn_model, cnn_arr)
                if heatmap is not None:
                    overlay = overlay_gradcam(heatmap, pil_img)
                    st.image(
                        overlay,
                        caption="Grad-CAM — CNN attention map",
                        use_container_width=True,
                    )
                else:
                    st.caption("Grad-CAM unavailable for this model.")

            # CNN result
            if cnn_label is not None:
                st.subheader("🔵 CNN Prediction")
                st.metric("Diagnosis", cnn_label, f"{cnn_conf:.1f}% confidence")
                with st.expander("CNN per-class probabilities"):
                    for lbl, p in zip(CLASS_LABELS, cnn_probs):
                        st.progress(float(p), text=f"{lbl}: {p*100:.1f}%")

            # ViT result
            if vit_label is not None:
                st.subheader("🔴 ViT Prediction")
                st.metric("Diagnosis", vit_label, f"{vit_conf:.1f}% confidence")
                with st.expander("ViT per-class probabilities"):
                    for lbl, p in zip(CLASS_LABELS, vit_probs):
                        st.progress(float(p), text=f"{lbl}: {p*100:.1f}%")

            # Symptom NLP
            if sym_pred is not None:
                st.subheader("📝 Symptom Analysis")
                st.info(f"Suggested class: **{sym_pred}** ({sym_conf:.1f}% confidence)")

        # ── Right column ─────────────────────────────────────────────────
        with col_right:
            if cnn_probs is not None and vit_probs is not None:
                st.subheader("📊 Model Comparison")
                fig = build_bar_chart(cnn_probs, vit_probs)
                st.pyplot(fig)
                plt.close(fig)

            st.subheader("📌 Final Verdict")
            if cnn_label is not None and vit_label is not None:
                if cnn_label == vit_label:
                    st.success(
                        f"✅ Both models agree: **{cnn_label}**\n\n"
                        "High confidence prediction."
                    )
                else:
                    st.warning(
                        f"⚠️ Models disagree.\n\n"
                        f"CNN → **{cnn_label}** ({cnn_conf:.1f}%)\n\n"
                        f"ViT → **{vit_label}** ({vit_conf:.1f}%)\n\n"
                        "Further clinical assessment recommended."
                    )
            elif cnn_label is not None:
                st.info(f"CNN prediction: **{cnn_label}** ({cnn_conf:.1f}%)")
            elif vit_label is not None:
                st.info(f"ViT prediction: **{vit_label}** ({vit_conf:.1f}%)")

            # Class info card
            primary_label = cnn_label or vit_label
            if primary_label and primary_label in CLASS_INFO:
                with st.expander("ℹ️ About this tumour type"):
                    st.write(CLASS_INFO[primary_label])

    except Exception as exc:
        st.error(f"An error occurred while processing the image: {exc}")
        logger.exception("Inference error")

else:
    with col_left:
        st.info("👈 Upload an MRI scan using the sidebar to begin analysis.")
        st.markdown(
            """
            **How to use:**
            1. Upload a brain MRI scan (JPG / PNG)
            2. Optionally enter symptoms for supplementary NLP analysis
            3. Enable Grad-CAM to visualise CNN attention
            4. Review CNN and ViT predictions side-by-side
            """
        )
    with col_right:
        st.markdown(
            """
            **Models in this system:**

            | Model | Architecture | Purpose |
            |-------|-------------|---------|
            | CNN   | MobileNetV2 | Primary classification |
            | ViT   | ViT-Base/16 | Validation & comparison |
            | NLP   | TF-IDF + LR | Symptom-based suggestion |
            """
        )

st.markdown("---")
st.caption(
    "Brain Tumor AI System · Manan Pal · B.Tech CSE · KIIT University · "
    "For educational and research purposes only."
)
