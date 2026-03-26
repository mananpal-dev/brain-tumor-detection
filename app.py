import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from tensorflow.keras.models import load_model
from transformers import ViTForImageClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor AI System",
    page_icon="🧠",
    layout="wide"
)

IMG_SIZE = (224, 224)
CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

CNN_MODEL_PATH = r"C:\AD Project\brain_tumor_detection\brain_tumor_detection\my_brain_tumor_mobilenetv2.h5"
VIT_MODEL_PATH = r"C:\AD Project\brain_tumor_detection\brain_tumor_detection\best_vit_model.pth"

device = torch.device("cpu")

# ───────────────────────────────────────────────
# LOAD MODELS
# ───────────────────────────────────────────────
@st.cache_resource
def load_cnn():
    return load_model(CNN_MODEL_PATH, compile=False)

@st.cache_resource
def load_vit():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=device))
    model.eval()
    return model

@st.cache_resource
def load_nlp():
    texts = [
        "headache dizziness blurred vision",
        "severe headache memory loss confusion",
        "nausea seizures vision problem",
        "seizures vomiting and nausea",
        "hormone issues weight gain fatigue",
        "growth problems infertility hormonal",
        "no headache no tumor normal",
        "healthy normal no symptoms"
    ]
    labels = ["glioma","glioma","meningioma","meningioma",
              "pituitary","pituitary","notumor","notumor"]

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(texts)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, labels)

    return tfidf, clf

cnn_model = load_cnn()
vit_model = load_vit()
tfidf, text_clf = load_nlp()

# ───────────────────────────────────────────────
# PREPROCESSING
# ───────────────────────────────────────────────
def preprocess_cnn(img):
    img = img.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0), img

def preprocess_vit(img):
    img = img.resize((224,224)).convert("RGB")
    arr = np.array(img) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2,0,1))
    arr = np.expand_dims(arr, 0)
    return torch.tensor(arr).float()

# ───────────────────────────────────────────────
# GRAD-CAM
# ───────────────────────────────────────────────
def get_gradcam(img_array):
    try:
        last_conv = "Conv_1"
        grad_model = tf.keras.models.Model(
            [cnn_model.inputs],
            [cnn_model.get_layer(last_conv).output, cnn_model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        conv_out = conv_out[0]

        heatmap = tf.reduce_mean(conv_out * pooled, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-10

        return heatmap
    except:
        return None

def overlay_heatmap(heatmap, img):
    heatmap = np.uint8(255 * heatmap)
    heatmap = plt.cm.jet(heatmap)[:, :, :3] * 255
    superimposed = heatmap * 0.5 + np.array(img)
    return Image.fromarray(np.uint8(superimposed))

# ───────────────────────────────────────────────
# UI HEADER
# ───────────────────────────────────────────────
st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor AI Diagnosis System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>CNN vs Vision Transformer + NLP (Multimodal AI)</h4>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────
with st.sidebar:
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])
    symptoms_text = st.text_area("Enter Symptoms (optional)")
    show_gradcam = st.checkbox("Show Grad-CAM", True)

# ───────────────────────────────────────────────
# MAIN LAYOUT
# ───────────────────────────────────────────────
col1, col2 = st.columns([5,4])

if uploaded_file:

    try:
        with st.spinner("Analyzing MRI..."):

            pil_img = Image.open(uploaded_file)

            # CNN
            cnn_input, display_img = preprocess_cnn(pil_img)
            cnn_preds = cnn_model.predict(cnn_input)[0]
            cnn_idx = np.argmax(cnn_preds)
            cnn_label = CLASS_LABELS[cnn_idx]
            cnn_conf = cnn_preds[cnn_idx] * 100

            # ViT
            vit_input = preprocess_vit(pil_img)
            with torch.no_grad():
                vit_out = vit_model(vit_input).logits
                vit_preds = torch.softmax(vit_out, dim=1).numpy()[0]

            vit_idx = np.argmax(vit_preds)
            vit_label = CLASS_LABELS[vit_idx]
            vit_conf = vit_preds[vit_idx] * 100

            # NLP
            sym_pred = None
            if symptoms_text.strip():
                X_test = tfidf.transform([symptoms_text.lower()])
                sym_pred = text_clf.predict(X_test)[0]

        # ─── LEFT ───
        with col1:
            st.image(display_img, caption="Uploaded MRI", width=400)

            st.subheader("🧠 CNN Prediction")
            st.metric("Prediction", cnn_label, f"{cnn_conf:.2f}%")

            st.subheader("🤖 ViT Prediction")
            st.metric("Prediction", vit_label, f"{vit_conf:.2f}%")

            if sym_pred:
                st.subheader("📝 Symptoms Analysis")
                st.warning(sym_pred.capitalize())

            if show_gradcam:
                heatmap = get_gradcam(cnn_input)
                if heatmap is not None:
                    overlay = overlay_heatmap(heatmap, display_img)
                    st.image(overlay, caption="Grad-CAM (CNN Focus)", width=400)

        # ─── RIGHT ───
        with col2:
            st.subheader("📊 Model Comparison")

            fig, ax = plt.subplots()
            ax.barh(CLASS_LABELS, cnn_preds * 100, alpha=0.6, label="CNN")
            ax.barh(CLASS_LABELS, vit_preds * 100, alpha=0.6, label="ViT")
            ax.legend()
            ax.set_xlim(0, 100)
            ax.set_title("Prediction Confidence Comparison")

            st.pyplot(fig)

            st.subheader("📌 Final Insight")

            if cnn_label == vit_label:
                st.success("✅ Both models agree — High confidence prediction")
            else:
                st.warning("⚠️ Models disagree — Further analysis recommended")

            st.write(f"CNN: {cnn_label}")
            st.write(f"ViT: {vit_label}")

            if sym_pred:
                st.write(f"Symptoms: {sym_pred}")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload an MRI image to begin")

st.markdown("---")
st.caption("Final Year Project • CNN + GAN + NLP • Multimodal AI System")
