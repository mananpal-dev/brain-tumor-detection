import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import io
import time

# ───────────────────────────────────────────────
#   Config & Constants
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier • Academic Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

IMG_SIZE = (224, 224)
CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
MODEL_PATH = r"C:\AD Project\brain_tumor_detection\brain_tumor_detection\my_brain_tumor_mobilenetv2.h5"

# ───────────────────────────────────────────────
#   Load Models & Vectorizer (cached)
# ───────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH, compile=False)

@st.cache_resource
def prepare_text_clf():
    symptoms_texts = [
        "headache dizziness blurred vision", "severe headache memory loss confusion",
        "nausea seizures vision problem", "seizures vomiting and nausea",
        "hormone issues weight gain fatigue", "growth problems infertility hormonal",
        "no headache no tumor normal", "healthy normal no symptoms",
    ]
    symptoms_labels = ["glioma", "glioma", "meningioma", "meningioma",
                       "pituitary", "pituitary", "notumor", "notumor"]

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(symptoms_texts)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, symptoms_labels)
    return tfidf, clf

model = load_cnn_model()
tfidf, text_clf = prepare_text_clf()

# ───────────────────────────────────────────────
#   Grad-CAM (simple implementation)
# ───────────────────────────────────────────────
def get_gradcam_heatmap(img_array, model, pred_index=None, last_conv_layer_name="top_conv"):
    # If you have tf-keras-vis → preferred
    try:
        from tf_keras_vis.gradcam import Gradcam
        gradcam = Gradcam(model)
        cam = gradcam(lambda m: m(img_array), img_array, seek=pred_index)
        heatmap = tf.squeeze(cam)
        return heatmap.numpy()
    except:
        pass

    # Manual fallback Grad-CAM
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap + 1e-10)
    return heatmap

def superimpose_heatmap(heatmap, img, alpha=0.5):
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = np.repeat(heatmap, 3, axis=-1)
    heatmap_colored = plt.cm.jet(heatmap[..., 0])[:, :, :3] * 255
    superimposed = heatmap_colored * alpha + np.array(img) * (1 - alpha)
    return Image.fromarray(np.uint8(superimposed))

# ───────────────────────────────────────────────
#   Preprocessing
# ───────────────────────────────────────────────
def preprocess_image(pil_img):
    img = pil_img.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0), img

# ───────────────────────────────────────────────
#   UI Layout
# ───────────────────────────────────────────────
st.title("🧠 Brain Tumor Classification • Multimodal Academic Demo")
st.markdown(
    "MobileNetV2 CNN + Symptom-based Logistic Regression + **Grad-CAM explainability**"
)

with st.sidebar:
    st.header("Upload & Options")
    uploaded_file = st.file_uploader(
        "MRI Image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"]
    )
    symptoms_text = st.text_area(
        "Optional: Describe symptoms (helps refine interpretation)",
        placeholder="e.g. severe headache, nausea, blurred vision, seizures...",
        height=120
    )
    show_gradcam = st.checkbox("Show Grad-CAM explainability heatmap", value=True)
    st.markdown("---")
    st.caption("Academic project features:\n• Explainable AI (Grad-CAM)\n• Multimodal prediction\n• Confidence visualization")

main_col1, main_col2 = st.columns([5, 4])

if uploaded_file is not None:
    try:
        pil_img = Image.open(uploaded_file)
        with st.spinner("Preprocessing image..."):
            img_array, display_img = preprocess_image(pil_img)

        with st.spinner("Running CNN inference..."):
            time.sleep(0.4)  # simulate slight delay for better UX
            preds = model.predict(img_array, verbose=0)[0]
            pred_idx = np.argmax(preds)
            confidence = float(preds[pred_idx]) * 100
            label = CLASS_LABELS[pred_idx]

        # ── Symptoms analysis ───────────────────────────────
        nlp_result = "No symptoms provided"
        if symptoms_text.strip():
            X_test = tfidf.transform([symptoms_text.lower()])
            sym_pred = text_clf.predict(X_test)[0]
            nlp_result = f"**{sym_pred.capitalize()}** (symptom-based logistic regression)"

        # ── Display results ─────────────────────────────────
        with main_col1:
            st.subheader("Prediction")
            cols = st.columns(3)
            cols[0].metric("Predicted Class", label.upper(), delta=None)
            cols[1].metric("Confidence", f"{confidence:.1f}%")
            cols[2].metric("Symptoms Suggestion", nlp_result.split()[0] if "No" not in nlp_result else "—")

            st.image(display_img, caption="Uploaded MRI", use_column_width=True)

            if show_gradcam:
                with st.spinner("Generating Grad-CAM..."):
                    try:
                        heatmap = get_gradcam_heatmap(img_array, model, pred_idx)
                        overlay = superimpose_heatmap(heatmap, display_img, alpha=0.55)
                        st.image(overlay, caption="Grad-CAM: Model attention area", use_column_width=True)
                    except Exception as e:
                        st.warning("Could not generate Grad-CAM. Missing layer name or tf-keras-vis.")

        with main_col2:
            st.subheader("Confidence Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.barh(CLASS_LABELS, preds * 100, color="cornflowerblue")
            ax.set_xlabel("Confidence (%)")
            ax.set_xlim(0, 100)
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            st.pyplot(fig)

            with st.expander("Interpretation & Disclaimer"):
                st.markdown(
                    f"""
                    **CNN says**: Most likely **{label}** ({confidence:.1f}% confidence)
                    
                    **Symptoms model says**: {nlp_result}
                    
                    **Combined view**: {label} appears most probable. 
                    Symptoms {'align' if label.lower().startswith(sym_pred) else 'partially match / contradict'}.
                    
                    ⚠️ This is **NOT** a medical diagnosis. For research/educational use only.
                    """
                )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload an MRI image to begin.", icon="🖼️")

st.markdown("---")
st.caption("Academic features: Grad-CAM • Confidence bars • Multimodal fusion • Caching • Clean layout")