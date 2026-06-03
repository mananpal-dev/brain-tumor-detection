"""
BrainGui.py — Brain Tumor Classifier (Tkinter Desktop GUI)
===========================================================
Lightweight desktop application for offline inference using the trained
MobileNetV2 CNN model and an optional TF-IDF + Logistic Regression
symptom analyser.

Usage
-----
    python BrainGui.py

Model path is resolved from:
  1. CNN_MODEL_PATH environment variable
  2. Default: my_brain_tumor_mobilenetv2.h5  (same directory)

Author : Manan Pal  (B.Tech CSE, KIIT University)
"""

from __future__ import annotations

import logging
import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE: tuple[int, int] = (224, 224)

# Class labels — alphabetical order (must match flow_from_directory)
CLASS_LABELS: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]

# Human-readable display names for each class
DISPLAY_LABELS: dict[str, str] = {
    "glioma":     "Glioma",
    "meningioma": "Meningioma",
    "notumor":    "No Tumor",    # .capitalize() would give "Notumor" — wrong
    "pituitary":  "Pituitary",
}

MODEL_PATH: str = os.environ.get("CNN_MODEL_PATH", "my_brain_tumor_mobilenetv2.h5")

# Colours
BG_DARK   = "#1E2A3A"
BG_CARD   = "#263545"
ACCENT    = "#2196F3"
ACCENT2   = "#4CAF50"
WARNING   = "#FF9800"
ERROR_COL = "#F44336"
FG_LIGHT  = "#ECEFF1"
FG_MUTED  = "#90A4AE"


# ---------------------------------------------------------------------------
# NLP model (trained at startup — tiny corpus, loads instantly)
# ---------------------------------------------------------------------------
def build_nlp_model() -> tuple[TfidfVectorizer, LogisticRegression]:
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
        "glioma", "glioma",
        "meningioma", "meningioma",
        "pituitary", "pituitary",
        "notumor", "notumor",
    ]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X     = tfidf.fit_transform(texts)
    clf   = LogisticRegression(max_iter=500)
    clf.fit(X, labels)
    return tfidf, clf


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess an MRI image for CNN inference.

    Returns shape (1, 224, 224, 3) normalised to [0, 1].
    """
    img = Image.open(image_path).resize(IMG_SIZE).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class TumorApp:
    """Tkinter GUI for brain tumour detection."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("🧠 Brain Tumor Classifier")
        self.root.geometry("720x780")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        # State
        self._current_image_path: str | None = None
        self._cnn_model = None
        self.tfidf, self.nlp_clf = build_nlp_model()

        self._build_ui()
        self._load_model_async()

    # ── Model loading ────────────────────────────────────────────────────
    def _load_model_async(self) -> None:
        """Load CNN model in a background thread to keep UI responsive."""
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self) -> None:
        path = Path(MODEL_PATH)
        if not path.is_file():
            self.root.after(0, lambda: self._set_status(
                f"⚠️ Model not found: {MODEL_PATH}", WARNING
            ))
            logger.error("Model not found: %s", MODEL_PATH)
            return

        self.root.after(0, lambda: self._set_status("Loading model…", FG_MUTED))
        try:
            self._cnn_model = load_model(str(path), compile=False)
            self.root.after(0, lambda: self._set_status("✅ Model loaded. Ready.", ACCENT2))
            logger.info("CNN model loaded from %s", path)
        except Exception as exc:
            msg = f"❌ Failed to load model: {exc}"
            self.root.after(0, lambda: self._set_status(msg, ERROR_COL))
            logger.exception("Model load error")

    # ── UI construction ──────────────────────────────────────────────────
    def _build_ui(self) -> None:
        # ── Title bar ────────────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg=BG_DARK)
        title_frame.pack(fill="x", padx=20, pady=(20, 5))

        tk.Label(
            title_frame,
            text="🧠 Brain Tumor Detection System",
            font=("Segoe UI", 20, "bold"),
            bg=BG_DARK,
            fg=FG_LIGHT,
        ).pack()
        tk.Label(
            title_frame,
            text="MobileNetV2 CNN + NLP Symptom Analyser",
            font=("Segoe UI", 10),
            bg=BG_DARK,
            fg=FG_MUTED,
        ).pack()

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=20, pady=10)

        # ── Image display card ───────────────────────────────────────────
        img_card = tk.Frame(self.root, bg=BG_CARD, relief="flat", bd=0)
        img_card.pack(padx=20, pady=5, fill="x")

        self.img_label = tk.Label(img_card, bg=BG_CARD, height=10)
        self.img_label.pack(pady=10)

        # ── Buttons row ──────────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, bg=BG_DARK)
        btn_frame.pack(pady=8)

        self.browse_btn = tk.Button(
            btn_frame,
            text="📂  Browse MRI Image",
            command=self._on_browse,
            font=("Segoe UI", 12, "bold"),
            bg=ACCENT, fg="white",
            activebackground="#1565C0", activeforeground="white",
            padx=16, pady=8, bd=0, cursor="hand2",
        )
        self.browse_btn.pack(side="left", padx=6)

        self.clear_btn = tk.Button(
            btn_frame,
            text="🔄  Clear",
            command=self._reset_ui,
            font=("Segoe UI", 11),
            bg=BG_CARD, fg=FG_LIGHT,
            activebackground="#37474F", activeforeground=FG_LIGHT,
            padx=12, pady=8, bd=0, cursor="hand2",
        )
        self.clear_btn.pack(side="left", padx=6)

        # ── Progress bar ─────────────────────────────────────────────────
        self.progress = ttk.Progressbar(self.root, mode="indeterminate", length=300)
        self.progress.pack(pady=4)
        self.progress.pack_forget()

        # ── CNN result box ───────────────────────────────────────────────
        result_card = tk.Frame(self.root, bg=BG_CARD, relief="flat")
        result_card.pack(padx=20, pady=5, fill="x")

        tk.Label(
            result_card, text="CNN Result", font=("Segoe UI", 11, "bold"),
            bg=BG_CARD, fg=FG_MUTED
        ).pack(anchor="w", padx=12, pady=(8, 0))

        self.result_label = tk.Label(
            result_card,
            text="—",
            font=("Segoe UI", 16, "bold"),
            bg=BG_CARD, fg=FG_LIGHT,
            wraplength=660, justify="left",
        )
        self.result_label.pack(anchor="w", padx=12, pady=(2, 10))

        # ── Confidence bar ───────────────────────────────────────────────
        self.conf_var = tk.DoubleVar(value=0)
        self.conf_bar = ttk.Progressbar(
            result_card, variable=self.conf_var,
            maximum=100, length=660, mode="determinate",
        )
        self.conf_bar.pack(padx=12, pady=(0, 10))

        # ── Symptoms section ─────────────────────────────────────────────
        sym_card = tk.Frame(self.root, bg=BG_CARD)
        sym_card.pack(padx=20, pady=5, fill="x")

        tk.Label(
            sym_card,
            text="Symptom Analyser (optional)",
            font=("Segoe UI", 11, "bold"),
            bg=BG_CARD, fg=FG_MUTED,
        ).pack(anchor="w", padx=12, pady=(8, 2))

        self.sym_entry = tk.Entry(
            sym_card,
            font=("Segoe UI", 12),
            bg="#37474F", fg=FG_LIGHT,
            insertbackground=FG_LIGHT,
            relief="flat", bd=0,
        )
        self.sym_entry.pack(fill="x", padx=12, ipady=7)
        self.sym_entry.insert(0, "e.g. headache, blurred vision, nausea")
        self.sym_entry.bind("<FocusIn>",  self._on_sym_focus_in)
        self.sym_entry.bind("<FocusOut>", self._on_sym_focus_out)

        tk.Button(
            sym_card,
            text="🔍  Analyse Symptoms",
            command=self._on_analyse_symptoms,
            font=("Segoe UI", 11),
            bg=ACCENT2, fg="white",
            activebackground="#388E3C", activeforeground="white",
            padx=12, pady=6, bd=0, cursor="hand2",
        ).pack(anchor="w", padx=12, pady=8)

        self.nlp_label = tk.Label(
            sym_card,
            text="",
            font=("Segoe UI", 12),
            bg=BG_CARD, fg=WARNING,
            wraplength=660, justify="left",
        )
        self.nlp_label.pack(anchor="w", padx=12, pady=(0, 10))

        # ── Status bar ───────────────────────────────────────────────────
        self.status_label = tk.Label(
            self.root,
            text="⏳ Initialising…",
            font=("Segoe UI", 10),
            bg=BG_DARK, fg=FG_MUTED,
        )
        self.status_label.pack(side="bottom", pady=8)

        # ── Disclaimer ───────────────────────────────────────────────────
        tk.Label(
            self.root,
            text="⚠️ For educational and research purposes only — not for clinical use.",
            font=("Segoe UI", 9, "italic"),
            bg=BG_DARK, fg=FG_MUTED,
        ).pack(side="bottom")

    # ── Entry placeholder helpers ────────────────────────────────────────
    _PLACEHOLDER = "e.g. headache, blurred vision, nausea"

    def _on_sym_focus_in(self, _event: tk.Event) -> None:
        if self.sym_entry.get() == self._PLACEHOLDER:
            self.sym_entry.delete(0, tk.END)
            self.sym_entry.config(fg=FG_LIGHT)

    def _on_sym_focus_out(self, _event: tk.Event) -> None:
        if not self.sym_entry.get():
            self.sym_entry.insert(0, self._PLACEHOLDER)
            self.sym_entry.config(fg=FG_MUTED)

    # ── UI helpers ───────────────────────────────────────────────────────
    def _set_status(self, text: str, colour: str = FG_MUTED) -> None:
        self.status_label.config(text=text, fg=colour)

    def _reset_ui(self) -> None:
        self._current_image_path = None
        self.img_label.config(image="")
        self.img_label.image = None                    # type: ignore[attr-defined]
        self.result_label.config(text="—", fg=FG_LIGHT)
        self.conf_var.set(0)
        self.nlp_label.config(text="")
        self._set_status("Ready.", FG_MUTED)

    def _show_progress(self, show: bool) -> None:
        if show:
            self.progress.pack(pady=4)
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    # ── Browse & predict ────────────────────────────────────────────────
    def _on_browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")],
        )
        if not path:
            return

        self._current_image_path = path
        self.result_label.config(text="—", fg=FG_LIGHT)
        self.conf_var.set(0)
        self._set_status("Processing…", FG_MUTED)
        self._show_progress(True)

        # Show thumbnail immediately
        try:
            thumb = Image.open(path).resize((200, 200))
            tk_img = ImageTk.PhotoImage(thumb)
            self.img_label.config(image=tk_img)
            self.img_label.image = tk_img             # type: ignore[attr-defined]
        except Exception:
            pass

        threading.Thread(target=self._predict, args=(path,), daemon=True).start()

    def _predict(self, path: str) -> None:
        """Run CNN inference in a background thread."""
        if self._cnn_model is None:
            self.root.after(0, lambda: messagebox.showerror(
                "Model Error",
                f"CNN model is not loaded.\n\nExpected: {MODEL_PATH}\n\n"
                "Set the CNN_MODEL_PATH environment variable.",
            ))
            self.root.after(0, lambda: self._show_progress(False))
            self.root.after(0, lambda: self._set_status("❌ Model not loaded.", ERROR_COL))
            return

        try:
            arr    = preprocess_image(path)
            preds  = self._cnn_model.predict(arr, verbose=0)[0]
            idx    = int(np.argmax(preds))
            label  = CLASS_LABELS[idx]
            conf   = float(preds[idx]) * 100

            # Display full confidence breakdown
            breakdown = "   ".join(
                f"{CLASS_LABELS[i]}: {preds[i]*100:.1f}%"
                for i in range(len(CLASS_LABELS))
            )
            display = DISPLAY_LABELS.get(label, label.capitalize())
            result_text = (
                f"Prediction: {display}   ({conf:.1f}% confidence)\n"
                f"{breakdown}"
            )

            colour = ACCENT2 if label == "notumor" else WARNING

            def _update() -> None:
                self.result_label.config(text=result_text, fg=colour)
                self.conf_var.set(conf)
                self._show_progress(False)
                self._set_status(f"✅ Prediction complete: {display}", ACCENT2)

            self.root.after(0, _update)

        except Exception as exc:
            logger.exception("Inference failed")
            self.root.after(0, lambda: self.result_label.config(
                text=f"Error: {exc}", fg=ERROR_COL
            ))
            self.root.after(0, lambda: self._show_progress(False))
            self.root.after(0, lambda: self._set_status("❌ Inference error.", ERROR_COL))

    # ── Symptom analysis ────────────────────────────────────────────────
    def _on_analyse_symptoms(self) -> None:
        text = self.sym_entry.get().strip()
        if not text or text == self._PLACEHOLDER:
            self.nlp_label.config(text="⚠️ Please enter some symptoms first.")
            return

        X     = self.tfidf.transform([text.lower()])
        pred  = self.nlp_clf.predict(X)[0]
        proba = float(max(self.nlp_clf.predict_proba(X)[0])) * 100
        display = DISPLAY_LABELS.get(pred, pred.capitalize())
        self.nlp_label.config(
            text=f"💊 Likely tumour type: {display}  ({proba:.1f}% NLP confidence)",
            fg=WARNING,
        )
        self._set_status("NLP analysis complete.", ACCENT2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app  = TumorApp(root)
    root.mainloop()