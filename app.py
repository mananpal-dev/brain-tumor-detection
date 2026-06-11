"""
Brain Tumor AI Diagnosis System
=====================================================
Multimodal inference pipeline:
  1. CNN  (MobileNetV2)  — primary classification from MRI image
  2. ViT  (ViT-Base/16)  — independent classification for model comparison
  3. NLP  (TF-IDF + LR)  — supplementary symptom-based prediction
  4. Grad-CAM             — visual explanation of CNN focus regions

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
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
icon = Image.open("assets/favicon.png")
st.set_page_config(
    page_title="Brain Tumor Detection AI · Manan Pal",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — Deep Navy Dark: midnight base + electric blue + warm amber
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Fonts ─────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

    /* ── Root palette ─────────────────────────────────────────────── */
    :root {
        /* Midnight Navy Base */
        --bg-void:       #080d14;
        --bg-base:       #0c1220;
        --bg-panel:      #101827;
        --bg-card:       #141e2e;
        --bg-card-hi:    #192436;

        /* Borders */
        --border-dim:    #1e2d42;
        --border-med:    #253650;
        --border-bright: #2e4468;

        /* Electric Blue — primary accent */
        --blue:          #3b82f6;
        --blue-dim:      #2563eb;
        --blue-bright:   #60a5fa;
        --blue-glow:     rgba(59,130,246,0.15);
        --blue-glow-sm:  rgba(59,130,246,0.07);

        /* Warm Amber — secondary accent */
        --amber:         #f59e0b;
        --amber-dim:     #d97706;
        --amber-bright:  #fbbf24;

        /* Semantic */
        --green:         #10b981;
        --green-dim:     #059669;
        --rose:          #f43f5e;
        --violet:        #8b5cf6;

        /* Typography */
        --txt-bright:    #f0f6ff;
        --txt-mid:       #94b4d4;
        --txt-dim:       #526a84;
        --txt-muted:     #2e4060;

        /* Fonts */
        --font-display:  'Inter', sans-serif;
        --font-body:     'Inter', sans-serif;
        --font-mono:     'JetBrains Mono', monospace;

        /* Radii */
        --radius-sm:     6px;
        --radius-md:     10px;
        --radius-lg:     16px;

        --transition:    0.2s ease;
    }

    /* ── Base reset ────────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-void) !important;
        color: var(--txt-bright) !important;
        font-family: var(--font-body) !important;
    }
    .block-container { padding-top: 1.2rem !important; max-width: 1400px !important; }

    /* ── Sidebar ───────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--bg-base) !important;
        border-right: 1px solid var(--border-dim) !important;
    }

    /* ── Sidebar logo block ────────────────────────────────────────── */
    .sb-logo {
        padding: 1rem 0 1.4rem 0;
        border-bottom: 1px solid var(--border-dim);
        margin-bottom: 1rem;
    }
    .sb-logo-mark {
        font-family: var(--font-display);
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--txt-bright);
        letter-spacing: -0.3px;
        line-height: 1.2;
    }
    .sb-logo-mark span { color: var(--blue-bright); }
    .sb-logo-sub {
        font-family: var(--font-mono);
        font-size: 0.6rem;
        color: var(--txt-dim);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* ── Sidebar headings ──────────────────────────────────────────── */
    .sb-heading {
        font-family: var(--font-mono);
        font-size: 0.58rem;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--txt-dim);
        margin: 1.2rem 0 0.5rem 0;
    }

    /* ── Model status pills ────────────────────────────────────────── */
    .status-pill {
        display: flex;
        align-items: center;
        gap: 9px;
        padding: 7px 10px;
        border-radius: var(--radius-sm);
        background: var(--bg-panel);
        border: 1px solid var(--border-dim);
        margin-bottom: 5px;
        font-family: var(--font-mono);
        font-size: 0.68rem;
        color: var(--txt-mid);
        transition: border-color var(--transition);
    }
    .status-pill:hover { border-color: var(--border-med); }
    .dot-ok {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--green);
        flex-shrink: 0;
        animation: pulse-dot 2.5s ease-in-out infinite;
    }
    .dot-err {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--rose);
        flex-shrink: 0;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.45; }
    }

    /* ── Warning strip ─────────────────────────────────────────────── */
    .warn-strip {
        background: rgba(245,158,11,0.06);
        border: 1px solid rgba(245,158,11,0.2);
        border-left: 3px solid var(--amber-dim);
        border-radius: var(--radius-sm);
        padding: 8px 10px;
        font-size: 0.72rem;
        color: var(--amber-dim);
        line-height: 1.55;
        margin-top: 6px;
    }
    .warn-strip code {
        font-family: var(--font-mono);
        background: rgba(245,158,11,0.1);
        padding: 1px 4px;
        border-radius: 3px;
        font-size: 0.68rem;
    }

    /* ── Hero ──────────────────────────────────────────────────────── */
    .hero {
        position: relative;
        overflow: hidden;
        background: var(--bg-panel);
        border: 1px solid var(--border-med);
        border-top: 2px solid var(--blue);
        border-radius: var(--radius-lg);
        padding: 2rem 2.4rem 1.8rem 2.4rem;
        margin-bottom: 1.6rem;
    }
    .hero::after {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(59,130,246,0.07) 0%, transparent 65%);
        pointer-events: none;
    }

    .hero-eyebrow {
        font-family: var(--font-mono);
        font-size: 0.6rem;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: var(--blue-bright);
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .hero-eyebrow::before {
        content: '';
        display: inline-block;
        width: 18px; height: 1px;
        background: var(--blue-bright);
    }

    .hero-title {
        font-family: var(--font-display);
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.8px;
        color: var(--txt-bright);
        line-height: 1.15;
        margin: 0 0 0.5rem 0;
    }
    .hero-title em {
        font-style: normal;
        color: var(--blue-bright);
    }

    .hero-desc {
        font-size: 0.88rem;
        color: var(--txt-mid);
        max-width: 580px;
        line-height: 1.75;
        margin: 0 0 1.3rem 0;
        font-weight: 400;
    }

    /* ── Pill badges ───────────────────────────────────────────────── */
    .badge-row { display: flex; gap: 6px; flex-wrap: wrap; }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 10px;
        border-radius: 20px;
        font-family: var(--font-mono);
        font-size: 0.63rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }
    .badge::before { content: ''; width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }

    .badge-blue   { background: rgba(59,130,246,0.1);  color: var(--blue-bright);  border: 1px solid rgba(59,130,246,0.25); }
    .badge-blue::before   { background: var(--blue-bright); }
    .badge-rose   { background: rgba(244,63,94,0.1);   color: #fb7185;             border: 1px solid rgba(244,63,94,0.25); }
    .badge-rose::before   { background: #fb7185; }
    .badge-amber  { background: rgba(245,158,11,0.1);  color: var(--amber-bright); border: 1px solid rgba(245,158,11,0.25); }
    .badge-amber::before  { background: var(--amber-bright); }
    .badge-green  { background: rgba(16,185,129,0.1);  color: #34d399;             border: 1px solid rgba(16,185,129,0.25); }
    .badge-green::before  { background: #34d399; }
    .badge-violet { background: rgba(139,92,246,0.1);  color: #a78bfa;             border: 1px solid rgba(139,92,246,0.25); }
    .badge-violet::before { background: #a78bfa; }

    /* ── Section headings ──────────────────────────────────────────── */
    .sec-head {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: var(--font-mono);
        font-size: 0.58rem;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--txt-dim);
        margin: 1.4rem 0 0.7rem 0;
    }
    .sec-head::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border-dim);
    }

    /* ── Prediction cards ──────────────────────────────────────────── */
    .pred-card {
        position: relative;
        background: var(--bg-card);
        border: 1px solid var(--border-med);
        border-radius: var(--radius-md);
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.75rem;
        overflow: hidden;
        transition: border-color var(--transition), background var(--transition);
    }
    .pred-card:hover {
        border-color: var(--border-bright);
        background: var(--bg-card-hi);
    }
    .pred-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        border-radius: 3px 0 0 3px;
    }
    .pred-card-cnn::before  { background: var(--blue); }
    .pred-card-vit::before  { background: var(--rose); }
    .pred-card-nlp::before  { background: var(--amber); }

    .pred-model-label {
        font-family: var(--font-mono);
        font-size: 0.58rem;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .color-cnn  { color: var(--blue-bright); }
    .color-vit  { color: #fb7185; }
    .color-nlp  { color: var(--amber-bright); }

    .icon { width: 18px; height: 18px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 0.72rem; }
    .icon-cnn   { background: rgba(59,130,246,0.12); }
    .icon-vit   { background: rgba(244,63,94,0.12); }
    .icon-nlp   { background: rgba(245,158,11,0.12); }

    .pred-diagnosis {
        font-family: var(--font-display);
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.3px;
        line-height: 1.15;
        color: var(--txt-bright);
    }
    .pred-confidence {
        margin-top: 4px;
        font-family: var(--font-mono);
        font-size: 0.7rem;
        color: var(--txt-dim);
    }
    .pred-confidence .conf-value { color: var(--txt-mid); }

    /* ── Confidence bars ───────────────────────────────────────────── */
    .conf-grid { margin-top: 4px; }
    .conf-row {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 8px;
    }
    .conf-lbl {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        color: var(--txt-dim);
        min-width: 90px;
    }
    .conf-track {
        flex: 1;
        height: 3px;
        background: var(--border-dim);
        border-radius: 2px;
        overflow: hidden;
    }
    .conf-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.6s ease;
    }
    .conf-pct {
        font-family: var(--font-mono);
        font-size: 0.66rem;
        color: var(--txt-dim);
        min-width: 40px;
        text-align: right;
    }

    /* ── Verdict card ──────────────────────────────────────────────── */
    .verdict-card {
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem;
        margin-top: 0.3rem;
    }
    .verdict-agree {
        background: rgba(16,185,129,0.05);
        border: 1px solid rgba(16,185,129,0.18);
        border-left: 3px solid var(--green-dim);
    }
    .verdict-disagree {
        background: rgba(245,158,11,0.05);
        border: 1px solid rgba(245,158,11,0.18);
        border-left: 3px solid var(--amber-dim);
    }
    .verdict-single {
        background: var(--bg-card-hi);
        border: 1px solid var(--border-med);
    }
    .verdict-title {
        font-family: var(--font-mono);
        font-size: 0.58rem;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .verdict-agree .verdict-title   { color: var(--green); }
    .verdict-disagree .verdict-title { color: var(--amber-bright); }
    .verdict-single .verdict-title   { color: var(--txt-dim); }
    .verdict-body {
        font-size: 0.86rem;
        line-height: 1.65;
        color: var(--txt-mid);
    }
    .verdict-agree .verdict-body    { color: #6ee7b7; }
    .verdict-disagree .verdict-body { color: #fcd34d; }
    .verdict-body strong { color: var(--txt-bright); }

    /* ── Info card ─────────────────────────────────────────────────── */
    .info-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-dim);
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem;
        font-size: 0.82rem;
        color: var(--txt-mid);
        line-height: 1.65;
        margin-top: 0.8rem;
    }
    .info-card-blue   { border-left: 3px solid var(--blue); }
    .info-card-amber  { border-left: 3px solid var(--amber); }
    .info-card strong { color: var(--txt-bright); }

    /* ── Stat chips ────────────────────────────────────────────────── */
    .stat-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    .stat-chip {
        font-family: var(--font-mono);
        font-size: 0.63rem;
        padding: 3px 8px;
        border-radius: 4px;
        background: var(--bg-card);
        border: 1px solid var(--border-dim);
        color: var(--txt-dim);
    }
    .stat-chip span { color: var(--txt-mid); }

    /* ── Upload placeholder ────────────────────────────────────────── */
    .upload-placeholder {
        background: var(--bg-panel);
        border: 1px dashed var(--border-med);
        border-radius: var(--radius-lg);
        padding: 3rem 2rem;
        text-align: center;
    }
    .upload-icon {
        font-size: 2.2rem;
        display: block;
        margin: 0 auto 0.8rem auto;
        opacity: 0.35;
        animation: float 3.5s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50%       { transform: translateY(-6px); }
    }
    .upload-title {
        font-family: var(--font-display);
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--txt-mid);
        margin-bottom: 0.4rem;
    }
    .upload-sub {
        font-size: 0.8rem;
        color: var(--txt-dim);
        line-height: 1.75;
        max-width: 360px;
        margin: 0 auto;
    }

    /* ── Architecture grid ─────────────────────────────────────────── */
    .arch-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.6rem;
        margin-top: 0.5rem;
    }
    .arch-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-dim);
        border-radius: var(--radius-md);
        padding: 1rem 1rem 0.9rem 1rem;
        transition: border-color var(--transition);
    }
    .arch-card:hover { border-color: var(--border-bright); }

    .arch-card-no {
        font-family: var(--font-mono);
        font-size: 0.55rem;
        color: var(--txt-muted);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .arch-card-title {
        font-family: var(--font-display);
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .arch-card-cnn .arch-card-title { color: var(--blue-bright); }
    .arch-card-vit .arch-card-title { color: #fb7185; }
    .arch-card-nlp .arch-card-title { color: var(--amber-bright); }
    .arch-card-body {
        font-size: 0.75rem;
        color: var(--txt-dim);
        line-height: 1.55;
    }
    .arch-card-cnn { border-top: 2px solid var(--blue-dim); }
    .arch-card-vit { border-top: 2px solid var(--rose); }
    .arch-card-nlp { border-top: 2px solid var(--amber-dim); }

    /* ── Class info cards ──────────────────────────────────────────── */
    .class-card {
        position: relative;
        background: var(--bg-panel);
        border: 1px solid var(--border-dim);
        border-radius: var(--radius-md);
        padding: 0.85rem 1rem 0.85rem 1.1rem;
        margin-bottom: 0.5rem;
        overflow: hidden;
        transition: border-color var(--transition);
    }
    .class-card:hover { border-color: var(--border-med); }
    .class-card-name {
        font-family: var(--font-display);
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--txt-bright);
        margin-bottom: 4px;
    }
    .class-card-desc {
        font-size: 0.75rem;
        color: var(--txt-dim);
        line-height: 1.55;
    }
    .class-card-tags { display: flex; gap: 5px; margin-top: 7px; flex-wrap: wrap; }
    .class-tag {
        font-family: var(--font-mono);
        font-size: 0.57rem;
        padding: 2px 6px;
        border-radius: 3px;
        background: var(--bg-card);
        color: var(--txt-dim);
        border: 1px solid var(--border-dim);
        letter-spacing: 0.4px;
    }

    /* ── Disclaimer ────────────────────────────────────────────────── */
    .disclaimer-bar {
        background: rgba(245,158,11,0.04);
        border: 1px solid rgba(245,158,11,0.12);
        border-radius: var(--radius-sm);
        padding: 8px 10px;
        font-family: var(--font-mono);
        font-size: 0.63rem;
        color: rgba(245,158,11,0.55);
        line-height: 1.5;
        margin-top: 1rem;
        letter-spacing: 0.2px;
    }

    /* ── Footer ────────────────────────────────────────────────────── */
    .footer-bar {
        text-align: center;
        padding: 1.4rem 0 0.6rem 0;
        margin-top: 2.5rem;
        border-top: 1px solid var(--border-dim);
        font-family: var(--font-mono);
        font-size: 0.66rem;
        color: var(--txt-dim);
        letter-spacing: 0.4px;
    }
    .footer-bar a { color: var(--blue-bright); text-decoration: none; }
    .footer-bar a:hover { color: var(--txt-bright); }
    .footer-name {
        font-family: var(--font-display);
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--txt-mid);
        margin-bottom: 2px;
    }

    /* ── Streamlit overrides ───────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: var(--bg-panel) !important;
        border: 1px dashed var(--border-med) !important;
        border-radius: var(--radius-md) !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--blue-dim) !important;
    }
    textarea, input[type="text"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-med) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--txt-bright) !important;
        font-family: var(--font-body) !important;
        font-size: 0.82rem !important;
    }
    textarea:focus, input[type="text"]:focus {
        border-color: var(--blue-dim) !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.08) !important;
        outline: none !important;
    }
    label { color: var(--txt-dim) !important; font-size: 0.76rem !important; font-family: var(--font-mono) !important; }
    hr { border-color: var(--border-dim) !important; margin: 0.8rem 0 !important; }
    [data-testid="stExpander"] {
        background: var(--bg-panel) !important;
        border: 1px solid var(--border-dim) !important;
        border-radius: var(--radius-sm) !important;
    }
    .stCheckbox > label { color: var(--txt-mid) !important; font-size: 0.8rem !important; }

    /* Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: var(--bg-panel) !important;
        border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
        gap: 0 !important;
        border-bottom: 1px solid var(--border-med) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent !important;
        font-family: var(--font-mono) !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.8px !important;
        color: var(--txt-dim) !important;
        padding: 8px 16px !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--blue-bright) !important;
        border-bottom: 2px solid var(--blue) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-panel"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-med) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
        padding: 0 !important;
    }

    /* Spinner */
    [data-testid="stSpinner"] { color: var(--blue) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE: tuple[int, int] = (224, 224)

CLASS_LABELS: list[str] = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

CLASS_INFO: dict[str, dict] = {
    "Glioma": {
        "summary": "Gliomas originate from glial support cells and account for ~30% of all brain tumours. They range from slow-growing (Grade I) to highly aggressive (Grade IV / Glioblastoma).",
        "prevalence": "~30% of brain tumours",
        "nature": "Malignant (varies by grade)",
        "color": "#3b82f6",
        "accent": "var(--blue)",
    },
    "Meningioma": {
        "summary": "Meningiomas arise from the meninges — the protective layers surrounding the brain. Approximately 85% are benign and among the most common primary brain tumours.",
        "prevalence": "~37% of brain tumours",
        "nature": "Usually benign",
        "color": "#f43f5e",
        "accent": "var(--rose)",
    },
    "No Tumor": {
        "summary": "No tumour-like region was detected in the provided MRI scan. This does not substitute a clinical radiological report.",
        "prevalence": "—",
        "nature": "—",
        "color": "#10b981",
        "accent": "var(--green)",
    },
    "Pituitary": {
        "summary": "Pituitary tumours affect the pituitary gland at the base of the brain. Most are benign adenomas that may cause hormonal imbalances due to their location.",
        "prevalence": "~15% of brain tumours",
        "nature": "Usually benign adenoma",
        "color": "#8b5cf6",
        "accent": "var(--violet)",
    },
}

CNN_MODEL_PATH: str = os.environ.get("CNN_MODEL_PATH", "my_brain_tumor_mobilenetv2.h5")
VIT_MODEL_PATH: str = os.environ.get("VIT_MODEL_PATH", "best_vit_model.pth")

TUMOR_COLORS: dict[str, str] = {
    "Glioma":     "#3b82f6",
    "Meningioma": "#f43f5e",
    "No Tumor":   "#10b981",
    "Pituitary":  "#8b5cf6",
}

device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initialising CNN…")
def load_cnn() -> tf.keras.Model | None:
    path = Path(CNN_MODEL_PATH)
    if not path.is_file():
        return None
    try:
        model = load_model(str(path), compile=False)
        logger.info("CNN model loaded from %s", path)
        return model
    except Exception as exc:
        logger.exception("CNN load error: %s", exc)
        return None


@st.cache_resource(show_spinner="Initialising ViT…")
def load_vit() -> ViTForImageClassification | None:
    path = Path(VIT_MODEL_PATH)
    if not path.is_file():
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
        logger.exception("ViT load error: %s", exc)
        return None


@st.cache_resource(show_spinner="Building NLP classifier…")
def load_nlp() -> tuple[TfidfVectorizer, LogisticRegression]:
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
    X = tfidf.fit_transform(texts)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X, labels)
    return tfidf, clf


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_for_cnn(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_for_vit(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).float()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_cnn(model, arr):
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_LABELS[idx], float(probs[idx]) * 100, probs


def run_vit(model, tensor):
    with torch.no_grad():
        logits = model(tensor).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return CLASS_LABELS[idx], float(probs[idx]) * 100, probs


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def compute_gradcam(model, arr, layer_name="Conv_1"):
    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(arr, training=False)
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(conv_out[0] * pooled, axis=-1)
        heatmap = np.maximum(heatmap.numpy(), 0)
        max_val = heatmap.max()
        if max_val > 0:
            heatmap /= max_val
        return heatmap
    except Exception as exc:
        logger.warning("Grad-CAM failed: %s", exc)
        return None


def overlay_gradcam(heatmap, pil_img):
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet_rgb = (plt.cm.jet(heatmap_uint8)[:, :, :3] * 255).astype(np.uint8)
    original = np.array(pil_img.resize(heatmap_uint8.shape[::-1]).convert("RGB"))
    blended = (jet_rgb * 0.4 + original * 0.6).astype(np.uint8)
    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def render_conf_bars(probs: np.ndarray, accent: str) -> None:
    rows_html = '<div class="conf-grid">'
    for lbl, p in zip(CLASS_LABELS, probs):
        pct = p * 100
        color = TUMOR_COLORS.get(lbl, accent)
        rows_html += f"""
        <div class="conf-row">
            <span class="conf-lbl">{lbl}</span>
            <div class="conf-track">
                <div class="conf-fill" style="width:{pct:.1f}%;background:{color};"></div>
            </div>
            <span class="conf-pct">{pct:.1f}%</span>
        </div>"""
    rows_html += '</div>'
    st.markdown(rows_html, unsafe_allow_html=True)


def build_bar_chart(cnn_probs: np.ndarray, vit_probs: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    fig.patch.set_facecolor("#101827")
    ax.set_facecolor("#141e2e")

    y = np.arange(len(CLASS_LABELS))
    h = 0.3

    ax.barh(y + h / 2, cnn_probs * 100, h,
            label="CNN · MobileNetV2", color="#3b82f6", alpha=0.9)
    ax.barh(y - h / 2, vit_probs * 100, h,
            label="ViT · Base/16",     color="#f43f5e", alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(CLASS_LABELS, color="#94b4d4",
                       fontsize=8.5, fontfamily="monospace")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", color="#526a84", fontsize=7.5)
    ax.tick_params(colors="#526a84", labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color("#1e2d42")
    ax.grid(axis="x", color="#1e2d42", linewidth=0.6, linestyle="--", alpha=0.8)
    ax.legend(
        loc="lower right", fontsize=7.5,
        labelcolor="#94b4d4",
        facecolor="#101827",
        edgecolor="#1e2d42",
    )
    plt.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
cnn_model = load_cnn()
vit_model = load_vit()
tfidf, symptom_clf = load_nlp()


# ===========================================================================
# ─── SIDEBAR ────────────────────────────────────────────────────────────────
# ===========================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sb-logo">
            <div class="sb-logo-mark">Brain Tumor <span>Detection</span></div>
            <div class="sb-logo-sub">AI · Multimodal MRI System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sb-heading'>MRI Input</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload MRI scan",
        type=["jpg", "jpeg", "png"],
        help="T1 or T2 weighted brain MRI in JPG / PNG.",
        label_visibility="collapsed",
    )
    st.caption("T1 / T2-weighted · JPG or PNG · Axial / Coronal / Sagittal")

    st.markdown("<div class='sb-heading'>Symptoms</div>", unsafe_allow_html=True)
    symptoms_input = st.text_area(
        "Symptoms",
        placeholder="e.g. headache, blurred vision, seizures…",
        height=72,
        label_visibility="collapsed",
    )

    st.markdown("<div class='sb-heading'>Options</div>", unsafe_allow_html=True)
    show_gradcam = st.checkbox("Show Grad-CAM attention", value=True)

    st.markdown("<div class='sb-heading'>Model Status</div>", unsafe_allow_html=True)
    for label, ok in [
        ("CNN · MobileNetV2", cnn_model is not None),
        ("ViT · Patch16-224", vit_model is not None),
        ("NLP · TF-IDF + LR", True),
    ]:
        dot = "dot-ok" if ok else "dot-err"
        st.markdown(
            f"<div class='status-pill'><span class='{dot}'></span>{label}</div>",
            unsafe_allow_html=True,
        )

    if not cnn_model:
        st.markdown(
            "<div class='warn-strip'>CNN weights not found.<br>"
            "Place <code>my_brain_tumor_mobilenetv2.h5</code> in the project root "
            "or set <code>CNN_MODEL_PATH</code>.</div>",
            unsafe_allow_html=True,
        )
    if not vit_model:
        st.markdown(
            "<div class='warn-strip'>ViT weights not found.<br>"
            "Place <code>best_vit_model.pth</code> in the root "
            "or set <code>VIT_MODEL_PATH</code>.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='disclaimer-bar'>⚠ RESEARCH PROTOTYPE · NOT A MEDICAL DEVICE<br>"
        "Not a substitute for clinical radiological diagnosis.</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# ─── HERO ───────────────────────────────────────────────────────────────────
# ===========================================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">Multimodal AI · v2.0 · Deep Learning Research</div>
        <div class="hero-title">Brain Tumor <em>Detection</em><br>via MRI Analysis</div>
        <div class="hero-desc">
            A dual-model pipeline combining a fine-tuned CNN and Vision Transformer
            for 4-class brain MRI classification — augmented with Grad-CAM
            visual explainability and NLP-based symptom correlation.
        </div>
        <div class="badge-row">
            <span class="badge badge-blue">MobileNetV2 CNN</span>
            <span class="badge badge-rose">ViT-Base/16</span>
            <span class="badge badge-amber">TF-IDF + LR</span>
            <span class="badge badge-green">Grad-CAM XAI</span>
            <span class="badge badge-violet">4-Class Softmax</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# ─── MAIN ───────────────────────────────────────────────────────────────────
# ===========================================================================
col_left, col_right = st.columns([5, 4], gap="large")

if uploaded_file is not None:
    try:
        pil_img = Image.open(uploaded_file)

        with st.spinner("Running inference pipeline…"):
            cnn_label = cnn_conf = cnn_probs = None
            if cnn_model is not None:
                cnn_arr = preprocess_for_cnn(pil_img)
                cnn_label, cnn_conf, cnn_probs = run_cnn(cnn_model, cnn_arr)
            else:
                cnn_arr = preprocess_for_cnn(pil_img)

            vit_label = vit_conf = vit_probs = None
            if vit_model is not None:
                vit_tensor = preprocess_for_vit(pil_img)
                vit_label, vit_conf, vit_probs = run_vit(vit_model, vit_tensor)

            sym_pred = sym_conf = None
            if symptoms_input.strip():
                X_sym = tfidf.transform([symptoms_input.lower()])
                sym_pred = symptom_clf.predict(X_sym)[0]
                sym_conf = float(max(symptom_clf.predict_proba(X_sym)[0])) * 100

        # ── LEFT ─────────────────────────────────────────────────────
        with col_left:
            display_img = pil_img.resize(IMG_SIZE).convert("RGB")

            if show_gradcam and cnn_model is not None:
                heatmap = compute_gradcam(cnn_model, cnn_arr)
                if heatmap is not None:
                    tab_orig, tab_cam = st.tabs(["ORIGINAL MRI", "GRAD-CAM ATTENTION"])
                    with tab_orig:
                        st.image(display_img, use_container_width=True)
                    with tab_cam:
                        overlay = overlay_gradcam(heatmap, pil_img)
                        st.image(overlay, use_container_width=True)
                        st.markdown(
                            "<div class='info-card info-card-blue'>"
                            "<strong>Grad-CAM</strong> (Gradient-weighted Class Activation Mapping) "
                            "highlights the pixel regions most influential on the CNN's prediction. "
                            "<em>Warmer colours</em> (red/yellow) indicate higher model attention.</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.image(display_img, use_container_width=True)
            else:
                st.image(display_img, use_container_width=True)

            # CNN card
            if cnn_label is not None:
                st.markdown(
                    f"""
                    <div class="pred-card pred-card-cnn">
                        <div class="pred-model-label color-cnn">
                            <div class="icon icon-cnn">▣</div>
                            CNN &nbsp;·&nbsp; MobileNetV2
                        </div>
                        <div class="pred-diagnosis">{cnn_label}</div>
                        <div class="pred-confidence">
                            Confidence: <span class="conf-value">{cnn_conf:.1f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("Per-class probabilities · CNN"):
                    render_conf_bars(cnn_probs, "#3b82f6")

            # ViT card
            if vit_label is not None:
                st.markdown(
                    f"""
                    <div class="pred-card pred-card-vit">
                        <div class="pred-model-label color-vit">
                            <div class="icon icon-vit">◈</div>
                            ViT &nbsp;·&nbsp; Base / Patch16-224
                        </div>
                        <div class="pred-diagnosis">{vit_label}</div>
                        <div class="pred-confidence">
                            Confidence: <span class="conf-value">{vit_conf:.1f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("Per-class probabilities · ViT"):
                    render_conf_bars(vit_probs, "#f43f5e")

            # NLP card
            if sym_pred is not None:
                st.markdown(
                    f"""
                    <div class="pred-card pred-card-nlp">
                        <div class="pred-model-label color-nlp">
                            <div class="icon icon-nlp">◉</div>
                            NLP &nbsp;·&nbsp; Symptom Classifier
                        </div>
                        <div class="pred-diagnosis">{sym_pred}</div>
                        <div class="pred-confidence">
                            Confidence: <span class="conf-value">{sym_conf:.1f}%</span>
                            &nbsp;·&nbsp; symptom-based
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── RIGHT ────────────────────────────────────────────────────
        with col_right:
            # Comparison chart
            if cnn_probs is not None and vit_probs is not None:
                st.markdown("<div class='sec-head'>Model Comparison</div>", unsafe_allow_html=True)
                fig = build_bar_chart(cnn_probs, vit_probs)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            # Verdict
            st.markdown("<div class='sec-head'>Final Verdict</div>", unsafe_allow_html=True)
            if cnn_label is not None and vit_label is not None:
                if cnn_label == vit_label:
                    st.markdown(
                        f"""<div class="verdict-card verdict-agree">
                            <div class="verdict-title">✓ Consensus Prediction</div>
                            <div class="verdict-body">
                                Both models independently classify this scan as
                                <strong>{cnn_label}</strong>.<br>
                                <span style="font-size:0.76rem;opacity:0.7;">
                                CNN {cnn_conf:.1f}% &nbsp;·&nbsp; ViT {vit_conf:.1f}%
                                </span>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""<div class="verdict-card verdict-disagree">
                            <div class="verdict-title">⚡ Model Disagreement</div>
                            <div class="verdict-body">
                                Further clinical review recommended.<br>
                                CNN → <strong>{cnn_label}</strong> ({cnn_conf:.1f}%)<br>
                                ViT &nbsp;→ <strong>{vit_label}</strong> ({vit_conf:.1f}%)
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            elif cnn_label is not None:
                st.markdown(
                    f"""<div class="verdict-card verdict-single">
                        <div class="verdict-title">CNN Result</div>
                        <div class="verdict-body">CNN → <strong>{cnn_label}</strong> ({cnn_conf:.1f}%)</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            elif vit_label is not None:
                st.markdown(
                    f"""<div class="verdict-card verdict-single">
                        <div class="verdict-title">ViT Result</div>
                        <div class="verdict-body">ViT → <strong>{vit_label}</strong> ({vit_conf:.1f}%)</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Class info
            primary_label = cnn_label or vit_label
            if primary_label and primary_label in CLASS_INFO:
                info = CLASS_INFO[primary_label]
                st.markdown(
                    f"""<div class="info-card info-card-blue" style="margin-top:1rem;">
                        <strong>About · {primary_label}</strong><br><br>
                        {info['summary']}
                        <div class="stat-row">
                            <div class="stat-chip">Prevalence &nbsp;<span>{info['prevalence']}</span></div>
                            <div class="stat-chip">Nature &nbsp;<span>{info['nature']}</span></div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Symptom correlation
            if sym_pred is not None and (cnn_label is not None or vit_label is not None):
                img_pred = cnn_label or vit_label
                if sym_pred == img_pred:
                    note_color = "info-card-blue"
                    icon = "✓"
                    note = f"NLP symptom analysis corroborates the imaging prediction: <strong>{sym_pred}</strong>."
                else:
                    note_color = "info-card-amber"
                    icon = "⚡"
                    note = (f"NLP suggests <strong>{sym_pred}</strong>, differing from imaging. "
                            "Clinical correlation is advised.")
                st.markdown(
                    f"""<div class="info-card {note_color}" style="margin-top:0.6rem;">
                        {icon} &nbsp;{note}
                    </div>""",
                    unsafe_allow_html=True,
                )

    except Exception as exc:
        st.error(f"An error occurred while processing the image: {exc}")
        logger.exception("Inference error")

# ===========================================================================
# ─── LANDING STATE ──────────────────────────────────────────────────────────
# ===========================================================================
else:
    with col_left:
        st.markdown(
            """
            <div class="upload-placeholder">
                <span class="upload-icon">🧠</span>
                <div class="upload-title">Upload a brain MRI to begin</div>
                <div class="upload-sub">
                    Use the sidebar uploader to load a T1 or T2-weighted brain MRI
                    in JPG or PNG. The system runs the CNN and ViT models in parallel,
                    returns per-class probabilities, a Grad-CAM attention overlay,
                    and an ensemble verdict.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sec-head' style='margin-top:1.6rem;'>Pipeline Architecture</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="arch-grid">
                <div class="arch-card arch-card-cnn">
                    <div class="arch-card-no">Model 01</div>
                    <div class="arch-card-title">MobileNetV2</div>
                    <div class="arch-card-body">
                        Fine-tuned CNN on Brain Tumor MRI Dataset.
                        Transfer learning from ImageNet weights;
                        4-class softmax output.
                    </div>
                </div>
                <div class="arch-card arch-card-vit">
                    <div class="arch-card-no">Model 02</div>
                    <div class="arch-card-title">ViT-Base/16</div>
                    <div class="arch-card-body">
                        Vision Transformer with custom head.
                        Global attention over 196 patches —
                        captures context beyond local receptive fields.
                    </div>
                </div>
                <div class="arch-card arch-card-nlp">
                    <div class="arch-card-no">Model 03</div>
                    <div class="arch-card-title">TF-IDF + LR</div>
                    <div class="arch-card-body">
                        Bigram TF-IDF vectorizer with Logistic
                        Regression. Supplementary class prediction
                        from free-text symptom descriptions.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("<div class='sec-head'>Classification Targets</div>", unsafe_allow_html=True)
        for label, info in CLASS_INFO.items():
            color = info["color"]
            nature_tag = info["nature"] if info["nature"] != "—" else None
            prev_tag   = info["prevalence"] if info["prevalence"] != "—" else None
            tags_html = ""
            if nature_tag:
                tags_html += f"<span class='class-tag'>{nature_tag}</span>"
            if prev_tag:
                tags_html += f"<span class='class-tag'>{prev_tag}</span>"
            st.markdown(
                f"""<div class="class-card" style="border-left: 3px solid {color};">
                    <div class="class-card-name">{label}</div>
                    <div class="class-card-desc">{info['summary']}</div>
                    <div class="class-card-tags">{tags_html}</div>
                </div>""",
                unsafe_allow_html=True,
            )

# ===========================================================================
# ─── FOOTER ─────────────────────────────────────────────────────────────────
# ===========================================================================
st.markdown(
    """
    <div class="footer-bar">
        <div class="footer-name">Manan Pal</div>
        B.Tech Computer Science &nbsp;·&nbsp; KIIT University
        &nbsp;&nbsp;·&nbsp;&nbsp;
        <a href="https://github.com/mananpal-dev/brain-tumor-detection" target="_blank">GitHub →</a>
        &nbsp;&nbsp;·&nbsp;&nbsp;
        Educational &amp; Research Use Only &nbsp;·&nbsp; Not a Clinical Tool
    </div>
    """,
    unsafe_allow_html=True,
)