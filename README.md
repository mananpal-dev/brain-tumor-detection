🧠 Brain Tumor Detection using Deep Learning with Synthetic MRI Generation
📌 Overview

This project presents an end-to-end multimodal deep learning system for brain tumor detection from MRI scans. It integrates multiple AI approaches to improve accuracy, robustness, and interpretability.

The system combines:

🧠 CNN (MobileNetV2) for primary image classification
🔬 Vision Transformer (ViT) for high-accuracy validation
🎨 GAN (Generative Adversarial Network) for synthetic MRI generation
🗣️ NLP-based symptom analysis for auxiliary prediction support

👉 The goal is to build a robust medical AI system using multimodal learning and synthetic data augmentation.

🚀 Key Features
🧠 Brain tumor classification from MRI scans
⚖️ Dual-model prediction (CNN + Vision Transformer)
🎨 Synthetic MRI generation using GAN
🗣️ Symptom-based NLP prediction module
📊 Model comparison with confidence scores
🔄 Data augmentation for better generalization
🌐 Streamlit-based interactive web app
🧾 Explainable AI using Grad-CAM
🧬 Classes Detected

The model classifies MRI scans into:

Glioma Tumor
Meningioma Tumor
Pituitary Tumor
No Tumor
📊 Model Performance
Model	Accuracy
🧠 CNN (MobileNetV2)	83.4%
🔬 Vision Transformer (ViT)	99.8%

👉 The Vision Transformer significantly improves classification reliability by capturing global image relationships better than CNN.

🧠 Deep Learning Models
🔹 1. CNN (MobileNetV2)
Transfer learning from ImageNet
Optimized for medical image classification
Lightweight and efficient

Architecture:

GlobalAveragePooling
Dense layers
Dropout regularization
Softmax output
🔹 2. Vision Transformer (ViT) ⭐ (Major Upgrade)

A transformer-based architecture applied to image patches.

Splits MRI images into patches
Uses self-attention mechanism
Captures long-range dependencies
Provides state-of-the-art accuracy (99.8%)

👉 Benefits:

Better global feature understanding
Higher robustness than CNN
Reduced misclassification rate
🔹 3. GAN (Synthetic MRI Generation)

Used to generate realistic MRI scans for training improvement.

Generator: Deep convolutional network
Discriminator: CNN-based classifier
Latent vector: 128-dimensional noise

👉 Benefits:

Expands dataset size
Reduces overfitting
Improves generalization
🔹 4. NLP Model (Symptom Analysis)
TF-IDF vectorization
Logistic Regression classifier

👉 Used for:

Predicting tumor type from symptoms
Supporting multimodal decision-making
🧪 Dataset
Source: Kaggle Brain MRI Dataset
Classes: Glioma, Meningioma, Pituitary, No Tumor

📌 Dataset link:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

⚠️ Note: Dataset not included due to size limitations.

⚙️ Installation
git clone https://github.com/mananpal-dev/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt
🧪 GAN Training (Synthetic Data Generation)
python mri_gan_generator.py

Generates:

Synthetic MRI images
Trained GAN models
Class-wise augmented dataset

Output:

synthetic_dataset/
🏋️ CNN Training
python brain_tumor_training.py

Includes:

Real + synthetic dataset
Data augmentation
Early stopping
Learning rate scheduling
🌐 Run Web App
streamlit run brain_tumor_streamlit.py
Features:
Upload MRI scan
CNN + ViT prediction comparison
Confidence visualization
Symptom-based prediction
Model agreement analysis
📊 Evaluation Metrics
Accuracy
Precision / Recall / F1-score
Confusion Matrix
CNN vs ViT comparison
Training loss curves
🎨 UI Improvements (Latest Update)
Modern Streamlit interface redesign
Faster prediction pipeline
Improved visualization of results
Clear model agreement/disagreement indicators
Clean and responsive UI layout
🚀 System Architecture Upgrade

This project evolved from a single-model system into a multimodal AI framework:

🧠 CNN → Fast baseline prediction (83.4%)
🔬 ViT → High-accuracy validation (99.8%)
🎨 GAN → Synthetic dataset expansion
🗣️ NLP → Symptom-based inference

👉 This significantly improves:

Accuracy
Robustness
Generalization
Clinical interpretability
👨‍💻 Author

Manan Pal
B.Tech CSE
KIIT University

📜 License

This project is for educational and research purposes only.
