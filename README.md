🧠 Brain Tumor Detection using Deep Learning with Synthetic MRI Generation
📌 Overview

This project presents an end-to-end deep learning pipeline for brain tumor detection from MRI scans using a multimodal AI approach.

The system combines:

Convolutional Neural Network (CNN – MobileNetV2) for image classification
Vision Transformer (ViT) for model comparison and improved reliability
Generative Adversarial Network (GAN) for synthetic MRI image generation
NLP-based symptom analysis for additional context

The goal is to improve prediction accuracy and robustness by integrating multiple models and synthetic data augmentation.

🚀 Key Features
Brain tumor classification from MRI images
Dual-model prediction (CNN + Vision Transformer)
Synthetic MRI dataset generation using GAN
Multimodal system (Image + Symptoms)
Data augmentation and improved generalization
Streamlit-based interactive web interface
Model comparison and confidence visualization
Explainable AI using Grad-CAM
🧬 Classes Detected

The model classifies MRI scans into:

Glioma Tumor
Meningioma Tumor
Pituitary Tumor
No Tumor
📁 Project Structure
brain-tumor-detection/
│
├── Training/                     # Real MRI training dataset
├── Testing/                      # Real MRI testing dataset
├── synthetic_dataset/            # GAN-generated images
│
├── gan_models/                   # Saved GAN generator models
├── gan_preview/                  # GAN training preview images
├── gan_checkpoints/              # GAN training checkpoints
│
├── mri_gan_generator.py          # GAN training & synthetic image generation
├── brain_tumor_training.py       # CNN training (real + synthetic data)
├── brain_tumor_streamlit.py      # Streamlit web application
├── metrics.py                    # Model evaluation (CNN / ViT comparison)
│
├── my_brain_tumor_mobilenetv2.h5 # Trained CNN model
├── best_vit_model.pth            # Trained Vision Transformer model
│
├── requirements.txt
├── .gitignore
└── README.md


🧠 Deep Learning Models


🔹 1. CNN (MobileNetV2)
Transfer learning from ImageNet
Used for primary tumor classification
Lightweight and efficient for medical images

Architecture additions:

GlobalAveragePooling
Dense layer
Dropout
Softmax output


🔹 2. Vision Transformer (ViT)
Transformer-based image model
Captures global image relationships
Used for model comparison and validation

👉 Helps improve prediction reliability by comparing outputs with CNN

🔹 3. GAN (Synthetic Data Generation)

Used to generate realistic MRI images for each class.

Generator: Deep convolutional network
Discriminator: CNN-based classifier
Latent space: 128-dimensional noise vector

👉 Purpose:

Increase dataset size
Improve model generalization
Reduce overfitting
🔹 4. NLP Model (Symptoms Analysis)
TF-IDF vectorization
Logistic Regression classifier

👉 Used to:

Predict tumor type based on symptoms
Support multimodal decision-making
🧪 Dataset
Source: Kaggle Brain MRI Dataset
Classes: Glioma, Meningioma, Pituitary, No Tumor

To enhance training:

Real MRI images are combined with GAN-generated synthetic images

⚠️ Note: Dataset is not included due to size limitations.
Download from:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

⚙️ Installation
git clone https://github.com/mananpal-dev/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt
🧪 GAN Training (Synthetic Data Generation)
python mri_gan_generator.py

This will:

Train GAN models for each class
Save generator models
Generate synthetic MRI images

Output:

synthetic_dataset/
🏋️ CNN Training
python brain_tumor_training.py


Training includes:

Real dataset
GAN-generated synthetic dataset
Data augmentation
Early stopping and learning rate scheduling
🌐 Running the Application
streamlit run brain_tumor_streamlit.py


Features:

Upload MRI image
Get predictions from CNN and ViT
View confidence scores
Compare model outputs
Optional symptom-based prediction


📊 Model Evaluation

Evaluation includes:

Accuracy
Loss curves
Classification report (Precision, Recall, F1-score)
CNN vs ViT comparison



🎨 UI Improvements (Latest Updates)

Updated Streamlit UI with modern layout
Removed deprecated parameters (use_column_width)
Added loading spinner for better UX
Improved prediction display using metrics
Enhanced visualization and model comparison
Clear agreement/disagreement feedback


🚀 System Upgrade

The system has been upgraded from a single CNN model to a multimodal AI system:

CNN → Primary prediction
ViT → Validation & comparison
GAN → Data augmentation
NLP → Symptom analysis

👉 This improves robustness, interpretability, and performance.


👨‍💻 Author

Manan Pal
B.Tech CSE
KIIT University

📜 License

This project is for educational and research purposes only.
