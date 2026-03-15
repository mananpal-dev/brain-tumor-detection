# Brain Tumor Detection using Deep Learning with Synthetic MRI Generation

## Overview

This project implements an end-to-end deep learning pipeline for **brain tumor detection from MRI scans** using **MobileNetV2 transfer learning**. To improve dataset size and diversity, a **Generative Adversarial Network (GAN)** is used to generate synthetic MRI images for four tumor classes.

The goal is to improve classification performance by combining **real medical images with GAN-generated synthetic data**.

---

## Project Features

- Brain tumor classification from MRI scans
- Synthetic MRI dataset generation using GAN
- Transfer learning using MobileNetV2
- Data augmentation for improved generalization
- Visualization of training metrics
- Modular pipeline for training and inference
- Streamlit-based interface for predictions

---

## Classes Detected

The model detects the following MRI categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

---

## Project Structure


brain-tumor-detection/
│
├── Training/ # Real MRI training dataset
├── Testing/ # Real MRI testing dataset
├── synthetic_dataset/ # GAN generated images
│
├── gan_models/ # Saved GAN models
├── gan_preview/ # GAN training preview images
│
├── mri_gan_generator.py # GAN training script
├── brain_tumor_training.py # CNN training with real + synthetic data
├── brain_tumor_streamlit.py # Web interface for prediction
│
├── requirements.txt
├── .gitignore
└── README.md


---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- PIL
- Streamlit

---

## Deep Learning Models

### GAN Model
Used to generate synthetic MRI images for each class.

- Generator: Deep convolutional generator
- Discriminator: CNN based discriminator
- Latent space: 128 dimensional noise vector

### Classification Model

Transfer learning using:

MobileNetV2

with additional layers:

- GlobalAveragePooling
- Dense Layer
- Dropout
- Softmax Output

---

## Dataset

The project uses a brain MRI dataset containing four classes.

Real MRI images are augmented with synthetic images generated using GAN to increase training diversity.

---

## Installation

Clone the repository


git clone https://github.com/YOUR\_USERNAME/brain-tumor-detection.git

cd brain-tumor-detection


Install dependencies


pip install -r requirements.txt


---

## Training GAN (Synthetic Dataset Generation)

Run:


python mri_gan_generator.py


This will:

- Train GAN for each class
- Save generator models
- Generate synthetic MRI images

Output:


synthetic_dataset/


---

## Training the Classification Model

Run:


python brain_tumor_training.py


The model will train using:

- Real MRI dataset
- GAN generated synthetic dataset

---

## Running the Streamlit App


streamlit run brain_tumor_streamlit.py


Upload an MRI scan to predict tumor type.

---

## Model Performance

Training includes:

- Early stopping
- Learning rate scheduling
- Data augmentation

Evaluation metrics include:

- Accuracy
- Loss curves
- Classification report

---

## Future Improvements

- Use StyleGAN for higher quality synthetic MRI images
- Deploy model as a web application
- Add segmentation for tumor localization
- Improve dataset size

---

## Author

Manan Pal  
B.Tech CSE  
KIIT University

---

## License

This project is for educational and research purposes.

## Dataset Source

The MRI brain tumor dataset used in this project was obtained from Kaggle.

Dataset Link:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset contains MRI images categorized into four classes:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The dataset is used for academic and research purposes only. 

To improve model generalization and increase training data size, additional synthetic MRI images were generated using a Generative Adversarial Network (GAN).

## Dataset

The project uses a publicly available Brain MRI dataset sourced from Kaggle. 
The dataset consists of MRI images categorized into four classes.

To improve model robustness, the training dataset is expanded using GAN-generated synthetic images.

Note: Due to size limitations, the dataset is not included in this repository. 
Please download it from the Kaggle link provided above.
