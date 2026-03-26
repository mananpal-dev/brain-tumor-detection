import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torch
from transformers import ViTForImageClassification

# ======================
# PATHS
# ======================
CNN_MODEL_PATH = r"C:\AD Project\brain_tumor_detection\brain_tumor_detection\my_brain_tumor_mobilenetv2.h5"
VIT_MODEL_PATH = r"C:\AD Project\brain_tumor_detection\brain_tumor_detection\best_vit_model.pth"
TEST_DIR = r"C:\AD Project\brain_tumor_detection\Testing"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ======================
# LOAD TEST DATA
# ======================
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

y_true = test_data.classes

# ======================
# CNN MODEL
# ======================
cnn_model = load_model(CNN_MODEL_PATH, compile=False)
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn_loss, cnn_acc = cnn_model.evaluate(test_data)

print("\n==============================")
print("📌 CNN MODEL")
print("==============================")
print(f"Accuracy: {cnn_acc*100:.2f}%")

# ======================
# ViT MODEL
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=4,
    ignore_mismatched_sizes=True
)

vit_model.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=device))
vit_model.to(device)
vit_model.eval()

vit_preds = []

for i in range(len(test_data)):
    images, _ = test_data[i]

    images = (images - 0.5) / 0.5
    images = torch.tensor(images).permute(0,3,1,2).float().to(device)

    with torch.no_grad():
        outputs = vit_model(images).logits
        preds = torch.argmax(outputs, dim=1)

    vit_preds.extend(preds.cpu().numpy())

vit_preds = np.array(vit_preds)

vit_acc = np.mean(vit_preds == y_true)

print("\n==============================")
print("📌 ViT MODEL")
print("==============================")
print(f"Accuracy: {vit_acc*100:.2f}%")

# ======================
# COMPARISON
# ======================
print("\n==============================")
print("🔥 FINAL COMPARISON")
print("==============================")
print(f"CNN Accuracy : {cnn_acc*100:.2f}%")
print(f"ViT Accuracy : {vit_acc*100:.2f}%")

# ======================
# GRAPH
# ======================
models = ['CNN', 'ViT']
accuracies = [cnn_acc*100, vit_acc*100]

plt.figure()
plt.bar(models, accuracies)
plt.title("CNN vs ViT Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0,100)
plt.show()