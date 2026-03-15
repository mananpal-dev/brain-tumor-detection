from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = load_model(r"C:\AD Project\brain_tumor_detection\brain_tumor_detection\my_brain_tumor_mobilenetv2.h5", compile=False)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    r"C:\AD Project\brain_tumor_detection\Testing",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc*100:.2f}%")