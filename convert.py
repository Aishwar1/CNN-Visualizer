import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Load pretrained model (same as your app)
model = ResNet50(weights="imagenet")

# Save it as SavedModel
model.save("saved_model")

print("SavedModel created ✅")