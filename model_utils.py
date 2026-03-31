# model_utils.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # IMPORTANT: non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
def compute_classification_metrics(y_true, y_pred, output_dir="outputs"):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted")),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = f"{output_dir}/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    metrics["confusion_matrix"] = "confusion_matrix.png"
    return metrics

# -------------------------------------------------
# Model registry
# -------------------------------------------------
MODEL_REGISTRY = {
    "resnet50": (ResNet50, resnet_preprocess),
    "vgg16": (VGG16, vgg_preprocess),
}

# -------------------------------------------------
# Load model + preprocess function
# -------------------------------------------------
def load_model_by_name(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not supported")

    model_cls, preprocess_fn = MODEL_REGISTRY[model_name]

    model = model_cls(
        weights="imagenet",
        include_top=False
    )
    model.trainable = False

    return model, preprocess_fn

# model architecture
def get_model_architecture(model):
    arch = []
    for layer in model.layers:
        arch.append({
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": layer.output_shape
        })
    return arch

# -------------------------------------------------
# Image preprocessing
# -------------------------------------------------
def load_and_preprocess_image(img_path, preprocess_fn):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_fn(img_array)
    return img_array

# -------------------------------------------------
# Extract feature maps
# -------------------------------------------------
def extract_feature_maps(model, img_array, layer_name):
    layer = model.get_layer(layer_name)

    if len(layer.output_shape) != 4:
        raise ValueError(f"Layer '{layer_name}' is not a Conv layer")

    intermediate_model = Model(
        inputs=model.input,
        outputs=layer.output
    )

    feature_maps = intermediate_model.predict(img_array)
    return feature_maps  # (1, H, W, C)

# -------------------------------------------------
# Save feature maps
# -------------------------------------------------
def save_feature_maps(feature_maps, output_dir, max_features):
    os.makedirs(output_dir, exist_ok=True)

    fmap = feature_maps[0]  # (H, W, C)
    total_channels = fmap.shape[-1]
    max_features = min(max_features, total_channels)

    saved_files = []

    for i in range(max_features):
        plt.figure(figsize=(2, 2))
        plt.imshow(fmap[:, :, i], cmap="viridis")
        plt.axis("off")

        filename = f"channel_{i}.png"
        plt.savefig(os.path.join(output_dir, filename),
                    bbox_inches="tight", pad_inches=0)
        plt.close()

        saved_files.append(filename)

    return saved_files, total_channels
# -------------------------------------------------

def load_model(model_name):
    if model_name == "resnet50":
        model = ResNet50(weights="imagenet")
        preprocess_fn = resnet_preprocess
    elif model_name == "vgg16":
        model = VGG16(weights="imagenet")
        preprocess_fn = vgg_preprocess
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, preprocess_fn
# -------------------------------------------------

def run_inference(model, preprocess_fn, image_path):

    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_fn(x)

    preds = model.predict(x)
    y_pred = np.argmax(preds, axis=1)

    # ⚠️ TEMPORARY PLACEHOLDER
    # Replace when you have real labels
    y_true = y_pred.copy()

    return y_true, y_pred

# -------------------------------------------------

def compute_metrics(model, img_array, y_true):
    preds = model.predict(img_array)
    y_pred = np.argmax(preds, axis=1) # caouse we doint have labelled data here!!

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro")),
        "recall": float(recall_score(y_true, y_pred, average="macro")),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion": confusion_matrix(y_true, y_pred).tolist()
    }
# -------------------------------------------------