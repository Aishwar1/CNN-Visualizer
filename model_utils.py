import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import onnxruntime as ort

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

# -------------------------------------------------
# Metrics
# -------------------------------------------------
def compute_classification_metrics(y_true, y_pred, output_dir="outputs"):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted")),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs(output_dir, exist_ok=True)
    cm_path = f"{output_dir}/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    metrics["confusion_matrix"] = "confusion_matrix.png"
    return metrics

# -------------------------------------------------
# Load ONNX model
# -------------------------------------------------
def load_model(model_name):
    session = ort.InferenceSession("model.onnx")
    input_name = session.get_inputs()[0].name

    if model_name == "resnet50":
        preprocess_fn = resnet_preprocess
    elif model_name == "vgg16":
        preprocess_fn = vgg_preprocess
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return (session, input_name), preprocess_fn

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
# Inference using ONNX
# -------------------------------------------------
def run_inference(model_data, preprocess_fn, image_path):
    session, input_name = model_data

    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_fn(x)

    x = x.astype(np.float32)

    preds = session.run(None, {input_name: x})[0]

    y_pred = np.argmax(preds, axis=1)
    y_true = y_pred.copy()

    return y_true, y_pred

# -------------------------------------------------
# Metrics (ONNX version)
# -------------------------------------------------
def compute_metrics(model_data, img_array, y_true):
    session, input_name = model_data

    img_array = img_array.astype(np.float32)
    preds = session.run(None, {input_name: img_array})[0]

    y_pred = np.argmax(preds, axis=1)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro")),
        "recall": float(recall_score(y_true, y_pred, average="macro")),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion": confusion_matrix(y_true, y_pred).tolist()
    }