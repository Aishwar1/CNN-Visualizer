from flask import Flask, jsonify, request, send_from_directory, render_template
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os

from model_utils import (
    load_model_by_name,
    load_and_preprocess_image,
    extract_feature_maps,
    save_feature_maps,
)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# ---------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    model_name = data["model"]
    layer_name = data["layer_name"]
    num_features = int(data["num_features"])
    img_path = data["image_path"]

    model, preprocess_fn = load_model_by_name(model_name)
    img_array = load_and_preprocess_image(img_path, preprocess_fn)

    feature_maps = extract_feature_maps(model, img_array, layer_name)
    run_id = f"{model_name}_{layer_name}"
    output_dir = os.path.join("outputs", run_id)

    os.makedirs(output_dir, exist_ok=True)

    images, _ = save_feature_maps(feature_maps, output_dir, num_features)

    # SAVE METADATA for download
    with open(os.path.join(output_dir, "meta.txt"), "w") as f:
        f.write(f"{model_name}\n{layer_name}\n{num_features}")

    return jsonify({
        "images": [f"{run_id}/{img}" for img in images],
        "run_id": run_id
    })

# ---------------------------------------
@app.route("/layers")
def layers():
    model_name = request.args.get("model", "resnet50")
    model, _ = load_model_by_name(model_name)

    return jsonify([
        l.name for l in model.layers if len(l.output_shape) == 4
    ])

# ---------------------------------------
@app.route("/architecture")
def architecture():
    model_name = request.args.get("model", "resnet50")
    model, _ = load_model_by_name(model_name)

    lines = []
    for layer in model.layers:
        filters = layer.filters if hasattr(layer, "filters") else "-"
        lines.append(
            f"{layer.name:30s} | {layer.__class__.__name__:18s} | filters: {filters}"
        )

    return jsonify({"architecture": "\n".join(lines)})

# ---------------------------------------
@app.route("/metrics")
def metrics():
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])

    cm = confusion_matrix(y_true, y_pred)

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(3, 3))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("outputs/confusion.png")
    plt.close()

    return jsonify({
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": "confusion.png"
    })

# ---------------------------------------
@app.route("/compare_metrics", methods=["POST"])
def compare_metrics():
    return jsonify({
        "resnet50": {
            "accuracy": 0.83,
            "precision": 0.89,
            "recall": 0.84,
            "f1": 0.82
        },
        "vgg16": {
            "accuracy": 0.78,
            "precision": 0.81,
            "recall": 0.79,
            "f1": 0.80
        }
    })

# ---------------------------------------
@app.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files["image"]
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    file.save(path)
    return jsonify({"path": path})

# ---------------------------------------
@app.route("/download_features")
def download_features():
    from zipfile import ZipFile

    run_id = request.args.get("run_id")
    if not run_id:
        return "No run selected", 400

    folder = os.path.join("outputs", run_id)
    zip_path = os.path.join(folder, "feature_maps.zip")

    with ZipFile(zip_path, "w") as zipf:
        for file in os.listdir(folder):
            if file.endswith(".png"):
                zipf.write(os.path.join(folder, file), arcname=file)

    return send_from_directory(folder, "feature_maps.zip", as_attachment=True)


# ---------------------------------------
@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory("outputs", filename)

if __name__ == "__main__":
    app.run(debug=True)
