import os, io, json, base64
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_model", "mask_detector.h5")
MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_model", "class_indices.json")

# Lazy load
_model = None
_idx_to_class = None

def load_model_and_map():
    global _model, _idx_to_class
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if _idx_to_class is None:
        with open(MAP_PATH, "r") as f:
            _idx_to_class = json.load(f)

def preprocess_bgr(img_bgr, size=(224, 224)):
    img = cv2.resize(img_bgr, size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

@app.route("/", methods=["GET"])
def index():
    has_model = os.path.exists(MODEL_PATH) and os.path.exists(MAP_PATH)
    return render_template("index.html", has_model=has_model)

@app.route("/predict", methods=["POST"])
def predict():
    has_model = os.path.exists(MODEL_PATH) and os.path.exists(MAP_PATH)
    if not has_model:
        return jsonify({"error": "Model not found. Please train first."}), 400

    load_model_and_map()

    if "image" in request.files:
        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        # Expect base64 from webcam canvas
        data_url = request.form.get("image_base64") or request.json.get("image_base64")
        if not data_url:
            return jsonify({"error": "No image provided"}), 400
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    pred = _model.predict(preprocess_bgr(img), verbose=0)[0]
    cls_id = int(np.argmax(pred))
    conf = float(np.max(pred))
    cls_name = _idx_to_class[str(cls_id)]

    #     MASK MESSAGE LOGIC
    if cls_name == "with_mask":
        message = "Welcome, you are wearing mask"
    else:
        message = "To enter this place please wear mask"

    return jsonify({
        "class": cls_name,
        "confidence": round(conf * 100, 2),
        "message": message
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

