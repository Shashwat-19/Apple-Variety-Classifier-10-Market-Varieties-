import os
import time
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Base directory for the Flask app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pointing to the model inside the existing streamlit app's directory to avoid duplicating large files
MODEL_PATH = os.path.join(BASE_DIR, "apple-variety-streamlit", "model", "apple_classifier_final.keras")
LABELS_PATH = os.path.join(BASE_DIR, "apple-variety-streamlit", "labels.json")
CONFIDENCE_THRESHOLD = 0.75

# Global variables for model and labels
model = None
labels = {}

def load_resources():
    global model, labels
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")

        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, 'r') as f:
                labels = json.load(f)
            print("Labels loaded successfully.")
        else:
            print(f"Warning: Labels not found at {LABELS_PATH}")
    except Exception as e:
        print(f"Error loading resources: {e}")

# Preprocess image consistent with EfficientNetV2 requirements
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    
    # EfficientNetV2S preprocessing
    return tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

# Load model and labels upon script execution
load_resources()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or not labels:
        return jsonify({"error": "Model or labels not loaded on the server."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading."}), 400

    if not file:
         return jsonify({"error": "Invalid file."}), 400
         
    try:
        image = Image.open(file.stream)
        processed_img = preprocess_image(image)
        
        start_time = time.time()
        predictions = model.predict(processed_img)
        inference_time = time.time() - start_time
        
        # Softmax to get probabilities (if not already applied in output layer, but tf.nn.softmax handles logits/probs safely)
        scores = tf.nn.softmax(predictions[0]).numpy() if predictions.shape[-1] > 1 else predictions[0]
        
        top_k = 5
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        top_score = float(predictions[0][top_indices[0]])
        top_class_idx = top_indices[0]
        top_class_name = labels.get(str(top_class_idx), f"Class {top_class_idx}")
        
        # Determine confidence level
        is_high_confidence = top_score >= CONFIDENCE_THRESHOLD
        
        # Top 5 predictions for frontend
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                "class_name": labels.get(str(idx), f"Class {idx}"),
                "score": float(predictions[0][idx])
            })

        response_data = {
            "top_class": top_class_name,
            "confidence": top_score,
            "inference_time_ms": round(inference_time * 1000, 2),
            "is_high_confidence": is_high_confidence,
            "top_predictions": top_predictions,
            "threshold": CONFIDENCE_THRESHOLD
        }
        
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5001)
