import os
import json
import uuid
from datetime import datetime

os.environ["KERAS_BACKEND"] = "numpy"

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import keras

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ["Fresh", "Slightly Aged", "Stale", "Spoiled", "Rotten"]
CLASS_DESCRIPTIONS = {
    "Fresh": "This item looks fresh and safe to consume. No visible signs of aging or spoilage.",
    "Slightly Aged": "Minor signs of aging detected. Still safe for consumption but best used soon.",
    "Stale": "Noticeable aging present. Quality has declined — consume with caution.",
    "Spoiled": "Significant spoilage detected. This item is not recommended for consumption.",
    "Rotten": "Severe decay detected. This item should be discarded immediately.",
}

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")

# Candidate paths for model
_CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(__file__), "models", "MobileNetV2_best_model.keras"),
    os.path.join(os.path.dirname(__file__), "..", "models", "MobileNetV2_best_model.keras"),
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "Food-Freshness-Detection-Using-Deep-Learning-main",
        "Output Files",
        "agglomerative_saved_models",
        "MobileNetV2_best_model.keras",
    ),
]
MODEL_PATH = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)), _CANDIDATE_PATHS[0])

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
print("Loading MobileNetV2 model …")
model = keras.saving.load_model(MODEL_PATH)
print("Model loaded successfully.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, normalise, and batch-dim an image for the model."""
    image = image.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history: list):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save history: {e}", flush=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/presentation")
def presentation():
    return render_template("presentation.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        return jsonify({"error": "Unsupported file type. Please upload a JPG, PNG, or WebP image."}), 400

    unique_name = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    # Predict
    try:
        with Image.open(filepath) as image:
            img_array = preprocess_image(image)

        # verbose=0 is essential on Windows to prevent Keras Progbar terminal
        # writes which cause [Errno 22] Invalid argument
        prediction = model.predict(img_array, verbose=0)
        probabilities = [float(p) for p in prediction[0]]
        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = round(probabilities[predicted_index] * 100, 1)

        result = {
            "id": unique_name.split(".")[0],
            "filename": os.path.basename(file.filename) or "image",
            "image_url": f"/static/uploads/{unique_name}",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "description": CLASS_DESCRIPTIONS[predicted_class],
            "probabilities": {
                name: round(prob * 100, 1)
                for name, prob in zip(CLASS_NAMES, probabilities)
            },
            "timestamp": datetime.now().strftime("%b %d, %Y at %I:%M %p"),
        }

        # Persist to history safely
        try:
            history = load_history()
            history.insert(0, result)
            save_history(history)
        except Exception as he:
            print(f"Warning: Failed to save history: {he}", flush=True)

        return jsonify(result)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("PREDICT ERROR TRACEBACK:\n", tb, flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(load_history())


@app.route("/history", methods=["DELETE"])
def clear_history():
    save_history([])
    # Clean up uploaded images
    for f in os.listdir(UPLOAD_FOLDER):
        fpath = os.path.join(UPLOAD_FOLDER, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

