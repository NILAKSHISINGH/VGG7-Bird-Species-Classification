from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_vgg7_birds.pth"
)

SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Import existing VGG7 architecture
from vgg7 import VGG7


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "ABBOTTS BABBLER",
    "ABBOTTS BOOBY",
    "ABYSSINIAN GROUND HORNBILL",
    "AFRICAN CROWNED CRANE",
    "AFRICAN EMERALD CUCKOO",
    "AFRICAN FIREFINCH",
    "AFRICAN OYSTER CATCHER",
    "AFRICAN PIED HORNBILL",
    "AFRICAN PYGMY GOOSE",
    "ALBATROSS",
    "ALBERTS TOWHEE",
    "ALEXANDRINE PARAKEET",
    "ALPINE CHOUGH",
    "ALTAMIRA YELLOWTHROAT",
    "AMERICAN AVOCET",
    "AMERICAN BITTERN",
    "AMERICAN COOT",
    "AMERICAN FLAMINGO",
    "AMERICAN GOLDFINCH",
    "AMERICAN KESTREL"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading VGG7 model...")

model = VGG7(num_classes=20)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Handle different checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model" in checkpoint:
        state_dict = checkpoint["model"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


model.load_state_dict(state_dict)

model.to(device)
model.eval()

print("VGG7 model loaded successfully.")
print("Model path:", MODEL_PATH)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "running",
        "message": "BirdVision AI backend is running",
        "model": "VGG7",
        "classes": 20,
        "device": str(device)
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check image
        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded"
            }), 400


        file = request.files["image"]

        # Open image
        image = Image.open(file).convert("RGB")

        # Transform
        image_tensor = transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(device)


        # ====================================================
        # MODEL INFERENCE
        # ====================================================

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = F.softmax(
                outputs,
                dim=1
            )[0]


        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        top_probs, top_indices = torch.topk(
            probabilities,
            3
        )

        predictions = []

        for probability, index in zip(
            top_probs,
            top_indices
        ):

            predictions.append({
                "species": CLASS_NAMES[index.item()],
                "confidence": round(
                    probability.item() * 100,
                    2
                )
            })


        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({
            "success": True,
            "prediction": predictions[0]["species"],
            "confidence": predictions[0]["confidence"],
            "top_predictions": predictions
        })


    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print(" BirdVision AI Backend")
    print(" VGG7 Bird Species Classifier")
    print("========================================")
    print()
    print("Server running at:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )