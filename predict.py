import os
import json
import gdown
import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input


# ==========================================================
# Base Path
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_xception.keras"
)

CLASS_JSON = os.path.join(
    BASE_DIR,
    "class_names.json"
)

# ==========================================================
# Google Drive Model
# ==========================================================

FILE_ID = "1umWGiya9JEbBvfY0YcnaOKTfTBx8MUf4"

MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"


# ==========================================================
# Predictor
# ==========================================================

class FruitPredictor:

    def __init__(self, model_path, class_json):

        # Pastikan folder model ada
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Download model jika belum ada
        if not os.path.exists(model_path):

            print("=" * 50)
            print("Downloading Xception Model...")
            print("=" * 50)

            gdown.download(
                MODEL_URL,
                model_path,
                quiet=False
            )

        print("=" * 50)
        print("Loading Xception Model...")
        print("=" * 50)

        self.model = load_model(model_path)

        with open(class_json, "r", encoding="utf-8") as f:

            classes = json.load(f)

        self.class_names = {
            int(k): v
            for k, v in classes.items()
        }

    # ==========================================================
    # Image Preprocessing
    # ==========================================================

    def preprocess_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image = image.resize((224, 224))

        image = np.array(
            image,
            dtype=np.float32
        )

        image = preprocess_input(image)

        image = np.expand_dims(
            image,
            axis=0
        )

        return image

    # ==========================================================
    # Prediction
    # ==========================================================

    def predict(self, image_path):

        image = self.preprocess_image(
            image_path
        )

        predictions = self.model.predict(
            image,
            verbose=0
        )[0]

        best_index = int(
            np.argmax(predictions)
        )

        prediction = self.class_names[
            best_index
        ]

        confidence = float(
            predictions[best_index]
        )

        top_indices = predictions.argsort()[-3:][::-1]

        top3 = []

        for idx in top_indices:

            top3.append({

                "class": self.class_names[int(idx)],

                "confidence": round(
                    float(predictions[idx]) * 100,
                    2
                )

            })

        return {

            "prediction": prediction,

            "confidence": confidence,

            "top3": top3

        }
