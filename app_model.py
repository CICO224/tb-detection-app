import io
import os

import joblib
import numpy as np
from PIL import Image


IMG_SIZE = 128
MODEL_PATH = "tb_linear_model.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. Make sure it is included in the deployment."
        )
    return joblib.load(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != "L":
        image = image.convert("L")

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)
    img_flattened = img_array.flatten()
    img_normalized = img_flattened.astype("float32") / 255.0
    return img_normalized.reshape(1, -1)


def predict_image_bytes(image_bytes: bytes) -> dict:
    if not image_bytes:
        raise ValueError("Empty file received")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError(f"Invalid image file: {exc}") from exc

    model = load_model()
    processed_image = preprocess_image(image)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Model must support predict_proba().")

    prediction_proba = model.predict_proba(processed_image)
    probability_tb = float(prediction_proba[0][1])
    predicted_class = int(model.predict(processed_image)[0])
    label = "TB" if predicted_class == 1 else "Normal"

    return {
        "probability": round(probability_tb, 4),
        "label": label,
        "status": "success",
        "raw_probability": round(probability_tb, 4),
    }
