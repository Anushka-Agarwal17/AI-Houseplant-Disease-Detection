import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("plant_health_model.h5")

# Load class labels
with open("class_indices.json") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

def predict_disease(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    confidence = float(np.max(predictions))
    class_idx = np.argmax(predictions)

    disease = class_names[class_idx]

    return {
        "disease": disease,
        "confidence": confidence
    }