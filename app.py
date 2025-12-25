import os
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model("plant_health_model.h5")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_labels = {v: k for k, v in class_indices.items()}

def get_care_tip(prediction, confidence):
    pred = prediction.lower()

    if confidence < 60:
        return (
            "⚠️ Low confidence prediction. Please upload a clearer image "
            "with good lighting and visible leaf surface."
        )

    if "virus" in pred:
        return (
            "🦠 Viral disease detected. Remove infected leaves immediately. "
            "Disinfect tools and control insects like aphids."
        )

    elif "blight" in pred or "spot" in pred or "mildew" in pred:
        return (
            "🍄 Fungal infection detected. Improve air circulation, "
            "avoid overwatering, and apply a suitable fungicide."
        )

    elif "bacterial" in pred:
        return (
            "🧫 Bacterial disease detected. Remove infected parts and "
            "avoid splashing water on leaves."
        )

    elif "rust" in pred:
        return (
            "🍂 Rust disease detected. Remove affected leaves and "
            "apply fungicide."
        )

    elif "chlorosis" in pred or "manganese" in pred:
        return (
            "🧪 Nutrient deficiency detected. Apply balanced fertilizer "
            "and check soil pH."
        )

    elif "aphid" in pred or "pest" in pred or "infestation" in pred:
        return (
            "🐛 Pest infestation detected. Use neem oil or insecticidal soap."
        )

    elif "healthy" in pred:
        return (
            "🌿 Plant is healthy. Continue regular watering and sunlight."
        )

    else:
        return (
            "🌱 General plant stress detected. Monitor watering, light, "
            "and soil condition."
        )

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    filename = None
    care = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            preds = model.predict(img_array)
            class_id = np.argmax(preds)
            prediction = class_labels[class_id]
            confidence = round(float(np.max(preds)) * 100, 2)
            filename = file.filename

            care = get_care_tip(prediction, confidence)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename,
        care=care
    )


if __name__ == "__main__":
    app.run(debug=True)
