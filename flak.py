import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__)

# Load trained CNN model
MODEL_PATH = "skin_cancer_model.h5"
model = load_model(MODEL_PATH)

# Model input size
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Class labels (full names for clarity)
CLASS_LABELS = [
    "Actinic keratoses (AKIEC)",
    "Basal cell carcinoma (BCC)",
    "Benign keratosis-like lesions (BKL)",
    "Dermatofibroma (DF)",
    "Melanoma (MEL)",
    "Melanocytic nevi (NV)",
    "Vascular lesions (VASC)",
    "Normal skin"
]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.6  # You can adjust this value

# Knowledge base for remedies, doctors, diet
CANCER_INFO = {
    "Actinic keratoses (AKIEC)": {
        "remedy": "Topical treatments (5-fluorouracil, imiquimod) and cryotherapy are common.",
        "doctors": "Dr. A. Mehta (Apollo Hospitals), Dr. R. Kumar (AIIMS)",
        "diet": "Rich in antioxidants: tomatoes, leafy greens, citrus fruits."
    },
    "Basal cell carcinoma (BCC)": {
        "remedy": "Surgical excision, Mohs surgery, or targeted therapy.",
        "doctors": "Dr. S. Rao (Fortis), Dr. P. Gupta (CMC Vellore)",
        "diet": "Omega-3 rich foods: fish, walnuts, flax seeds."
    },
    "Benign keratosis-like lesions (BKL)": {
        "remedy": "Usually harmless. Removal only for cosmetic reasons via cryotherapy or curettage.",
        "doctors": "Dr. N. Sharma (Apollo), Dr. A. Das (Manipal Hospitals)",
        "diet": "Balanced diet with fruits, vegetables, and hydration."
    },
    "Dermatofibroma (DF)": {
        "remedy": "Generally harmless. Excision only if painful or growing.",
        "doctors": "Dr. V. Nair (AIIMS), Dr. S. Singh (Max Healthcare)",
        "diet": "Protein-rich diet with lean meat, legumes, and vitamin C foods."
    },
    "Melanoma (MEL)": {
        "remedy": "Surgical removal, immunotherapy, targeted therapy.",
        "doctors": "Dr. R. Menon (Tata Memorial), Dr. S. Chawla (Apollo)",
        "diet": "Dark leafy greens, berries, vitamin D-rich foods, and green tea."
    },
    "Melanocytic nevi (NV)": {
        "remedy": "Usually benign. Monitor for changes; removal if suspicious.",
        "doctors": "Dr. P. Iyer (Fortis), Dr. A. Khan (Apollo)",
        "diet": "General healthy diet with vitamins A, C, and E."
    },
    "Vascular lesions (VASC)": {
        "remedy": "Laser therapy or sclerotherapy if treatment is required.",
        "doctors": "Dr. K. Patel (AIIMS), Dr. D. Reddy (Apollo)",
        "diet": "Iron-rich foods (spinach, beans, fish) to support blood health."
    },
    "Normal skin": {
        "remedy": "No cancer detected. Maintain good skincare and sun protection.",
        "doctors": "General dermatologist consultation not required unless symptoms develop.",
        "diet": "Balanced diet with hydration, fruits, and vegetables."
    }
}

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index1.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    try:
        # Preprocess image
        img = image.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        preds = model.predict(img_array)
        confidence = np.max(preds)
        predicted_class = CLASS_LABELS[np.argmax(preds)]

        # Apply confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_class = "Normal skin"

        info = CANCER_INFO.get(predicted_class, {})

        if predicted_class == "Normal skin":
            result_text = f"No cancer detected. (Confidence: {confidence:.2f})"
        else:
            result_text = f"Cancer detected: {predicted_class} (Confidence: {confidence:.2f})"

        return render_template("result.html",
                               prediction=result_text,
                               remedy=info.get("remedy", ""),
                               doctors=info.get("doctors", ""),
                               diet=info.get("diet", ""))
    except Exception as e:
        return f"Error during prediction: {str(e)}", 500

# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
