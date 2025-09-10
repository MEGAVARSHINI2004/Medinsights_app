# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load models & encoders
model = joblib.load("random_forest_model.pkl")  # or xgb_model.pkl
sex_encoder = joblib.load("sex_encoder.pkl")
loc_encoder = joblib.load("loc_encoder.pkl")
dx_encoder = joblib.load("dx_encoder.pkl")

# Load metadata to fit scaler
df = pd.read_csv("dataset/HAM10000_metadata.csv").dropna()
scaler_features = df[["age", "sex", "localization"]].copy()
scaler_features["sex"] = sex_encoder.transform(df["sex"])
scaler_features["localization"] = loc_encoder.transform(df["localization"])
scaler = StandardScaler()
scaler.fit(scaler_features)

# Dropdown options
VALID_SEX = sex_encoder.classes_.tolist()
VALID_LOCALIZATION = loc_encoder.classes_.tolist()

# Disease Mapping
disease_map = {
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis-like Lesions",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratoses",
    "vasc": "Vascular Lesions",
    "df": "Dermatofibroma"
}

app = Flask(__name__)

@app.route("/")
def home():
    return render_template(
        "index.html",
        sexes=VALID_SEX,
        localizations=VALID_LOCALIZATION
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        sex = request.form["sex"]
        localization = request.form["localization"]

        sex_encoded = sex_encoder.transform([sex])[0]
        loc_encoded = loc_encoder.transform([localization])[0]

        input_data = np.array([[age, sex_encoded, loc_encoded]])
        input_scaled = scaler.transform(input_data)

        prediction_num = model.predict(input_scaled)[0]
        prediction_code = dx_encoder.inverse_transform([prediction_num])[0]
        disease_name = disease_map.get(prediction_code, prediction_code)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Disease: {disease_name}",
            sexes=VALID_SEX,
            localizations=VALID_LOCALIZATION
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            sexes=VALID_SEX,
            localizations=VALID_LOCALIZATION
        )

if __name__ == "__main__":
    app.run(debug=True)
