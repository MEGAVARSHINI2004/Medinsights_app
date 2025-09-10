from flask import Flask, request
import cv2
import numpy as np
from skimage.feature import hog
import joblib
import os

app = Flask(__name__)
model = joblib.load("cancer_model.pkl")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file uploaded"
        file = request.files["image"]
        if file.filename == "":
            return "No file selected"

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Preprocess uploaded image
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True)
        features = features.reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]

        return f"Prediction: {prediction}"

    return '''
    <h2>Upload Skin Image</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image"/>
        <input type="submit" value="Predict"/>
    </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
