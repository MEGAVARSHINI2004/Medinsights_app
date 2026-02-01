# Skin Cancer Detection Web Application

A **web-based application** that uses a **Convolutional Neural Network (CNN)** to detect various types of skin cancer from images. This project provides predictions along with information on remedies, doctors, and diet suggestions for better awareness.


## Overview

Skin cancer is one of the most common types of cancer worldwide. Early detection is critical for effective treatment. This project uses deep learning to classify skin images into different categories, including both cancerous and normal skin.  

The system predicts the type of lesion and provides:  
- Recommended remedies  
- Suggested doctors  
- Diet recommendations


## Features

- Upload skin images (JPEG/PNG) via a simple web interface.  
- Detects multiple types of skin cancer:
  - Actinic keratoses (AKIEC)  
  - Basal cell carcinoma (BCC)  
  - Benign keratosis-like lesions (BKL)  
  - Dermatofibroma (DF)  
  - Melanoma (MEL)  
  - Melanocytic nevi (NV)  
  - Vascular lesions (VASC)  
- Recognizes **Normal skin**.  
- Displays uploaded image alongside prediction results.  
- Provides additional information for awareness and treatment.

  

## Dataset

- **HAM10000 dataset** for cancerous skin images.  
- Separate folder for **Normal skin images**.  
- Images resized to 224x224 pixels for CNN input.  



## Installation

1. Clone the repository:

git clone (https://github.com/MEGAVARSHINI2004/Medinsights_app/)
cd skin-cancer-detection
Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies:


pip install -r requirements.txt
Usage
Run the Flask application:
python app.py
Open your browser and navigate to:
http://127.0.0.1:5000/

Upload a skin image and click Predict.
View the prediction, remedy, doctors, and diet suggestions.

Project Structure:
Skin-Cancer-Detection/
│
├── app.py                # Flask web app
├── train.py              # Model training script
├── med.py                # Dataset loading and preprocessing
├── skin_cancer_model.h5  # Trained CNN model
├── templates/
│   ├── index1.html       # Upload page
│   └── result.html       # Prediction results page
├── static/uploads/       # Folder to store uploaded images
└── README.md



Technologies Used
Python 3.11
TensorFlow & Keras
Flask (Web Framework)
Bootstrap 5 (Frontend Styling)
NumPy & OpenCV (Image Processing)

Contributing
Contributions are welcome! If you want to improve this project, you can:
Add more skin image datasets
Enhance model accuracy
Improve frontend UI/UX
Add additional health recommendations
Please create a pull request or open an issue to discuss changes.

Screenshot:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/94028952-54f5-4bcf-8f60-863e34208e52" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/496c50c2-5b82-4800-93a2-6643467d7e6f" />


