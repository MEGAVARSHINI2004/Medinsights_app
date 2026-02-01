import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# -------------------------------
# Paths
# -------------------------------
ham_metadata = "dataset/HAM10000_metadata.csv"
ham_images_path = "dataset/HAM10000_images_part_1/"
normal_skin_path = "dataset/Normal skin/"

# -------------------------------
# Load HAM10000 metadata & images
# -------------------------------
def load_and_preprocess_data():
    df = pd.read_csv(ham_metadata)
    
    # Check for images in both part1 and part2
    part1_path = "dataset/HAM10000_images_part_1/"
    part2_path = "dataset/HAM10000_images_part_2/"
    
    # Get available images from part1
    available_images_part1 = set()
    if os.path.exists(part1_path):
        available_images_part1 = {f.replace(".jpg", "") for f in os.listdir(part1_path) if f.endswith(".jpg")}
    
    # Get available images from part2
    available_images_part2 = set()
    if os.path.exists(part2_path):
        available_images_part2 = {f.replace(".jpg", "") for f in os.listdir(part2_path) if f.endswith(".jpg")}
    
    # Combine both sets
    available_images = available_images_part1.union(available_images_part2)
    
    df = df[df["image_id"].isin(available_images)]
    print("✅ Using HAM10000 part1 & part2 images:", len(df))

    images = []
    labels = []

    # Load HAM10000 images
    for i, row in df.iterrows():
        img_file = os.path.join(part1_path, row["image_id"] + ".jpg")
        
        # If not found in part1, try part2
        if not os.path.exists(img_file):
            img_file = os.path.join(part2_path, row["image_id"] + ".jpg")
        
        img = cv2.imread(img_file)
        if img is None:
            continue
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        images.append(img)
        labels.append(row["dx"])   # disease label

    # -------------------------------
    # Add Normal skin dataset
    # -------------------------------
    normal_files = [f for f in os.listdir(normal_skin_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print("✅ Normal skin images found:", len(normal_files))

    for f in normal_files:
        img_file = os.path.join(normal_skin_path, f)
        img = cv2.imread(img_file)
        if img is None:
            continue
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        images.append(img)
        labels.append("normal")   # consistent label for normal

    # -------------------------------
    # Convert to numpy
    # -------------------------------
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

def get_class_mapping():
    return {
        'akiec': "Actinic keratoses (AKIEC)",
        'bcc': "Basal cell carcinoma (BCC)", 
        'bkl': "Benign keratosis-like lesions (BKL)",
        'df': "Dermatofibroma (DF)",
        'mel': "Melanoma (MEL)",
        'nv': "Melanocytic nevi (NV)",
        'vasc': "Vascular lesions (VASC)",
        'normal': "Normal skin"
    }

if __name__ == "__main__":
    images, labels = load_and_preprocess_data()
    print("✅ Total images:", images.shape[0])
    print("✅ Unique labels:", np.unique(labels))