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
df = pd.read_csv(ham_metadata)

# Only use images that exist in part1
available_images = {f.replace(".jpg", "") for f in os.listdir(ham_images_path)}
df = df[df["image_id"].isin(available_images)]
print("✅ Using only HAM10000 part1 images:", len(df))

images = []
labels = []

# Load HAM10000 images
for i, row in df.iterrows():
    img_file = os.path.join(ham_images_path, row["image_id"] + ".jpg")
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
normal_files = [f for f in os.listdir(normal_skin_path) if f.lower().endswith(".jpg")]
print("✅ Normal skin images found:", len(normal_files))

for f in normal_files:
    img_file = os.path.join(normal_skin_path, f)
    img = cv2.imread(img_file)
    if img is None:
        continue
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    images.append(img)
    labels.append("Normal skin")   # label for normal

# -------------------------------
# Convert to numpy
# -------------------------------
images = np.array(images)
labels = pd.Categorical(labels)   # unique string labels
labels_cat = to_categorical(labels.codes)

print("✅ Total images:", images.shape[0])
print("✅ Classes:", list(labels.categories))

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_cat, test_size=0.2, random_state=42, stratify=labels.codes
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Number of classes:", len(labels.categories))
