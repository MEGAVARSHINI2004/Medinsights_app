import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import cv2

# -------------------------------
# Paths
CANCER_DIR = r"C:\Users\megav\Medinsights\dataset\HAM10000_images_part_1"
NORMAL_DIR = r"C:\Users\megav\Medinsights\dataset\Normal skin"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# -------------------------------
# Load Images
def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    return images, labels

# Cancer classes (from dataset)
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # cancer classes
normal_class = 'normal'

X = []
y = []

# Load cancer images (label = 1)
for c in classes:
    cancer_images, cancer_labels = load_images(CANCER_DIR, 1)
    X.extend(cancer_images)
    y.extend(cancer_labels)
    break  # since all in one folder, break after 1st loop

# Load normal images (label = 0)
normal_images, normal_labels = load_images(NORMAL_DIR, 0)
X.extend(normal_images)
y.extend(normal_labels)

X = np.array(X) / 255.0
y = np.array(y)

print(f"✅ Total Images: {len(X)} | Normal: {np.sum(y==0)} | Cancer: {np.sum(y==1)}")

# -------------------------------
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -------------------------------
# Class Weights for imbalance
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
class_weights = dict(enumerate(class_weights))
print("✅ Class Weights:", class_weights)

# -------------------------------
# Data Augmentation
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True)
datagen.fit(X_train)

# -------------------------------
# Model: MobileNetV2 Transfer Learning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base model initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=preds)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------------
# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# -------------------------------
# Train Model
history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(X_test, y_test),
                    epochs=EPOCHS,
                    class_weight=class_weights,
                    callbacks=[early_stop, checkpoint])

# -------------------------------
# Save Final Model
model.save("skin_cancer_model.h5")
print("✅ Model saved as skin_cancer_model.h5")
