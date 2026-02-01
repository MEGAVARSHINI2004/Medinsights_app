import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import pandas as pd
from med import load_and_preprocess_data, get_class_mapping

# -------------------------------
# Configuration
# -------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
RANDOM_STATE = 42

# -------------------------------
# Load and prepare data
# -------------------------------
print("🔄 Loading and preprocessing data...")
images, labels = load_and_preprocess_data()

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

print(f"✅ Total Images: {len(images)}")
print(f"✅ Classes: {label_encoder.classes_}")
print(f"✅ Class distribution: {np.bincount(labels_encoded)}")

# Convert to categorical
labels_categorical = to_categorical(labels_encoded, num_classes=num_classes)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_categorical, test_size=0.2, stratify=labels_encoded, random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=np.argmax(y_train, axis=1), random_state=RANDOM_STATE
)

print(f"✅ Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# -------------------------------
# Class Weights for imbalance
# -------------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights = dict(enumerate(class_weights))
print("✅ Class Weights:", class_weights)

# -------------------------------
# Data Augmentation
# -------------------------------
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()  # No augmentation for validation

# -------------------------------
# Model: MobileNetV2 Transfer Learning
# -------------------------------
def create_model(num_classes):
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False  # Freeze base model initially

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

model = create_model(num_classes)
model.summary()

# -------------------------------
# Callbacks
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_skin_cancer_model.h5', 
    monitor='val_accuracy',
    save_best_only=True, 
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# -------------------------------
# Train Model
# -------------------------------
print("🔄 Starting training...")
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE),
    validation_steps=len(X_val) // BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# -------------------------------
# Fine-tuning - FIXED VERSION
# -------------------------------
print("🔄 Starting fine-tuning...")

# Load the best model from initial training
best_model = tf.keras.models.load_model('best_skin_cancer_model.h5')

# Get the base model for fine-tuning
base_model = best_model.layers[1]
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = len(base_model.layers) // 2

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate
best_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001/10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 10
total_epochs = len(history.history['loss']) + fine_tune_epochs

# Fine-tuning callbacks
fine_tune_checkpoint = ModelCheckpoint(
    'final_skin_cancer_model.h5', 
    monitor='val_accuracy',
    save_best_only=True, 
    verbose=1
)

history_fine = best_model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE),
    validation_steps=len(X_val) // BATCH_SIZE,
    epochs=total_epochs,
    initial_epoch=len(history.history['loss']),
    class_weight=class_weights,
    callbacks=[early_stop, fine_tune_checkpoint, reduce_lr],
    verbose=1
)

# Load the best fine-tuned model for evaluation
print("🔄 Loading best fine-tuned model...")
final_model = tf.keras.models.load_model('final_skin_cancer_model.h5')

# -------------------------------
# Evaluate Model
# -------------------------------
test_loss, test_accuracy = final_model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_accuracy:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

# -------------------------------
# Save Final Model and Metadata
# -------------------------------
final_model.save("skin_cancer_model.h5")

# Save label encoder
import joblib
joblib.dump(label_encoder, 'label_encoder.pkl')

print("✅ Model saved as skin_cancer_model.h5")
print("✅ Label encoder saved as label_encoder.pkl")

# Save class mapping
class_mapping = get_class_mapping()
import json
with open('class_mapping.json', 'w') as f:
    json.dump(class_mapping, f)
print("✅ Class mapping saved as class_mapping.json")

print("\n🎯 Training Statistics:")
print(f"   Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"   Final test accuracy: {test_accuracy:.4f}")
print(f"   Dataset size: {len(images)} images")
print(f"   Classes: {num_classes}")