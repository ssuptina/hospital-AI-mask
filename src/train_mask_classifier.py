"""
Enhanced Mask Classifier Training Script
3-Class: Mask Correct | Mask Incorrect | No Mask
Architecture: MobileNetV2 with Transfer Learning
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import MODEL_CONFIG, CLASS_LABELS

# TensorFlow
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print(f"TensorFlow version: {tf.__version__}")
print(f"Classes: {CLASS_LABELS}")

DATASET_DIR = MODEL_CONFIG["DATASET_DIR"]
IMAGE_SIZE  = MODEL_CONFIG["IMAGE_SIZE"]
BATCH_SIZE  = MODEL_CONFIG["BATCH_SIZE"]
EPOCHS      = MODEL_CONFIG["EPOCHS"]
LR          = MODEL_CONFIG["LEARNING_RATE"]
VAL_SPLIT   = MODEL_CONFIG["VALIDATION_SPLIT"]

# ── Data generators ──────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VAL_SPLIT,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VAL_SPLIT
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\nTraining samples  : {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Classes           : {train_gen.class_indices}")

# Save class index mapping
idx_map = {int(v): k for k, v in train_gen.class_indices.items()}
os.makedirs(str(BASE_DIR), exist_ok=True)
with open(str(BASE_DIR / "class_indices.json"), "w") as f:
    json.dump(idx_map, f, indent=2)
print(f"Class indices saved to: {BASE_DIR / 'class_indices.json'}")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_SAVE_PATH = str(BASE_DIR / "best_model.h5")

if os.path.exists(MODEL_SAVE_PATH):
    print(f"\n[INFO] Loading existing model from {MODEL_SAVE_PATH} to resume training...")
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_SAVE_PATH)
    
    # ── Stage: Resume Fine-Tuning ───────────────────────────────────────────────
    print("\n[Stage] Resuming fine-tuning for additional epochs...")
    model.compile(optimizer=Adam(LR * 0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]
    
    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=15, callbacks=callbacks, verbose=1
    )
    
else:
    print("\n[INFO] No existing model found. Starting from scratch...")
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(CLASS_LABELS), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    print("\n[Stage 1] Training head (frozen base)...")
    model.fit(train_gen, validation_data=val_gen, epochs=5, callbacks=callbacks, verbose=1)

    print("\n[Stage 2] Fine-tuning last 50 layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    model.compile(optimizer=Adam(LR * 0.1), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks, verbose=1)

    print("\n[Stage 3] Full fine-tuning...")
    base_model.trainable = True
    model.compile(optimizer=Adam(LR * 0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks, verbose=1)

print(f"\nModel saved: {MODEL_SAVE_PATH}")
print("Training complete!")
