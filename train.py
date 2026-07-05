import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ==========================================
# 1. PATH DATASET
# ==========================================

BASE_DIR = "dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
TEST_DIR = os.path.join(BASE_DIR, "test")

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 2. CONFIG
# ==========================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 8

EPOCHS = 5

LEARNING_RATE = 0.0001

# ==========================================
# 3. DATA AUGMENTATION
# ==========================================

train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=20,

    zoom_range=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    fill_mode="nearest"
)

valid_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# ==========================================
# 4. DATASET
# ==========================================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\n===================================")
print("Classes :")
print(train_generator.class_indices)
print("===================================\n")

# ==========================================
# 5. SAVE CLASS NAMES
# ==========================================

class_names = {
    v: k for k, v in train_generator.class_indices.items()
}

with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("Class names saved!")

# ==========================================
# 6. LOAD XCEPTION
# ==========================================

base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze semua layer pretrained
for layer in base_model.layers:
    layer.trainable = False

# ==========================================
# 7. BUILD MODEL
# ==========================================

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)

outputs = Dense(
    train_generator.num_classes,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=outputs
)

# ==========================================
# 8. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 9. CALLBACKS
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "model/best_xception.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
# ==========================================
# 10. TRAIN MODEL
# ==========================================

print("\n========================================")
print("TRAINING DIMULAI...")
print("========================================\n")

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ]
)

# ==========================================
# 11. SAVE FINAL MODEL
# ==========================================

model.save("model/fruit_xception.keras")

print("\n========================================")
print("FINAL MODEL DISIMPAN")
print("========================================\n")

# ==========================================
# 12. EVALUATION
# ==========================================

print("\n========================================")
print("EVALUASI MODEL")
print("========================================\n")

loss, accuracy = model.evaluate(test_generator)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

# ==========================================
# 13. TRAINING GRAPH
# ==========================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig("model/training_result.png")

plt.show()

print("\n========================================")
print("TRAINING SELESAI")
print("========================================")
