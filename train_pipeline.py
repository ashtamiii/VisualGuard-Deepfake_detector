import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from models.spatial_model import build_spatial_model

IMG_SIZE = 224
BATCH_SIZE = 32

train_dir = r"C:\Users\ashta\VisualGuard-Deepfake-Detection\dataset\Celeb-DF Preprocessed\train"
val_dir = r"C:\Users\ashta\VisualGuard-Deepfake-Detection\dataset\Celeb-DF Preprocessed\val"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

model = build_spatial_model()

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save("models/deepfake_spatial_model.keras")