import tensorflow as tf
import os

IMAGE_SIZE = 224
CHANNELS = 3
NUM_CLASSES = 8
BATCH_SIZE = 32
RANDOM_STATE = 42

CSV_FILE = r"..\cleaned_file_final.csv"
DATA_ROOT = r"..\preprocessed_images"
LABEL_COLUMN = "labels"
IMAGE_COLUMN = "filepath"

ROTATION_FACTOR = 0.1
ZOOM_FACTOR = 0.1

augmentation_layers = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(ROTATION_FACTOR),
    tf.keras.layers.RandomZoom(ZOOM_FACTOR),
])