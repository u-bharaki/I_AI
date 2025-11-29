# dataset.py – TFA'sız, tamamen uyumlu

import pandas as pd
import tensorflow as tf
import os
from config import *


def load_dataframe():
    df = pd.read_csv(CSV_FILE)
    df["label_id"] = df[LABEL_COLUMN].astype("category").cat.codes
    return df


def resolve_path(x):
    if os.path.isabs(x):
        return x
    return os.path.join(DATA_ROOT, x)


def process_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=CHANNELS)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label


# ---- TFA OLMADAN ROTATION ----
rotation_layer = tf.keras.layers.RandomRotation(
    factor=0.05,         # ~5 derece
    fill_mode="nearest"  # siyah boşluk oluşmasın
)


def augment_image(img, lbl):
    # horizontal flip
    img = tf.image.random_flip_left_right(img)

    # rotation
    img = rotation_layer(img)

    # brightness
    img = tf.image.random_brightness(img, 0.05)

    # contrast
    img = tf.image.random_contrast(img, 0.95, 1.05)

    return img, lbl


def dataframe_to_dataset(df, shuffle=True, repeat=False):
    file_paths = [resolve_path(x) for x in df[IMAGE_COLUMN]]
    labels = df["label_id"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).cache()

    if repeat:
        ds = ds.repeat()

    return ds
