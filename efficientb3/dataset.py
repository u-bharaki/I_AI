# dataset.py

import tensorflow as tf
import pandas as pd
import os
from config import *
from retina_preprocess import retina_preprocess
import numpy as np

def load_dataframe():
    df = pd.read_csv(CSV_FILE)
    df["label_id"] = df[LABEL_COLUMN].astype("category").cat.codes
    return df

def resolve_path(x):
    if os.path.isabs(x):
        return x
    return os.path.join(DATA_ROOT, x)

# --------------------
# Preprocessing
# --------------------
def decode_and_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=CHANNELS)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    # retina preprocessing
    img = tf.py_function(retina_preprocess, [img], Tout=tf.float32)
    img.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    label = tf.one_hot(label, NUM_CLASSES)
    return img, label

# --------------------
# Augmentation
# --------------------
augment_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.GaussianNoise(0.05),
])

def augment(img, label):
    img = augment_layer(img)
    return img, label

# --------------------
# Balanced sampler
# --------------------
def balanced_dataset(df):
    datasets = []
    class_counts = df["label_id"].value_counts().to_dict()
    max_count = max(class_counts.values())

    for cls in range(NUM_CLASSES):
        df_cls = df[df["label_id"] == cls]
        paths = [resolve_path(p) for p in df_cls[IMAGE_COLUMN]]
        labels = df_cls["label_id"].tolist()

        # Oversample to max_count
        rep = max_count // len(paths) + 1
        paths = paths * rep
        labels = labels * rep

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        datasets.append(ds)

    final = tf.data.Dataset.sample_from_datasets(datasets, weights=None)
    final = final.shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return final

# Normal val/test dataset
def build_eval_dataset(df):
    paths = [resolve_path(x) for x in df[IMAGE_COLUMN]]
    labels = df["label_id"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
