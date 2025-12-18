import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from config import *

def load_dataframe():
    df = pd.read_csv(CSV_FILE)
    df["label_id"] = df[LABEL_COLUMN].apply(lambda x: CLASS_NAMES.index(x))
    return df

def resolve_path(x):
    return x if os.path.isabs(x) else os.path.join(DATA_ROOT, x)

# --- AUGMENTATION (SADE & TIBBİ UYGUN) ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomContrast(0.1),
])

def process_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=CHANNELS)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    return img, label

def augment_image(img, lbl):
    img = data_augmentation(img, training=True)
    img = preprocess_input(img)
    lbl = tf.one_hot(lbl, NUM_CLASSES)
    return img, lbl

def normalize_only(img, lbl):
    img = preprocess_input(img)
    lbl = tf.one_hot(lbl, NUM_CLASSES)
    return img, lbl

def dataframe_to_dataset(df, augment=False, shuffle=True, repeat=False):
    paths = [resolve_path(x) for x in df[IMAGE_COLUMN]]
    labels = df["label_id"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2048)

    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    if repeat:
        ds = ds.repeat()

    return ds.prefetch(tf.data.AUTOTUNE)
