import pandas as pd
import tensorflow as tf
import os
from config import *
from tensorflow.keras.applications.efficientnet import preprocess_input


def load_dataframe():
    df = pd.read_csv(CSV_FILE)
    df["label_id"] = df[LABEL_COLUMN].astype("category").cat.codes
    return df


def resolve_path(x):
    if os.path.isabs(x):
        return x
    return os.path.join(DATA_ROOT, x)


# EfficientNetB0 preprocessing + decoding
def process_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=CHANNELS)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32)

    img = preprocess_input(img)  # EfficientNet preprocessing

    label = tf.one_hot(label, NUM_CLASSES)
    return img, label


# Augmentation — optimize edilmiş
rotation_layer = tf.keras.layers.RandomRotation(factor=0.05, fill_mode="reflect")

def augment_image(img, lbl):
    img = tf.image.random_flip_left_right(img)
    img = rotation_layer(img)
    img = tf.image.random_brightness(img, max_delta=0.05)
    return img, lbl


# *** En doğru data pipeline ***
def dataframe_to_dataset(df, shuffle=True, repeat=False):
    file_paths = [resolve_path(x) for x in df[IMAGE_COLUMN]]
    labels = df["label_id"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()   # → repeat hatasını önleyen en önemli adım

    if shuffle:
        ds = ds.shuffle(2048)

    ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)

    if repeat:
        ds = ds.repeat()

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
