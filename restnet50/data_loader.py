import pandas as pd
import tensorflow as tf
import os
from config import *
from tensorflow.keras.applications.resnet50 import preprocess_input


def load_dataframe():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"{CSV_FILE} bulunamadı.")
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

    # ResNet50 Preprocessing (RGB -> BGR, Zero-center)
    img = preprocess_input(img)

    label = tf.one_hot(label, NUM_CLASSES)
    return img, label


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomZoom(0.1)
])


def augment_image(img, lbl):
    img = data_augmentation(img, training=True)
    return img, lbl


def dataframe_to_dataset(df, shuffle=True, repeat=False, augment=False):
    file_paths = [resolve_path(x) for x in df[IMAGE_COLUMN]]
    labels = df["label_id"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2048)

    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)

    if repeat:
        ds = ds.repeat()

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds