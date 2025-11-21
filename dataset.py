# dataset.py

import pandas as pd
import tensorflow as tf
from config import IMAGE_SIZE, CHANNELS, NUM_CLASSES, BATCH_SIZE
import os

DATA_ROOT = 'C:/Users/duygu/IdeaProjects/yap470projesel/preprocessed_images'
CSV_FILE = 'C:/Users/duygu/IdeaProjects/yap470projesel/cleaned_file_final.csv'
LABEL_COLUMN = 'labels'
IMAGE_COLUMN = 'filepath'


def load_dataframe():
    """CSV dosyasını yükler ve gerekli sütunları hazırlar."""
    df = pd.read_csv(CSV_FILE)
    df['label_id'] = df[LABEL_COLUMN].astype('category').cat.codes
    return df

def process_image(file_path, label):
    """
    Görüntü dosyasını okur, yeniden boyutlandırır ve normalleştirir.
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=CHANNELS)

    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    label = tf.one_hot(label, depth=NUM_CLASSES)

    return img, label

def augment_image(img, lbl):
    """Sadece TRAIN için data augmentation uygular."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

    img = tf.clip_by_value(img, 0.0, 1.0)

    return img, lbl

def dataframe_to_dataset(df, shuffle=True, repeat=False):
    """
    Pandas DataFrame'i tf.data.Dataset objesine dönüştürür.
    """
    file_paths = df[IMAGE_COLUMN].apply(lambda x: os.path.join(DATA_ROOT, x)).tolist()
    labels = df["label_id"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    ds = ds.map(
        lambda x, y: process_image(x, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        ds = ds.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.shuffle(1000)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds