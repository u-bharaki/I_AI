import os
import cv2
import numpy as np
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

# --- CLAHE IMPLEMENTATION ---
def apply_clahe_cv2(image):
    """
    OpenCV kullanarak CLAHE uygular.
    Girdi: RGB formatında numpy array (0-255)
    Çıktı: RGB formatında numpy array (float32)
    """
    # TensorFlow tensor'ünü numpy array'e ve uint8'e çevir (OpenCV için)
    image = image.astype(np.uint8)

    # LAB renk uzayına geç
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE oluştur ve L kanalına uygula
    # clipLimit=2.0 (standart), tileGridSize=(8,8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Kanalları birleştir
    limg = cv2.merge((cl, a, b))

    # Tekrar RGB'ye dön
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return final_img.astype(np.float32)

def tf_apply_clahe(image, label):
    """
    Python fonksiyonunu TensorFlow graph içine gömer.
    """
    [image_clahe] = tf.numpy_function(apply_clahe_cv2, [image], [tf.float32])
    # Shape kaybolduğu için tekrar set etmemiz gerekir
    image_clahe.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    return image_clahe, label

# --- AUGMENTATION ---
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

def dataframe_to_dataset(df, augment=False, shuffle=True, repeat=False, use_clahe=True):
    paths = [resolve_path(x) for x in df[IMAGE_COLUMN]]
    labels = df["label_id"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # 1. Resmi oku ve resize et
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    # 2. CLAHE uygula (ResNet Preprocessing'den ÖNCE yapılmalı)
    if use_clahe:
        ds = ds.map(tf_apply_clahe, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2048)

    # 3. Augmentation ve Normalizasyon (ResNet preprocess_input burada çağrılır)
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    if repeat:
        ds = ds.repeat()

    return ds.prefetch(tf.data.AUTOTUNE)