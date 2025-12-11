import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from config import *
from data_loader import load_dataframe, resolve_path
from tensorflow.keras.applications.resnet50 import preprocess_input

def evaluate_model():
    print("Final Model Test Ediliyor (ResNet50)...")

    if not os.path.exists(CSV_FILE):
        print("CSV bulunamadı.")
        return

    df = load_dataframe()
    class_names = df[LABEL_COLUMN].astype("category").cat.categories.tolist()

    # Split (Train ile aynı random_state olmalı)
    _, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_STATE)
    _, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=RANDOM_STATE)

    print(f"Test Edilecek Resim Sayısı: {len(test_df)}")

    test_paths = [resolve_path(x) for x in test_df[IMAGE_COLUMN]]
    test_labels = test_df["label_id"].tolist()

    def process_test_image(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=CHANNELS)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        img = preprocess_input(img) # ResNet Preprocessing
        return img

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(process_test_image).batch(BATCH_SIZE)

    model_path = "best_resnet50_model.keras"
    if not os.path.exists(model_path):
        print(f"{model_path} bulunamadı!")
        return

    model = tf.keras.models.load_model(model_path)

    print("Tahminler alınıyor...")
    predictions = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_labels

    print("\n" + "="*60)
    print("RESNET50 PERFORMANS RAPORU")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('ResNet50 Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.show()

if __name__ == "__main__":
    evaluate_model()