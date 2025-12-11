import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from config import *

def evaluate_model():
    print(f"Test Resim Boyutu: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("Final Model Test Ediliyor...")

    # 1. Veri Hazırlığı
    if not os.path.exists(CSV_FILE):
        print(f"HATA: CSV dosyası bulunamadı: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    df["label_id"] = df[LABEL_COLUMN].astype("category").cat.codes
    class_names = df[LABEL_COLUMN].astype("category").cat.categories.tolist()

    # Splitler birebir aynı olmalı
    _, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=RANDOM_STATE)

    print(f"Toplam Test Edilecek Resim Sayısı: {len(test_df)}")

    test_paths = [os.path.join(DATA_ROOT, x) if not os.path.isabs(x) else x for x in test_df[IMAGE_COLUMN]]
    test_labels = test_df["label_id"].tolist()

    def process_image_val(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=CHANNELS)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        img = tf.cast(img, tf.float32)
        return preprocess_input(img)

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(process_image_val).batch(BATCH_SIZE)

    # 2. Final Model Yükleniyor
    model_path = "best_model_ultra.keras"
    if not os.path.exists(model_path):
        print(f"HATA: '{model_path}' bulunamadı! Önce train_final.py çalıştırılmalı.")
        return

    print(f"Model yükleniyor: {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    # 3. Tahmin
    print("Tahminler hesaplanıyor...")
    predictions = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_labels

    # 4. Raporlama
    print("\n" + "="*60)
    print("FİNAL PERFORMANS RAPORU")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Final Model Confusion Matrix')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.show()

if __name__ == "__main__":
    evaluate_model()