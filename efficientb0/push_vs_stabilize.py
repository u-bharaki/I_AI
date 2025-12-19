import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from config import *

def evaluate_and_compare_cm():
    print("Veriler hazırlanıyor...")
    df = pd.read_csv(CSV_FILE)
    df["label_id"] = df[LABEL_COLUMN].astype("category").cat.codes
    class_names = df[LABEL_COLUMN].astype("category").cat.categories.tolist()

    _, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=RANDOM_STATE)

    def process_test_image(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=CHANNELS)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        img = tf.cast(img, tf.float32)
        return preprocess_input(img)

    test_paths = [os.path.join(DATA_ROOT, x) if not os.path.isabs(x) else x for x in test_df[IMAGE_COLUMN]]
    test_labels = test_df["label_id"].values
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths).map(process_test_image).batch(BATCH_SIZE)

    model_paths = {
        "PUSH": "best_model_ultra.keras",
        "STABILIZE": "best_model_stabilized.keras"
    }

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for i, (name, path) in enumerate(model_paths.items()):
        if not os.path.exists(path):
            print(f"HATA: {path} bulunamadı!")
            continue

        print(f"{name} Modeli yükleniyor ve tahmin yapılıyor...")
        model = tf.keras.models.load_model(path)
        preds = model.predict(test_ds)
        y_pred = np.argmax(preds, axis=1)

        cm = confusion_matrix(test_labels, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f'{name} Modeli Karışıklık Matrisi')
        axes[i].set_ylabel('Gerçek Sınıf')
        axes[i].set_xlabel('Tahmin Edilen Sınıf')

    plt.tight_layout()
    plt.savefig("push_vs_stabilize_cm.png", dpi=300)
    print("✅ Karşılaştırmalı matris 'push_vs_stabilize_cm.png' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    evaluate_and_compare_cm()