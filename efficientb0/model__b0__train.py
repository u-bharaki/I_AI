import pandas as pd
import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.applications.efficientnet import preprocess_input
from datetime import datetime

# Config ve Model yapısını koruyoruz
from config import *

# --- 1. VERİ HAZIRLIĞI ---
def load_dataframe():
    df = pd.read_csv(CSV_FILE)
    df["label_id"] = df[LABEL_COLUMN].astype("category").cat.codes
    return df

def resolve_path(x):
    if os.path.isabs(x): return x
    return os.path.join(DATA_ROOT, x)

def process_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=CHANNELS)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label

# Augmentation'ı biraz azalttık (Sakinleşme evresi)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1), # 0.2 -> 0.1
    tf.keras.layers.RandomContrast(0.1),
])

def augment_image(img, lbl):
    img = data_augmentation(img, training=True)
    return img, lbl

def dataframe_to_dataset(df, shuffle=True, repeat=False):
    file_paths = [resolve_path(x) for x in df[IMAGE_COLUMN]]
    labels = df["label_id"].tolist()
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2048)
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    if repeat:
        ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# --- 2. EĞİTİM AKIŞI ---
if __name__ == "__main__":
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

    print(f"Hedef Resim Boyutu: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("Son rötuşlar için veriler hazırlanıyor...")

    df = load_dataframe()
    # Splitler aynı olmalı
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=RANDOM_STATE)

    train_ds = dataframe_to_dataset(train_df, shuffle=True, repeat=True)
    val_ds = dataframe_to_dataset(val_df, shuffle=False)

    steps_per_epoch = len(train_df) // BATCH_SIZE
    validation_steps = len(val_df) // BATCH_SIZE

    # --- ÖNCEKİ MODELİ YÜKLE ---
    # Sıfırdan başlamıyoruz! 'Manual' modeli eğitip zekileştirdik, şimdi onu dengeleyeceğiz.
    prev_model_path = "best_model_manual.keras"
    if not os.path.exists(prev_model_path):
        raise FileNotFoundError("Önceki model bulunamadı! Lütfen önce 'best_model_manual.keras' dosyasını oluştur.")

    print(f">>> Eğitilmiş Model Yükleniyor: {prev_model_path}")
    model = tf.keras.models.load_model(prev_model_path)

    # --- YENİ "YUMUŞAK" AĞIRLIKLAR ---
    # N sınıfını serbest bırakıyoruz (0.5 -> 1.0)
    # D ve O sınıfını hala önemsiyoruz ama biraz gevşetiyoruz (4.0 -> 2.0)
    class_weights_final = {
        0: 1.5,  # A
        1: 1.0,  # C
        2: 2.0,  # D (Dengeli koruma)
        3: 1.5,  # G
        4: 3.0,  # H (Hala yardıma muhtaç)
        5: 1.0,  # M
        6: 1.0,  # N (Normale döndü!)
        7: 2.5,  # O
    }
    print(">>> Final Dengeleme Ağırlıkları:", class_weights_final)

    # --- DERLEME VE EĞİTİM ---
    # Learning Rate'i çok düşük tutuyoruz (1e-5). Modelin kafasını karıştırmadan ince ayar yapıyoruz.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint = ModelCheckpoint("best_model_final.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
    # Early stop biraz daha kısa, amaç sadece dengeyi bulmak
    early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
    csv_logger = CSVLogger(f"training_log_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    print("\n>>> Final Stage: Dengeleme ve Yüksek Skor Koşusu Başlıyor...")
    model.fit(
        train_ds,
        epochs=15,  # Kısa ve etkili bir tur
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stop, reduce_lr, csv_logger],
        class_weight=class_weights_final
    )

    print("\n✅ TÜM EĞİTİM TAMAMLANDI!")
    print("Final Model: best_model_final.keras")
    print("Şimdi evaluate.py dosyasında model ismini 'best_model_final.keras' yapıp test edebilirsin.")