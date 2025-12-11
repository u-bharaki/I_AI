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

# Kendi dosyalarımız (config.py ve model.py aynı klasörde olmalı)
from config import *
from model import build_model

# --- 1. VERİ YÜKLEME VE İŞLEME ---
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
    # 224x224 boyutuna resize (EfficientNetB0 standardı)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label

# --- GÜÇLÜ AUGMENTATION (Overfitting Önleyici) ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2), # %20 döndürme
    tf.keras.layers.RandomZoom(0.2),     # %20 zoom
    tf.keras.layers.RandomContrast(0.1), # Kontrast
])

def augment_image(img, lbl):
    # Augmentation sadece eğitimde (training=True) çalışır
    img = data_augmentation(img, training=True)
    return img, lbl

def dataframe_to_dataset(df, shuffle=True, repeat=False):
    file_paths = [resolve_path(x) for x in df[IMAGE_COLUMN]]
    labels = df["label_id"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache (RAM yetmezse bu satırı sil)
    ds = ds.cache()

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
    # GPU Bellek Ayarı
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

    print(f"Hedef Resim Boyutu: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("CSV ve Veriler Hazırlanıyor...")

    df = load_dataframe()

    # Train / Val / Test Split
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=RANDOM_STATE
    )

    print(f"Eğitim Verisi: {len(train_df)}, Doğrulama: {len(val_df)}")

    train_ds = dataframe_to_dataset(train_df, shuffle=True, repeat=True)
    val_ds = dataframe_to_dataset(val_df, shuffle=False)

    steps_per_epoch = len(train_df) // BATCH_SIZE
    validation_steps = len(val_df) // BATCH_SIZE

    # -------------------------------------------------------------------------
    # KRİTİK ADIM: MANUEL SINIF AĞIRLIKLARI
    # Raporuna göre D, O ve H sınıfları öğrenilemiyor. N sınıfı çok baskın.
    # Sıralama: ['A', 'C', 'D', 'G', 'H', 'M', 'N', 'O']
    # -------------------------------------------------------------------------
    class_weights = {
        0: 2.0,  # A - Biraz destek
        1: 1.0,  # C - İyi durumda
        2: 4.0,  # D - KRİTİK (Çok hata var, ağırlık artırıldı)
        3: 2.0,  # G - Biraz destek
        4: 5.0,  # H - KRİTİK (Çok az veri var, en yüksek öncelik)
        5: 1.0,  # M - İyi durumda
        6: 0.5,  # N - ÇOK BASKIN (Modelin buraya kaçmasını engellemek için kısıldı)
        7: 4.0,  # O - KRİTİK (Model hiç öğrenememiş)
    }
    print(">>> Manuel Sınıf Ağırlıkları Uygulandı:", class_weights)

    # Callbacks
    checkpoint = ModelCheckpoint(
        "best_model_manual.keras", # Dosya adı değişti, karışmasın
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=12, # Sabır biraz daha artırıldı
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )

    logfile = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_logger = CSVLogger(logfile)

    # ---------------------------------------------------------
    # STAGE 1: FROZEN TRAINING
    # ---------------------------------------------------------
    print("\n>>> Stage 1: Frozen Training")
    model, base_model = build_model(trainable=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FROZEN),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        epochs=EPOCHS_FROZEN,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stop, reduce_lr, csv_logger],
        class_weight=class_weights # Manuel ağırlıklar burada!
    )

    # ---------------------------------------------------------
    # STAGE 2: FULL FINE-TUNING
    # ---------------------------------------------------------
    print("\n>>> Stage 2: Full Fine-Tuning (Derinlemesine Öğrenme)")

    base_model.trainable = True

    # BatchNorm katmanlarını dondur (İstatistikleri korumak için)
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_UNFREEZE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    total_epochs = EPOCHS_FROZEN + EPOCHS_UNFREEZE

    model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=EPOCHS_FROZEN, # Kaldığı yerden devam
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stop, reduce_lr, csv_logger],
        class_weight=class_weights # Manuel ağırlıklar burada!
    )

    print("\nEğitim Tamamlandı! En iyi model 'best_model_manual.keras' olarak kaydedildi.")