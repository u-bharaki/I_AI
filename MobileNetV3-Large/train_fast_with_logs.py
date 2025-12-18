import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from datetime import datetime

# Mevcut paketlerin
from config import *
from model import build_model
from dataset import load_dataframe, dataframe_to_dataset

def train():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_dataframe()
    # EfficientNetB0 ile birebir aynı split oranları
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=RANDOM_STATE)

    train_ds = dataframe_to_dataset(train_df, shuffle=True, repeat=True)
    val_ds = dataframe_to_dataset(val_df, shuffle=False)

    # --- B0 DOSYASINDAN ALINAN OPTİMİZE AĞIRLIKLAR ---
    # Sıralama: ['A', 'C', 'D', 'G', 'H', 'M', 'N', 'O']
    class_weights = {
        0: 2.0,  # A - Biraz destek
        1: 1.0,  # C - Stabil
        2: 4.0,  # D - Kritik (En çok N ile karışan sınıf)
        3: 2.0,  # G - Destek
        4: 5.0,  # H - Kritik (En az veri)
        5: 1.0,  # M - Stabil
        6: 0.5,  # N - Baskın (Modelin buraya kaçmasını engellemek için düşük)
        7: 4.0   # O - Kritik (Modelin öğrenmekte zorlandığı)
    }
    print(">>> B0 Stratejisi ile Sınıf Ağırlıkları Uygulandı:", class_weights)

    model, base_model = build_model(trainable=False)

    # Label Smoothing'i B0'daki gibi dengeli tutuyoruz (0.1)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    callbacks = [
        ModelCheckpoint("models/best_mobilenet_b0_weights.keras", monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_lr=1e-7),
        CSVLogger(f"logs/b0_strategy_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    ]

    steps_per_epoch = len(train_df) // BATCH_SIZE
    validation_steps = len(val_df) // BATCH_SIZE

    # STAGE 1: Isınma
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn, metrics=["accuracy"])
    model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_data=val_ds,
              validation_steps=validation_steps, epochs=5, callbacks=callbacks, class_weight=class_weights)

    # STAGE 2: Full Fine-Tuning
    base_model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss_fn, metrics=["accuracy"])
    model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_data=val_ds,
              validation_steps=validation_steps, epochs=35, callbacks=callbacks, class_weight=class_weights)

if __name__ == "__main__":
    train()