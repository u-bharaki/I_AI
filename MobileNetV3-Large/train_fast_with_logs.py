import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from datetime import datetime

from config import *
from model import build_model
from dataset import load_dataframe, dataframe_to_dataset

def train():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_dataframe()
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=RANDOM_STATE)

    # Eğitim setine 'augment=True' ekledik
    train_ds = dataframe_to_dataset(train_df, shuffle=True, repeat=True, augment=True)
    val_ds = dataframe_to_dataset(val_df, shuffle=False)

    # Gönderdiğin Agresif Sınıf Ağırlıkları
    class_weights = {
        0: 3.0, 1: 1.0, 2: 3.0, 3: 3.0,
        4: 10.0, # H sınıfı için hayati destek
        5: 1.0, 6: 0.3, 7: 5.0
    }

    model, base_model = build_model(trainable=False)

    # Gönderdiğin %15 Label Smoothing (Focal Loss etkisi yaratır)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.15)

    callbacks = [
        ModelCheckpoint("models/final_best_model.keras", monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7),
        CSVLogger(f"logs/final_training_log_{datetime.now().strftime('%m%d_%H%M')}.csv")
    ]

    print("\n>>> Stage 1: Isınma (Dondurulmuş Katmanlar)...")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn, metrics=["accuracy"])
    model.fit(train_ds, steps_per_epoch=len(train_df)//BATCH_SIZE, validation_data=val_ds,
              validation_steps=len(val_df)//BATCH_SIZE, epochs=7,
              callbacks=callbacks, class_weight=class_weights)

    print("\n>>> Stage 2: Fine-Tuning (Tüm Katmanlar Açık)...")
    base_model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss_fn, metrics=["accuracy"])
    model.fit(train_ds, steps_per_epoch=len(train_df)//BATCH_SIZE, validation_data=val_ds,
              validation_steps=len(val_df)//BATCH_SIZE, epochs=50,
              callbacks=callbacks, class_weight=class_weights)

if __name__ == "__main__":
    train()