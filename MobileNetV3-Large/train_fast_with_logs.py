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

    train_ds = dataframe_to_dataset(train_df, shuffle=True, repeat=True, augment=True)
    val_ds = dataframe_to_dataset(val_df, shuffle=False)

    class_weights = {0: 2.0, 1: 1.0, 2: 2.0, 3: 2.0, 4: 4.0, 5: 1.0, 6: 1.0, 7: 2.5}

    model, base_model = build_model(trainable=True)

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint("models/fast_mobile_model.keras", monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7),
        CSVLogger(f"logs/fast_train_log_{datetime.now().strftime('%m%d_%H%M')}.csv")
    ]

    print("\n>>> ...")
    model.fit(
        train_ds,
        steps_per_epoch=len(train_df)//BATCH_SIZE,
        validation_data=val_ds,
        validation_steps=len(val_df)//BATCH_SIZE,
        epochs=20,
        callbacks=callbacks,
        class_weight=class_weights
    )

if __name__ == "__main__":
    train()