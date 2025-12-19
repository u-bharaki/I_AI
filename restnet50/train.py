import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

from config import *
from data_loader import load_dataframe, dataframe_to_dataset
from model import build_resnet50_model
from loss_utils import CategoricalFocalLoss


def main():
    try:
        df = load_dataframe()

        train_df, temp_df = train_test_split(
            df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_STATE
        )
        val_df, _ = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=RANDOM_STATE
        )

        train_ds = dataframe_to_dataset(train_df, augment=True, repeat=True, use_clahe=True)
        val_ds = dataframe_to_dataset(val_df, augment=False, use_clahe=True)

        steps_per_epoch = len(train_df) // BATCH_SIZE
        val_steps = len(val_df) // BATCH_SIZE

        print("=== WARMUP ===")
        model = build_resnet50_model(trainable=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LR_WARMUP),
            loss=CategoricalFocalLoss(),
            metrics=["accuracy"]
        )

        model.fit(
            train_ds,
            epochs=EPOCHS_WARMUP,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=val_steps
        )

        print("=== FINE TUNING ===")
        for layer in model.layers[-30:]:
            layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
            loss=CategoricalFocalLoss(),
            metrics=["accuracy"]
        )

        callbacks = [
            ModelCheckpoint("best_resnet50_model.keras",
                            monitor="val_accuracy",
                            save_best_only=True,
                            mode="max"),
            EarlyStopping(patience=8, restore_best_weights=True),
            ReduceLROnPlateau(patience=4, factor=0.3),
            CSVLogger("resnet_final_log.csv")
        ]

        model.fit(
            train_ds,
            epochs=EPOCHS_FINETUNE,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=val_steps,
            callbacks=callbacks
        )

    finally:
        pass

if __name__ == "__main__":
    main()