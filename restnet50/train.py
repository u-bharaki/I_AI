import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from config import *
from data_loader import load_dataframe, dataframe_to_dataset
from model import build_resnet50_model
from datetime import datetime

def main():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

    print("Veriler hazırlanıyor...")
    df = load_dataframe()

    # Stratified Split
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_STATE)
    val_df, _ = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=RANDOM_STATE)

    train_ds = dataframe_to_dataset(train_df, shuffle=True, repeat=True, augment=True)
    val_ds = dataframe_to_dataset(val_df, shuffle=False, repeat=False, augment=False)

    steps_per_epoch = len(train_df) // BATCH_SIZE
    validation_steps = len(val_df) // BATCH_SIZE

    print("ResNet50 Modeli oluşturuluyor...")
    # İlk etapta base model eğitilebilir olsun (Fine-tuning için True)
    # Donanım yetersizse False yapıp önce head eğitilmeli.
    model = build_resnet50_model(trainable=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Class Weights (Manuel ayarlarınız)
    class_weights = {
        0: 1.5, 1: 1.0, 2: 2.0, 3: 1.5, 4: 3.0, 5: 1.0, 6: 1.0, 7: 2.5
    }

    callbacks = [
        ModelCheckpoint("best_resnet50_model.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7, verbose=1),
        CSVLogger(f"resnet_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    ]

    print("Eğitim Başlıyor...")
    model.fit(
        train_ds,
        epochs=50,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights
    )
    print("Eğitim Tamamlandı.")

if __name__ == "__main__":
    main()