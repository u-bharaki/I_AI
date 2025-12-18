import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from config import *
from data_loader import load_dataframe, dataframe_to_dataset
from model import build_resnet50_model
from datetime import datetime
from terminal_logger import start_logging, stop_logging


def main():

    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

    logger = start_logging(log_dir="logs")

    try:
        print("--- Veriler Hazırlanıyor ---")
        df = load_dataframe()

        print(f"Categorical Focal Loss Aktif Edildi")

        train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_STATE)
        val_df, _ = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=RANDOM_STATE)

        train_ds = dataframe_to_dataset(train_df, shuffle=True, repeat=True, augment=True)
        val_ds = dataframe_to_dataset(val_df, shuffle=False, repeat=False, augment=False)

        steps_per_epoch = len(train_df) // BATCH_SIZE
        validation_steps = len(val_df) // BATCH_SIZE

        print("\n--- AŞAMA 1: WARMUP (ResNet Donduruldu) ---")
        model = build_resnet50_model(trainable=False)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR_WARMUP),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy",
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.Precision(name="precision")]
        )
        """
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df["label_id"]),
            y=train_df["label_id"]
        )
        """
        class_weights = {
            0: 1.5,  # amd
            1: 1.2,  # cataract
            2: 1.0,  # diabetes
            3: 1.3,  # glaucoma
            4: 2.0,  # hypertension
            5: 1.2,  # myopia
            6: 0.6,  # normal
            7: 1.0  # other
        }
        class_weight_dict = dict(enumerate(class_weights))

        model.fit(
            train_ds,
            epochs=EPOCHS_WARMUP,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            class_weight=class_weight_dict
        )

        print("\n--- AŞAMA 2: FINE-TUNING (Tüm Model Eğitiliyor) ---")

        for layer in model.layers[-30:]:
            layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy",
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.Precision(name="precision")]
        )

        callbacks = [
            ModelCheckpoint("best_resnet50_model.keras", monitor="val_accuracy", save_best_only=True, mode="max",
                            verbose=1),
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            CSVLogger(f"resnet_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        ]

        model.fit(
            train_ds,
            epochs=EPOCHS_FINETUNE,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            class_weight=class_weight_dict,
            callbacks=callbacks
        )

        print("Eğitim Tamamlandı.")
    except Exception as e:
        print(f"HATA: {e}")
        raise

    finally:
        stop_logging(logger)

if __name__ == "__main__":
    main()