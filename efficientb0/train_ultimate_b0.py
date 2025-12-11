import pandas as pd
import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import regularizers # L2 için gerekli
from datetime import datetime

# --- AYARLAR ---
# B0 için ideal boyutlara geri dönüyoruz ama teknikleri değiştiriyoruz
IMAGE_SIZE = 224
BATCH_SIZE = 32 # B0 olduğu için 32 yapabiliriz (Hızlanır)
CHANNELS = 3
NUM_CLASSES = 8
RANDOM_STATE = 42

CSV_FILE = r"..\cleaned_file_final.csv"
DATA_ROOT = r"..\preprocessed_images"
LABEL_COLUMN = "labels"
IMAGE_COLUMN = "filepath"

# --- 1. MODEL MİMARİSİ (L2 REGULARIZATION İLE GÜÇLENDİRİLMİŞ) ---
def build_model_optimized(trainable=False):
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    )
    base_model.trainable = trainable

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # KATMAN 1: Dropout %50 + BatchNormalization
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # KATMAN 2: L2 Regularization Eklendi (Ezber Bozucu)
    # kernel_regularizer=regularizers.l2(0.001) -> Ağırlıkları baskılar
    x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)

    # KATMAN 3: Dropout %40 + BN
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # ÇIKIŞ: Softmax
    out = Dense(NUM_CLASSES, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=out), base_model

# --- 2. VERİ HAZIRLIĞI ---
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

# Dengeli Augmentation (Ne çok sert ne çok yumuşak)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
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

# --- 3. EĞİTİM AKIŞI ---
if __name__ == "__main__":
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

    print(f"Model: EfficientNetB0 (OPTIMIZED) | Resim Boyutu: {IMAGE_SIZE}x{IMAGE_SIZE}")

    df = load_dataframe()
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=RANDOM_STATE)

    train_ds = dataframe_to_dataset(train_df, shuffle=True, repeat=True)
    val_ds = dataframe_to_dataset(val_df, shuffle=False)

    steps_per_epoch = len(train_df) // BATCH_SIZE
    validation_steps = len(val_df) // BATCH_SIZE

    # Manuel Ağırlıklar (Başarı getiren formül)
    class_weights = {
        0: 1.5, 1: 1.0, 2: 2.0, 3: 1.5, 4: 3.0, 5: 1.0, 6: 0.9, 7: 2.5
    }

    # Klasör oluştur
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    checkpoint = ModelCheckpoint("saved_models/best_model_b0_ultimate.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
    # Sabrı artırdık (Patience 12), çünkü L2 Regülasyonu öğrenmeyi biraz yavaşlatır ama sağlamlaştırır.
    early_stop = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1)
    csv_logger = CSVLogger(f"logs/training_log_b0_ultimate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    print("\n>>> Stage 1: Frozen Training (Yeni Katmanlar Isınıyor)")
    model, base_model = build_model_optimized(trainable=False)

    # AdamW kullanıyoruz (Weight Decay = 1e-4) -> Bu da bir çeşit L2'dir.
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds, epochs=10,
        steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps,
        callbacks=[checkpoint, early_stop, reduce_lr, csv_logger], class_weight=class_weights
    )

    print("\n>>> Stage 2: Full Fine-Tuning (Sıkı Yönetim)")
    base_model.trainable = True

    # BatchNorm'u yine donduruyoruz (İstatistikler bozulmasın)
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Fine-tuning için düşük LR ama aktif Weight Decay
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds, epochs=40, initial_epoch=10,
        steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps,
        callbacks=[checkpoint, early_stop, reduce_lr, csv_logger], class_weight=class_weights
    )

    print("\n✅ EĞİTİM BİTTİ! Model: saved_models/best_model_b0_ultimate.keras")