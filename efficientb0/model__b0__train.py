import tensorflow as tf
from dataset import load_dataframe, dataframe_to_dataset
from model import build_model
from config import *
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from datetime import datetime


print("CSV yükleniyor...")
df = load_dataframe()

# Train / Val / Test Split
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_STATE
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=RANDOM_STATE
)

# Datasets
train_ds = dataframe_to_dataset(train_df, shuffle=True, repeat=True)
val_ds = dataframe_to_dataset(val_df, shuffle=False)
test_ds = dataframe_to_dataset(test_df, shuffle=False)

steps_per_epoch = len(train_df) // BATCH_SIZE
validation_steps = len(val_df) // BATCH_SIZE

# Class Weights
labels = train_df["label_id"].values
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = {i: weights[i] for i in range(len(weights))}

# Loss: Focal kaldırıldı — çok büyük fark yaratır!
loss_fn = "categorical_crossentropy"

# Callbacks
checkpoint = ModelCheckpoint(
    "best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max"
)

early_stop = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7
)

logfile = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_logger = CSVLogger(logfile)


###########################################################
# Stage 1 — Frozen Training
###########################################################
print("\n>>> Stage 1: Frozen Training")
model, base_model = build_model(trainable=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FROZEN),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    epochs=EPOCHS_FROZEN,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stop, reduce_lr, csv_logger],
    class_weight=class_weights,
    verbose=2
)


###########################################################
# Stage 2 — Fine-Tuning last 40 layers
###########################################################
print("\n>>> Stage 2: Fine Tuning")

# Freeze all except last 40
for layer in base_model.layers[:-40]:
    layer.trainable = False
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    epochs=EPOCHS_FROZEN + EPOCHS_FINETUNE,
    initial_epoch=EPOCHS_FROZEN,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stop, reduce_lr, csv_logger],
    class_weight=class_weights,
    verbose=2
)


###########################################################
# Stage 3 — Warmup Low LR
###########################################################
print("\n>>> Stage 3: Warmup")

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_WARMUP),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    epochs=EPOCHS_FROZEN + EPOCHS_FINETUNE + EPOCHS_WARMUP,
    initial_epoch=EPOCHS_FROZEN + EPOCHS_FINETUNE,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=[checkpoint, csv_logger],
    class_weight=class_weights,
    verbose=2
)

print("\nEğitim tamamlandı!")
print("En iyi model dosyası: best_model.keras")
print("Log dosyası:", logfile)
