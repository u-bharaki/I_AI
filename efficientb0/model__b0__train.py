# model__b0__train.py

import tensorflow as tf
from dataset import load_dataframe, dataframe_to_dataset
from model import build_model
from config import *
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("CSV yükleniyor...")
df = load_dataframe()

train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=RANDOM_STATE, stratify=df["label_id"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df["label_id"]
)

train_ds = dataframe_to_dataset(train_df, shuffle=True, repeat=True)
val_ds = dataframe_to_dataset(val_df, shuffle=False)
test_ds = dataframe_to_dataset(test_df, shuffle=False)

steps_per_epoch = len(train_df) // BATCH_SIZE
validation_steps = len(val_df) // BATCH_SIZE

# Class weights
y_train = train_df["label_id"].values
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: weights[i] for i in range(len(weights))}

loss_focal = CategoricalFocalCrossentropy(gamma=2)


checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6
)


# ------------------------------
# AŞAMA 1 — Dondurulmuş Eğitim
# ------------------------------
model = build_model(trainable=False)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(LEARNING_RATE_FROZEN),
    loss=loss_focal,
    metrics=["accuracy"]
)

print("\n### Stage 1: Frozen Training ###\n")
model.fit(
    train_ds,
    epochs=EPOCHS_FROZEN,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weights
)


# ------------------------------
# AŞAMA 2 — Fine-Tuning (Son 40 Layer Açılır)
# ------------------------------
print("\n### Stage 2: Fine-Tuning ###\n")

base_model = model.layers[0]
for layer in base_model.layers[:-40]:
    layer.trainable = False
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(LEARNING_RATE_FINETUNE),
    loss=loss_focal,
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    epochs=EPOCHS_FROZEN + EPOCHS_FINETUNE,
    initial_epoch=EPOCHS_FROZEN,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weights
)


# ------------------------------
# AŞAMA 3 — Warmup (Very Low LR)
# ------------------------------
print("\n### Stage 3: Warmup Training ###\n")

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(LEARNING_RATE_WARMUP),
    loss=loss_focal,
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    epochs=EPOCHS_FROZEN + EPOCHS_FINETUNE + EPOCHS_WARMUP,
    initial_epoch=EPOCHS_FROZEN + EPOCHS_FINETUNE,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=[checkpoint],
    class_weight=class_weights
)

print("Eğitim tamamlandı! En iyi model: best_model.keras")
