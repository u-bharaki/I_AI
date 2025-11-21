# train.py

import tensorflow as tf
from dataset import load_dataframe, dataframe_to_dataset
from model import build_model
from config import EPOCHS, RANDOM_STATE, LEARNING_RATE, NUM_CLASSES
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

print("Veri çerçevesi yükleniyor...")
df = load_dataframe()

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=RANDOM_STATE, stratify=df['label_id'])

val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df['label_id'])

print(f"Train Adet: {len(train_df)}")
print(f"Validation Adet: {len(val_df)}")
print(f"Test Adet: {len(test_df)}")

print("Dataset objeleri oluşturuluyor...")
train_ds = dataframe_to_dataset(train_df, shuffle=True, repeat=True)
val_ds = dataframe_to_dataset(val_df, shuffle=False)
test_ds = dataframe_to_dataset(test_df, shuffle=False)

print("Model oluşturuluyor ve derleniyor...")
model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Eğitimi Başlıyor...")
from config import BATCH_SIZE

steps_per_epoch = len(train_df) // BATCH_SIZE
validation_steps = len(val_df) // BATCH_SIZE

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('best_efficientnet_goz_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=callbacks # CALLBACKS EKLE
)

print("\nModel Test Kümesinde Değerlendiriliyor...")
loss, acc = model.evaluate(test_ds)
print(f"Test Kaybı (Loss): {loss:.4f}")
print(f"Test Doğruluğu (Accuracy): {acc:.4f}")
