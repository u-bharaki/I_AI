import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from config import *
from dataset import load_dataframe, balanced_dataset, build_eval_dataset
from model import build_model
from sklearn.model_selection import train_test_split

#For system analysis
import psutil
import subprocess
import threading
import time
import math

# ============================================
#       MODEL YÜKLEME (CHECKPOINT RESUME)
# ============================================

checkpoint_path = "best_model.keras"

if os.path.exists(checkpoint_path):
    print("🔄 Kayıtlı model bulundu, yükleniyor...")
    model = tf.keras.models.load_model(checkpoint_path)
    start_from_checkpoint = True
else:
    print("➡ Yeni model oluşturuluyor...")
    model = build_model(trainable=False)
    start_from_checkpoint = False



# ============================================
#               System Analysis
# ============================================

system_log = {
    "time": [],
    "cpu": [],
    "ram": [],
    "gpu": []
}

def get_gpu_usage():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except:
        return 0.0  # GPU yoksa 0 yaz

def system_monitor():
    start = time.time()
    while monitoring_active:
        elapsed = time.time() - start
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        gpu = get_gpu_usage()

        system_log["time"].append(elapsed)
        system_log["cpu"].append(cpu)
        system_log["ram"].append(ram)
        system_log["gpu"].append(gpu)

        time.sleep(1)  # her saniye ölçüm

def save_system_usage_graphs():
    # CPU
    plt.figure(figsize=(10,5))
    plt.plot(system_log["time"], system_log["cpu"], label="CPU %")
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.legend()
    plt.savefig("system_use_cpu.png")
    plt.close()

    # RAM
    plt.figure(figsize=(10,5))
    plt.plot(system_log["time"], system_log["ram"], label="RAM %", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("RAM Usage (%)")
    plt.title("RAM Usage Over Time")
    plt.legend()
    plt.savefig("system_use_ram.png")
    plt.close()

    # GPU
    plt.figure(figsize=(10,5))
    plt.plot(system_log["time"], system_log["gpu"], label="GPU %", color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("GPU Usage (%)")
    plt.title("GPU Usage Over Time")
    plt.legend()
    plt.savefig("system_use_gpu.png")
    plt.close()

    print("\n📊 System usage graphs saved:")
    print(" - system_use_cpu.png")
    print(" - system_use_ram.png")
    print(" - system_use_gpu.png\n")

# ============================================
#               DEVICE SEÇİMİ
# ============================================
choice = input("GPU ile mi (G) yoksa CPU ile mi (C) eğitmek istersiniz?: ").strip().lower()

if choice == "c":
    print("CPU seçildi. GPU devre dışı bırakılıyor…")
    tf.config.set_visible_devices([], 'GPU')
else:
    print("GPU etkin. CUDA kullanılacak…")

monitoring_active = True
monitor_thread = threading.Thread(target=system_monitor, daemon=True)
monitor_thread.start()

print("📡 System monitor started...")


# ============================================
#               DATASET AYARLARI
# ============================================
df = load_dataframe()

train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=RANDOM_STATE, stratify=df["label_id"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df["label_id"]
)

# ------------- TRAIN PIPELINE ---------------
train_ds = balanced_dataset(train_df)
train_ds = train_ds.repeat()
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# ------------- VAL & TEST PIPELINE ----------
val_ds = build_eval_dataset(val_df)
test_ds = build_eval_dataset(test_df)

steps_per_epoch = 200
validation_steps = None

# ============================================
#               MODEL
# ============================================

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
]

initial_epoch_stage1 = 0
if start_from_checkpoint:
    # Model.training history checkpoint kaydetmez, epoch'ı optimizer’dan alıyoruz
    try:
        initial_epoch_stage1 = int(model.optimizer.iterations.numpy() / steps_per_epoch)
    except:
        initial_epoch_stage1 = 0

print(f"➡ Stage-1 eğitim {initial_epoch_stage1}. epoch'tan devam edecek.")



# ============================================
#          STAGE 1: Frozen Training
# ============================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FROZEN),
    loss=loss,
    metrics=["accuracy"]
)

hist1 = model.fit(
    train_ds,
    epochs=EPOCHS_FROZEN,
    initial_epoch=initial_epoch_stage1,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# ============================================
#          STAGE 2: Fine-Tuning
# ============================================
model.trainable = True  
total_layers = len(model.layers)
print(f"Toplam katman sayısı: {total_layers}")

# Fine-tune yapılacak sınır (Toplam - FineTuneMiktarı)
fine_tune_at = total_layers - LAYERS_TO_FINE_TUNE

# Sınıra kadar olan tüm katmanları dondur
for layer in model.layers[:fine_tune_at]:
    layer.trainable = False

# Sınırdan sonrasını aç (BatchNormalization hariç)
for layer in model.layers[fine_tune_at:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

print(f"Fine-Tuning başladı! Son {LAYERS_TO_FINE_TUNE} katman eğitiliyor (BN hariç).")

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
    loss=loss,
    metrics=["accuracy"]
)

hist2 = model.fit(
    train_ds,
    epochs=EPOCHS_FROZEN + EPOCHS_FINETUNE,
    initial_epoch=EPOCHS_FROZEN,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# ============================================
#          STAGE 3: Warmup
# ============================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_WARMUP),
    loss=loss,
    metrics=["accuracy"]
)

hist3 = model.fit(
    train_ds,
    epochs=EPOCHS_FROZEN + EPOCHS_FINETUNE + EPOCHS_WARMUP,
    initial_epoch=EPOCHS_FROZEN + EPOCHS_FINETUNE,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=[callbacks[0]]
)

print("Eğitim tamamlandı!")

# ============================================
#          TEST DEĞERLENDİRME
# ============================================
print("Test verisi üzerinde tahmin yapılıyor...")
y_true = []
y_pred = []

for x, y in test_ds:
    preds = model.predict(x)
    y_true.extend(np.argmax(y.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ============================================
#          METRİKLER
# ============================================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro')
rec = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

try:
    auc = roc_auc_score(tf.one_hot(y_true, NUM_CLASSES), tf.one_hot(y_pred, NUM_CLASSES), multi_class='ovr')
except:
    auc = 0.0

print("\n========= TEST METRİKLERİ =========")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro): {rec:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print(f"ROC-AUC (macro): {auc:.4f}")

# ============================================
#      CONFUSION MATRIX GRAFİĞİ
# ============================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================
#      EĞİTİM GRAFİKLERİ (ACC / LOSS)
# ============================================
history = {}
for h in [hist1, hist2, hist3]:
    for k, v in h.history.items():
        history.setdefault(k, []).extend(v)

plt.figure(figsize=(12, 5))
plt.plot(history["accuracy"], label="train acc")
plt.plot(history["val_accuracy"], label="val acc")
plt.legend(); plt.title("Accuracy")
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history["loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.legend(); plt.title("Loss")
plt.show()

# ============================================
#      TAHMİN TABLOSU
# ============================================
labels = ["other", "normal", "amd", "hyper", "myopia", "cataract", "glaucoma", "diabetes"]

report = pd.DataFrame(columns=["Class", "Actual", "Predicted", "Correct"])

for i, name in enumerate(labels):
    actual = (y_true == i).sum()
    predicted = (y_pred == i).sum()
    correct = ((y_true == i) & (y_pred == i)).sum()
    report.loc[i] = [name, actual, predicted, correct]

print("\n========= TAHMİN RAPORU =========")
print(report)

monitoring_active = False
monitor_thread.join()

save_system_usage_graphs()

