import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import cv2
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
import pickle

# Constants

validation_steps = None
steps_per_epoch = 200

# ============================================
#       CUSTOM CALLBACKS (F1 & LOGGING)
# ============================================

class TrainingStateLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename="training_state.pkl"):
        super().__init__()
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        # Kaydedilecek veriler
        state = {
            "epoch": epoch + 1, # Bir sonraki başlangıç epoch'u
            "history": self.model.history.history if hasattr(self.model, 'history') else {}
        }
        with open(self.filename, "wb") as f:
            pickle.dump(state, f)

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Validation seti üzerinde tahmin al
        y_true = []
        y_pred = []
        
        # Validation veri seti batch'ler halinde gelir
        for x, y in self.val_ds:
            preds = self.model.predict(x, verbose=0)
            y_true.extend(np.argmax(y.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
            
        # Metrikleri hesapla (Macro average)
        _f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        _prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        _rec = recall_score(y_true, y_pred, average='macro', zero_division=0)

        # Loglara ekle 
        logs['val_f1'] = _f1
        logs['val_precision'] = _prec
        logs['val_recall'] = _rec
        
        print(f" — val_f1: {_f1:.4f} — val_prec: {_prec:.4f} — val_rec: {_rec:.4f}")

# ============================================
#       GRAD-CAM FONKSİYONLARI
# ============================================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_activation", pred_index=None):
    # 1. Modeli, son conv katmanını ve çıktı katmanını verecek şekilde ayarla
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Gradyanları kaydet
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. Gradyan hesapla (Çıktının, son conv katmanına göre türevi)
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Global Average Pooling benzeri işlem (her filtre için ortalama gradyan)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Isı haritasını oluştur
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalize et (0 ile 1 arasına çek)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    # Heatmap'i 0-255 arasına çek
    heatmap = np.uint8(255 * heatmap)

    # Renk haritası uygula (JET colormap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Görüntü boyutuna getir
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    # Orijinal resmi 0-255 ve uint8 formatına çevir.
    if img.max() <= 1.0:
        img = np.uint8(255 * img)

    # Heatmap ile orijinal resmi birleştir
    superimposed_img = jet * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img

# ============================================
#       MODEL LOADING
# ============================================

last_ckpt_path = "last_checkpoint.keras"
state_path = "training_state.pkl"
best_ckpt_path = "best_model.keras"

initial_global_epoch = 0 # Toplamda kaçıncı epochta olduğumuz

if os.path.exists(last_ckpt_path):
    print(f"Son kayıt noktası bulundu: {last_ckpt_path}. Yükleniyor...")
    model = tf.keras.models.load_model(last_ckpt_path)
    
    # State dosyasından epoch bilgisini çek
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
            initial_global_epoch = state.get("epoch", 0)
            print(f"➡ Kaldığı epoch: {initial_global_epoch}")
    else:
        print("State dosyası bulunamadı, epoch tahmini optimizer'dan yapılacak.")
        try:
            initial_global_epoch = int(model.optimizer.iterations.numpy() / steps_per_epoch)
        except:
            initial_global_epoch = 0
            
    start_from_checkpoint = True
    
elif os.path.exists(best_ckpt_path):
    print("Last checkpoint yok ama Best model bulundu. Best modelden devam ediliyor (Epoch bilgisi kayıp olabilir).")
    model = tf.keras.models.load_model(best_ckpt_path)
    start_from_checkpoint = True
else:
    print("Yeni model oluşturuluyor...")
    model = build_model(trainable=False)
    start_from_checkpoint = False
    initial_global_epoch = 0


# ============================================
#               System Analysis
# ============================================

system_log = {
    "time": [],
    "cpu": [],
    "ram": [],
    "gpu": []
}

monitoring_active = True

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

    print("\nSystem usage graphs saved:")
    print(" - system_use_cpu.png")
    print(" - system_use_ram.png")
    print(" - system_use_gpu.png\n")

# ============================================
#               DEVICE SEÇİMİ
# ============================================
choice = input("GPU ile mi (G) yoksa CPU ile mi (C) eğitmek istersiniz?: ").strip().lower()

hist1, hist2, hist3 = None, None, None

if choice == "c":
    print("CPU seçildi. GPU devre dışı bırakılıyor…")
    tf.config.set_visible_devices([], 'GPU')
else:
    print("GPU etkin. CUDA kullanılacak…")

monitor_thread = threading.Thread(target=system_monitor, daemon=True)
monitor_thread.start()

print("System monitor started...")


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

# ============================================
#               MODEL & CALLBACKS
# ============================================

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras", 
    monitor="val_accuracy", 
    save_best_only=True, 
    verbose=1
)

# Her epoch sonunda üzerine yazar.
checkpoint_last = tf.keras.callbacks.ModelCheckpoint(
    "last_checkpoint.keras", 
    save_best_only=False,
    save_weights_only=False,
    verbose=0
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="./logs",
    histogram_freq=1
)

metrics_callback = MetricsCallback(val_ds)

callbacks = [
    checkpoint_best, 
    checkpoint_last,
    metrics_callback,
    tensorboard_callback,
    TrainingStateLogger(),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
]

# ============================================
#          STAGE 1: Frozen Training
# ============================================

stage1_limit = EPOCHS_FROZEN

if initial_global_epoch < stage1_limit:
    print(f"--- Stage 1 Başlıyor (Frozen) ---")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FROZEN),
        loss=loss,
        metrics=["accuracy"]
    )

    hist1 = model.fit(
        train_ds,
        epochs=EPOCHS_FROZEN,
        initial_epoch=initial_global_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

else:
    print("Stage 1 zaten tamamlanmış, atlanıyor.")

# ============================================
#          STAGE 2: Fine-Tuning
# ============================================

stage2_limit = EPOCHS_FROZEN + EPOCHS_FINETUNE

if initial_global_epoch < stage2_limit:
    print(f"--- Stage 2 Başlıyor (Fine-Tune) ---")

    model.trainable = True  
    total_layers = len(model.layers)
    print(f"Toplam katman sayısı: {total_layers}")

    # Fine-tune yapılacak sınır
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
        initial_epoch=initial_global_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

else:
    print("Stage 2 zaten tamamlanmış, atlanıyor.")

# ============================================
#          STAGE 3: Warmup
# ============================================

stage3_limit = EPOCHS_FROZEN + EPOCHS_FINETUNE + EPOCHS_WARMUP

if initial_global_epoch < stage3_limit:
    print(f"--- Stage 3 Başlıyor (Warmup) ---")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_WARMUP),
        loss=loss,
        metrics=["accuracy"]
    )

    hist3 = model.fit(
        train_ds,
        epochs=stage3_limit,
        initial_epoch=initial_global_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

else:
    print("Stage 3 (Warmup) zaten tamamlanmış, atlanıyor.")

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

executed_hists = [h for h in [hist1, hist2, hist3] if h is not None]

for h in executed_hists:
    for k, v in h.history.items():
        history.setdefault(k, []).extend(v)

if history:
    plt.figure(figsize=(12, 5))
    plt.plot(history["accuracy"], label="train acc")

    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="val acc")
    plt.legend(); plt.title("Accuracy")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(history["loss"], label="train loss")

    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val loss")
    plt.legend(); plt.title("Loss")
    plt.show()

else:
    print("Grafik çizilecek yeni eğitim verisi yok.")

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

print("\n🔍 Grad-CAM Görselleri Oluşturuluyor (Rastgele 3 örnek)...")

# Test veri setinden bir batch (32 resim) çek
try:
    for images, labels in test_ds.take(1):
        # İlk 3 resim üzerinde dene
        for i in range(3):
            img_tensor = images[i] # (512, 512, 3) - Preprocessed
            label_idx = np.argmax(labels[i]) # Gerçek sınıf
            
            # Batch boyutu ekle: (1, 512, 512, 3)
            img_batch = tf.expand_dims(img_tensor, axis=0)
            
            # EfficientNetB3'ün son conv layer ismi genelde 'top_activation'dır.
            heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name="top_activation")
            
            # Görselleştir
            res = save_and_display_gradcam(img_tensor.numpy(), heatmap)
            
            plt.figure(figsize=(6, 6))
            plt.imshow(res)
            plt.title(f"True Class: {label_idx} (Grad-CAM)")
            plt.axis("off")
            plt.show()
            
except Exception as e:
    print(f"⚠ Grad-CAM hatası: {e}")
    print("İpucu: 'last_conv_layer_name' parametresini model.summary()'e bakarak güncellemen gerekebilir.")

monitoring_active = False
monitor_thread.join()

save_system_usage_graphs()

