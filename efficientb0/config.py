import os

# --- AYARLAR ---
# EfficientNetB0 için en ideal boyut 224'tür.
# 512 yapmak modelin kafasını karıştırır ve eğitimi yavaşlatır.
IMAGE_SIZE = 224
CHANNELS = 3
NUM_CLASSES = 8

# Batch Size: 32 (RAM hatası alırsan 16'ya düşür)
BATCH_SIZE = 32
RANDOM_STATE = 42

# Learning Rates
LR_FROZEN = 1e-3
LR_UNFREEZE = 1e-5  # Hassas ayar için çok düşük hız

# Epochs
EPOCHS_FROZEN = 10   # Isınma turları
EPOCHS_UNFREEZE = 35 # Asıl öğrenme süreci (Artırıldı)

# Dosya Yolları (Kendi bilgisayarına göre ise dokunma)
CSV_FILE = r"..\cleaned_file_final.csv"
DATA_ROOT = r"..\preprocessed_images"

LABEL_COLUMN = "labels"
IMAGE_COLUMN = "filepath"