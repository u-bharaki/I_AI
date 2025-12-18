import os

# --- AYARLAR ---
IMAGE_SIZE = 224
CHANNELS = 3
NUM_CLASSES = 8
BATCH_SIZE = 32 # Daha kararlı öğrenme için 32 idealdir
RANDOM_STATE = 42

EPOCHS_FROZEN = 5
EPOCHS_FINE_TUNE = 35 # Modelin öğrenmesi için süreyi uzattık

# Yollar
CSV_FILE = r"..\cleaned_file_final.csv"
DATA_ROOT = r"..\preprocessed_images"
LABEL_COLUMN = "labels"
IMAGE_COLUMN = "filepath"