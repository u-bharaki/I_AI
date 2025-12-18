import os

IMAGE_SIZE = 224
CHANNELS = 3
NUM_CLASSES = 8
BATCH_SIZE = 32
RANDOM_STATE = 42

EPOCHS_WARMUP = 5
EPOCHS_FINETUNE = 40

LR_WARMUP = 3e-4
LR_FINETUNE = 1e-5

CSV_FILE = r"../cleaned_file_final.csv"
DATA_ROOT = r"../preprocessed_images"

LABEL_COLUMN = "Diagnosis"
IMAGE_COLUMN = "filepath"

CLASS_NAMES = [
    "amd", "cataract", "diabetes", "glaucoma",
    "hypertension", "myopia", "normal", "other"
]

# Focal Loss alpha (class imbalance için)
alpha = [2.0, 1.2, 1.0, 2.0, 2.5, 0.8, 0.5, 2.0]
