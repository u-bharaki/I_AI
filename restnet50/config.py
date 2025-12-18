import os

IMAGE_SIZE = 224
CHANNELS = 3
NUM_CLASSES = 8

BATCH_SIZE = 32
RANDOM_STATE = 42

EPOCHS_WARMUP = 10
EPOCHS_FINETUNE = 50

LR_WARMUP = 1e-3
LR_FINETUNE = 5e-5

CSV_FILE = r"../cleaned_file_final.csv"
DATA_ROOT = r"../preprocessed_images"

LABEL_COLUMN = "Diagnosis"
IMAGE_COLUMN = "filepath"

CLASS_NAMES = [
    "amd",
    "cataract",
    "diabetes",
    "glaucoma",
    "hypertension",
    "myopia",
    "normal",
    "other"
]

alpha = [
        2.5,  # amd
        1.2,  # cataract
        0.8,  # diabetes
        2.0,  # glaucoma
        3.0,  # hypertension
        0.5,  # myopia
        0.3,  # normal
        2.5  # other
    ]