import os

IMAGE_SIZE = 224
CHANNELS = 3
NUM_CLASSES = 8

BATCH_SIZE = 32
RANDOM_STATE = 42

LR_FROZEN = 1e-3
LR_UNFREEZE = 1e-5

EPOCHS_FROZEN = 10
EPOCHS_UNFREEZE = 35

CSV_FILE = r"..\cleaned_file_final.csv"
DATA_ROOT = r"..\preprocessed_images"

LABEL_COLUMN = "labels"
IMAGE_COLUMN = "filepath"