IMAGE_SIZE = 512
CHANNELS = 3
NUM_CLASSES = 8

BATCH_SIZE = 16
RANDOM_STATE = 42

# Learning rates
LR_FROZEN = 1e-3
LR_FINETUNE = 1e-5
LR_WARMUP = 1e-6

# Epochs
EPOCHS_FROZEN = 10
EPOCHS_FINETUNE = 20
EPOCHS_WARMUP = 5

# Dataset
CSV_FILE = r"..\cleaned_file_final.csv"
DATA_ROOT = r"..\preprocessed_images"

LABEL_COLUMN = "labels"
IMAGE_COLUMN = "filepath"
