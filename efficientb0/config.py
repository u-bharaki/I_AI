# config.py

IMAGE_SIZE = 512
NUM_CLASSES = 8
CHANNELS = 3

EPOCHS_FROZEN = 10         # 1. aşama
EPOCHS_FINETUNE = 20       # 2. aşama
EPOCHS_WARMUP = 5          # 3. aşama (en düşük LR)

BATCH_SIZE = 32
LEARNING_RATE_FROZEN = 1e-3
LEARNING_RATE_FINETUNE = 1e-4
LEARNING_RATE_WARMUP = 1e-5

RANDOM_STATE = 42

CSV_FILE = r"C:\Users\duygu\IdeaProjects\yap470projesel\cleaned_file_final.csv"
DATA_ROOT = r"C:\Users\duygu\IdeaProjects\yap470projesel\preprocessed_images"

LABEL_COLUMN = "labels"
IMAGE_COLUMN = "filepath"