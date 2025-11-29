# preprocessing.py

import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

from cache import load_cache, save_cache

# ---- SENİN TAM DOSYA YOLLARIN ----
CSV_PATH = r"..\cleaned_file_final.csv"
DATA_ROOT = r"..\preprocessed_images"

SCRIPT_PATH = __file__  # preprocessing.py dosyasının tam yolu


def load_and_prepare_data(img_size=32):
    """Önce cache'e bakar. Yoksa preprocessing yapar."""

    # 1) CACHE KONTROLÜ
    Xc, yc, meta = load_cache(img_size, CSV_PATH, SCRIPT_PATH)
    if Xc is not None:
        print("[INFO] Cache bulundu → preprocessing yapılmadı.")
        return Xc, yc, None

    # 2) CACHE YOKSA → NORMAL PREPROCESSING
    df = pd.read_csv(CSV_PATH)

    df.dropna(subset=['filepath', 'Diagnosis', 'Patient Age', 'Patient Sex'], inplace=True)
    df['Patient Age'] = pd.to_numeric(df['Patient Age'], errors='coerce')
    df.dropna(subset=['Patient Age'], inplace=True)

    print(f"{len(df)} satır bulundu. Özellik çıkarma başlıyor...")

    X_features = []
    y_labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Görüntüler işleniyor"):
        img_path = os.path.join(DATA_ROOT, row['filepath'])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (img_size, img_size))
        img_flat = img_resized.flatten() / 255.0

        age = row['Patient Age']
        sex = 1 if row['Patient Sex'] == "Male" else 0

        feature_vector = np.concatenate(([age, sex], img_flat))

        X_features.append(feature_vector)
        y_labels.append(row["Diagnosis"])

    X = np.array(X_features)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # 3) CACHE'E KAYDET
    save_cache(X, y, img_size, CSV_PATH, SCRIPT_PATH)

    print(f"[INFO] Özellik çıkarma tamamlandı. X shape={X.shape}, y shape={y.shape}")

    return X, y, le
