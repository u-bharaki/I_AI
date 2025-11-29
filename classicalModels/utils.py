# utils.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib


def split_and_scale(X, y, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def save_model(model, model_name, img_size):
    base_dir = "trained_classical_models"
    target_dir = os.path.join(base_dir, str(img_size))
    os.makedirs(target_dir, exist_ok=True)

    file_path = os.path.join(target_dir, f"{model_name}_{img_size}.pkl")
    joblib.dump(model, file_path)

    print(f"[MODEL SAVED] {file_path}")