# compare_sizes.py

import time
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocessing import load_and_prepare_data


class ProgressBarModel:
    """Scikit-Learn modeli için training sırasında progress bar gösterir."""

    def __init__(self, model, steps=60):
        self.model = model
        self.steps = steps
        self.train_time = None

    def fit(self, X, y):
        bar = tqdm(total=self.steps,
                   desc=f"Training {self.model.__class__.__name__}",
                   ncols=90,
                   leave=True)

        start = time.time()

        for _ in range(self.steps):
            time.sleep(0.01)
            bar.update(1)

        bar.close()

        # Gerçek model eğitimi
        self.model.fit(X, y)
        self.train_time = time.time() - start
        return self

    def predict(self, X):
        return self.model.predict(X)


def evaluate_models(X, y, img_size):
    print(f"\n=== IMG_SIZE={img_size} → MODEL EĞİTİMİ ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression":
            ProgressBarModel(LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1)),

        "KNN (k=7)":
            ProgressBarModel(KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),

        "Decision Tree":
            ProgressBarModel(DecisionTreeClassifier(max_depth=10)),

        "Random Forest":
            ProgressBarModel(RandomForestClassifier(n_estimators=100,
                                                    max_depth=10,
                                                    n_jobs=-1))
    }

    results = {}

    for name, model in models.items():
        print(f"\n[{name}] eğitiliyor...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results[name] = (acc, model.train_time)

        print(f"{name}: Accuracy={acc*100:.2f}%  |  Süre={model.train_time:.2f} sn")

    return results


def main():
    all_results = {}

    for size in [32, 64, 128, 256]:
        print(f"\n[INFO] IMG_SIZE={size} için veri yükleniyor...")

        X, y, _ = load_and_prepare_data(img_size=size)

        results = evaluate_models(X, y, size)
        all_results[size] = results

    print("\n========== KARŞILAŞTIRMA ==========")
    for size in all_results:
        print(f"\n--- IMG_SIZE={size} ---")
        for model, (acc, dt) in all_results[size].items():
            print(f"{model}: {acc*100:.2f}%  (süre: {dt:.2f} sn)")


if __name__ == "__main__":
    main()
