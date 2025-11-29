# compare_sizes.py

from utils import save_model

import time
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score)

from preprocessing import load_and_prepare_data

GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

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

    results = []

    for name, model in models.items():
        print(f"\n[{name}] eğitiliyor...")
        model.fit(X_train, y_train)

        save_model(model.model, name, img_size)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        try:
            roc = roc_auc_score(y_test, model.model.predict_proba(X_test), multi_class="ovo", average="weighted")
        except:
            roc = None

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name} - IMG {img_size}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"{GRAPH_DIR}/cm_{name}_{img_size}.png")
        plt.close()

        results.append({
            "model": name,
            "img_size": img_size,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
            "train_time": model.train_time
        })

        print(f"{name}: Accuracy={acc*100:.2f}%  | Precision={prec:.2f} | Recall={rec:.2f} | F1={f1:.2f} | Time={model.train_time:.2f} sec")

    return results

def plot_comparison(all_results):
    df = pd.DataFrame(all_results)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="img_size", y=metric, hue="model")
        plt.title(f"{metric.upper()} Comparison by IMG_SIZE")
        plt.xlabel("IMG SIZE")
        plt.ylabel(metric.upper())
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{GRAPH_DIR}/{metric}_comparison.png")
        plt.close()


def main():
    all_results = []

    for size in [32, 64, 128]:
        print(f"\n[INFO] IMG_SIZE={size} için veri yükleniyor...")

        X, y, _ = load_and_prepare_data(img_size=size)

        results = evaluate_models(X, y, size)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv("graphs/model_metrics.csv", index=False)

    plot_comparison(all_results)
    print("\nAll metrics are saved -> graphs/model_metrics.csv")
    print("Graphs saved into -> graphs/ directory.")

if __name__ == "__main__":
    main()
