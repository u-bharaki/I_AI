# model_knn.py

from preprocessing import load_and_prepare_data
from utils import split_and_scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def main():
    X, y, _ = load_and_prepare_data()
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    model = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[KNN] Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
