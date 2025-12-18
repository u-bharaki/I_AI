import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from config import *
from dataset import load_dataframe, dataframe_to_dataset

def evaluate():
    df = load_dataframe()
    class_names = df[LABEL_COLUMN].astype("category").cat.categories.tolist()

    _, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_STATE)
    _, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=RANDOM_STATE)

    test_ds = dataframe_to_dataset(test_df, shuffle=False)
    model = tf.keras.models.load_model("models/best_improved_model.keras")

    predictions = model.predict(test_ds)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_df["label_id"].values

    print("\nYENİ PERFORMANS RAPORU:\n", classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Geliştirilmiş Confusion Matrix (Sınıf Ağırlıklı)')
    plt.savefig("logs/improved_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate()