import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_training_history():
    log_files = glob.glob("logs/fast_train_log_*.csv")
    if not log_files:
        print("Hata: Log dosyası bulunamadı!")
        return

    latest_log = max(log_files, key=os.path.getctime)
    print(f"Grafikler çiziliyor: {latest_log}")

    data = pd.read_csv(latest_log)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(data['accuracy'], label='Eğitim Doğruluğu', color='blue', marker='o')
    plt.plot(data['val_accuracy'], label='Doğrulama Doğruluğu', color='orange', marker='o')
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(data['loss'], label='Eğitim Kaybı', color='red', marker='o')
    plt.plot(data['val_loss'], label='Doğrulama Kaybı', color='darkred', marker='o')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("logs/training_performance.png")
    plt.show()

if __name__ == "__main__":
    plot_training_history()