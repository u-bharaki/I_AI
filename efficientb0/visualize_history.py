import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_training_history():
    # 1. En son oluşturulan log dosyasını bul
    list_of_files = glob.glob('training_log_*.csv')
    if not list_of_files:
        print("HATA: Hiçbir log dosyası (.csv) bulunamadı!")
        return

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"İncelenen Dosya: {latest_file}")

    # 2. Veriyi Oku
    data = pd.read_csv(latest_file)

    # 3. Grafikleri Çiz
    epochs = range(1, len(data) + 1)

    plt.figure(figsize=(14, 6))

    # --- GRAFİK 1: ACCURACY (Doğruluk) ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, data['accuracy'], 'bo-', label='Training Acc (Ders Başarısı)')
    plt.plot(epochs, data['val_accuracy'], 'ro-', label='Validation Acc (Sınav Başarısı)')
    plt.title('Doğruluk (Accuracy) Grafiği')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # --- GRAFİK 2: LOSS (Hata) ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, data['loss'], 'b-', label='Training Loss (Ders Hatası)')
    plt.plot(epochs, data['val_loss'], 'r-', label='Validation Loss (Sınav Hatası)')
    plt.title('Kayıp (Loss) Grafiği - EN ÖNEMLİSİ')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_history()