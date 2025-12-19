import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def compare_push_vs_stabilize():
    log_files = glob.glob("training_log_*.csv")

    target_files = [f for f in log_files if 'push' in f.lower() or 'stabilize' in f.lower()]

    if len(target_files) < 2:
        print("HATA: Karşılaştırma için 'push' ve 'stabilize' logları aynı klasörde bulunamadı!")
        return

    plt.figure(figsize=(20, 12))

    plt.subplot(2, 2, 1)
    for file in target_files:
        df = pd.read_csv(file)
        name = "PUSH" if 'push' in file.lower() else "STABILIZE"
        color = 'purple' if name == "PUSH" else 'brown'

        plt.plot(df['val_accuracy'], label=f'{name} (Val Accuracy)', linewidth=3, color=color)
        plt.plot(df['accuracy'], linestyle='--', alpha=0.5, color=color, label=f'{name} (Train Accuracy)')

    plt.title('PUSH vs STABILIZE: Başarı Oranları', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    for file in target_files:
        df = pd.read_csv(file)
        name = "PUSH" if 'push' in file.lower() else "STABILIZE"
        plt.plot(df['val_loss'], label=f'{name} Loss', linewidth=3)

    plt.title('PUSH vs STABILIZE: Kayıp (Hata) Karşılaştırması', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    for file in target_files:
        df = pd.read_csv(file)
        name = "PUSH" if 'push' in file.lower() else "STABILIZE"
        ratio = df['accuracy'] / df['val_accuracy']
        plt.plot(ratio, label=f'{name} Overfit Ratio', linewidth=3)

    plt.axhline(y=1.0, color='black', linestyle='-', alpha=0.6)
    plt.ylim(0.90, 1.40) # PUSH'u çok uçurmadan, STABILIZE ile farkını net gösteren aralık
    plt.title('Ezberleme Oranı (Ratio)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    for file in target_files:
        df = pd.read_csv(file)
        name = "PUSH" if 'push' in file.lower() else "STABILIZE"
        plt.plot(df['val_accuracy'].diff(), label=f'{name} Öğrenme İvmesi', alpha=0.7)

    plt.title('Öğrenme İvmesi (Epoch bazlı değişim)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('PUSH (Zirve Performans) vs STABILIZE (Güvenilir Performans) Detaylı Analizi', fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = "push_vs_stabilize_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Özel karşılaştırma raporu hazır: {save_path}")

if __name__ == "__main__":
    compare_push_vs_stabilize()