import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_final_performance_dashboard():
    log_files = glob.glob("training_log_*.csv")

    if not log_files:
        print("HATA: Klasörde 'training_log_*.csv' formatında dosya bulunamadı!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    axes = axes.flatten()

    for file in log_files:
        try:
            df = pd.read_csv(file)

            base_name = os.path.basename(file)
            if 'log_' in base_name:
                stage_name = base_name.split('log_')[1].split('_')[0].upper()
            else:
                stage_name = base_name.replace('.csv', '').upper()

            axes[0].plot(df['accuracy'], label=f'Train Acc ({stage_name})', linewidth=2)

            axes[1].plot(df['val_accuracy'], label=f'Val Acc ({stage_name})', linewidth=2)

            axes[2].plot(df['val_loss'], label=f'Val Loss ({stage_name})', linestyle='--')

            overfit_ratio = df['accuracy'] / df['val_accuracy']
            axes[3].plot(overfit_ratio, label=f'Ratio ({stage_name})', linewidth=2.5)

        except Exception as e:
            print(f"{file} okunurken hata oluştu: {e}")

    axes[0].set_title('Eğitim Doğruluğu (Training Accuracy)', fontsize=14, fontweight='bold')
    axes[1].set_title('Doğrulama Doğruluğu (Validation Accuracy)', fontsize=14, fontweight='bold')
    axes[2].set_title('Doğrulama Kaybı (Validation Loss)', fontsize=14, fontweight='bold')
    axes[3].set_title('Overfitting Oranı (Training Acc / Val Acc)', fontsize=14, fontweight='bold')

    axes[3].set_ylim(0.80, 1.60)
    axes[3].axhline(y=1.0, color='black', linestyle='-', linewidth=2, alpha=0.6, label='İdeal Denge (1.0)')

    for ax in axes:
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Değer', fontsize=12)
        ax.legend(fontsize='x-small', loc='best', ncol=2)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle('Yapay Zeka Eğitim Süreçleri - Kapsamlı Performans Raporu', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_name = "final_performance_dashboard_report.png"
    plt.savefig(save_name, dpi=300)
    plt.close()

    print(f"\n✅ Başarılı! Rapor kaydedildi: {save_name}")
    print("Y-ekseni (Overfit) 0.80 - 1.60 arasına sabitlendi, farklar dengelendi.")

if __name__ == "__main__":
    plot_final_performance_dashboard()