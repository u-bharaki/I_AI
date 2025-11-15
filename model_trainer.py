import pandas as pd
import numpy as np
import cv2  # OpenCV (görüntü işleme için)
import os
from tqdm import tqdm  # Güzel bir ilerleme çubuğu için
import warnings

# Model kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Uyarıları gizle (Logistic Regression'daki max_iter uyarıları için)
warnings.filterwarnings('ignore')

# --- 1. AYARLAR ---
# BU KISIMLARI KENDİ PROJENİZE GÖRE DÜZENLEYİN
IMG_SIZE = 32  # 32x32 piksel (Daha hızlı eğitim için düşük tutuldu)
DATA_DIR = "preprocessed_images"  # Görüntülerin olduğu ana klasör
CSV_PATH = "cleaned_file_final.csv"  # Görüntüdeki CSV dosyanızın yolu
RANDOM_STATE = 42  # Sonuçların tekrarlanabilir olması için


# --- --- --- --- --- --- --- --- --- --- --- ---

def load_and_prepare_data(csv_path, data_dir, img_size):
    """
    CSV dosyasını okur, görüntüleri yükler, demografik verilerle birleştirir
    ve modellerin anlayacağı X, y formatına getirir.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"HATA: {csv_path} dosyası bulunamadı. Lütfen CSV_PATH değişkenini kontrol edin.")
        return None, None, None

    # Eksik verileri temizle
    df.dropna(subset=['filepath', 'Diagnosis', 'Patient Age', 'Patient Sex'], inplace=True)
    df['Patient Age'] = pd.to_numeric(df['Patient Age'], errors='coerce')
    df.dropna(subset=['Patient Age'], inplace=True)  # Sayıya dönüşmeyenleri at

    print(f"Toplam {len(df)} geçerli satır bulundu. Özellikler çıkarılıyor...")

    X_features = []  # 3074 (2 + 3072) boyutlu vektörler buraya
    y_labels = []  # Etiketler buraya

    # tqdm ile ilerleme çubuğu oluştur
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Görüntüler işleniyor"):
        try:
            # 1. Görüntü Özellikleri (3072 özellik)
            img_path = os.path.join(data_dir, row['filepath'])
            if not os.path.exists(img_path):
                # print(f"Uyarı: {img_path} bulunamadı, atlanıyor.")
                continue

            image = cv2.imread(img_path)
            img_resized = cv2.resize(image, (img_size, img_size))
            img_flat = img_resized.flatten()
            img_norm = img_flat / 255.0

            # 2. Demografik Özellikler (2 özellik)
            age = row['Patient Age']
            sex = 1 if row['Patient Sex'] == 'Male' else 0

            # 3. Tüm Özellikleri Birleştir (3074 özellik)
            final_feature_vector = np.concatenate(([age, sex], img_norm))

            X_features.append(final_feature_vector)
            y_labels.append(row['Diagnosis'])

        except Exception as e:
            # print(f"Hata: {img_path} işlenemedi. Hata: {e}, atlanıyor.")
            pass  # Bozuk dosyaları veya yolları atla

    if not X_features:
        print("HATA: Hiçbir görüntü işlenemedi. DATA_DIR yolunu ve dosya adlarını kontrol edin.")
        return None, None, None

    # Listeleri NumPy dizisine çevir
    X = np.array(X_features)

    # Etiketleri metinden sayıya çevir (Label Encoding)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    print(f"\nÖzellik çıkarma tamamlandı: X şekli {X.shape}, y şekli {y.shape}")
    return X, y, le


def main():
    # --- 2. VERİ HAZIRLAMA ---
    X, y, label_encoder = load_and_prepare_data(CSV_PATH, DATA_DIR, IMG_SIZE)

    # Veri yüklenemediyse programı durdur
    if X is None:
        return

    # --- 3. TRAIN/TEST SPLIT VE ÖLÇEKLEME (SCALING) ---
    print("\nVeri Train/Test olarak ayrılıyor ve ölçekleniyor...")

    # Veriyi Train ve Test olarak ayır
    # stratify=y, sınıfların (Diagnosis) train ve test setine orantılı dağılmasını sağlar
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,  # Verinin %25'ini test için ayır
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Veriyi Ölçekle (StandardScaler)
    # Bu adım KNN ve Logistic Regression için KRİTİKTİR.
    # Yaş (0-90) ve Pikseller (0-1) farklı ölçeklerde olduğu için aynı ölçeğe getirir.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Veri eğitime hazır.")

    # --- 4. MODELLERİN TANIMLANMASI ---

    # Eğitilecek tüm modelleri bir sözlük (dictionary) içinde tanımla
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            solver='saga',  # Büyük veri setleri için daha hızlı bir çözücü
            max_iter=1000,  # Modelin yakınsaması için yeterli iterasyon
            n_jobs=-1  # Tüm CPU çekirdeklerini kullan
        ),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(
            n_neighbors=7,  # Komşu sayısı (deneyerek ayarlanabilir)
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=10  # Aşırı öğrenmeyi (overfitting) engellemek için derinliği sınırla
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,  # 100 adet karar ağacı kullan
            max_depth=10,
            n_jobs=-1
        )
    }

    # Sonuçları saklamak için bir sözlük
    results = {}

    # --- 5. MODEL EĞİTİMİ VE DEĞERLENDİRME DÖNGÜSÜ ---
    print("\n" + "=" * 30)
    print(" MODEL EĞİTİMİ BAŞLIYOR ")
    print("=" * 30)

    for name, model in models.items():
        print(f"\n[{name}] modeli eğitiliyor...")

        # Modeli ölçeklenmiş veri ile eğit
        model.fit(X_train_scaled, y_train)

        # Test verisi ile tahmin yap
        y_pred = model.predict(X_test_scaled)

        # Başarı oranını (accuracy) hesapla
        score = accuracy_score(y_test, y_pred)

        # Sonucu kaydet
        results[name] = score

        print(f"[{name}] Test Başarı Oranı: {score * 100:.2f}%")

    # --- 6. SONUÇLARIN KARŞILAŞTIRILMASI ---
    print("\n" + "=" * 40)
    print(" TÜM MODELLERİN KARŞILAŞTIRMASI ")
    print("=" * 40)

    # Sonuçları bir DataFrame'e dönüştürerek daha güzel göster
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Test Accuracy'])
    results_df['Test Accuracy'] = (results_df['Test Accuracy'] * 100).round(2)  # Yüzdeye çevir

    # En başarılıdan en başarısız olana doğru sırala
    results_df = results_df.sort_values(by='Test Accuracy', ascending=False)

    print(results_df)

    print("\n" + "-" * 40)
    best_model_name = results_df.index[0]
    best_model_score = results_df.iloc[0, 0]
    print(f"🏆 En başarılı model: {best_model_name} (Başarı: {best_model_score}%)")
    print("-" * 40)


if __name__ == "__main__":
    main()