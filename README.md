# I_AI

👁️ Göz Hastalıklarının Yapay Zeka ile Teşhisi ve Model Karşılaştırması
Bu proje, retina fundus görüntülerinden çeşitli göz hastalıklarının (diyabetik retinopati, katarakt vb.) otomatik teşhisi için çeşitli makine öğrenmesi ve derin öğrenme modellerini geliştirmeyi, eğitmeyi ve karşılaştırmayı amaçlamaktadır.

Proje şu anda geliştirme aşamasındadır; veri ön işleme (EDA) tamamlanmış olup, modellerin eğitimi ve karşılaştırılması süreci devam etmektedir.

🎯 Proje Amacı ve Temel Özellikler
👁️ Otomatik Teşhis: Fundus görüntülerini analiz ederek "Normal", "Katarakt", "Diyabetik Retinopati" gibi 8 farklı sınıftan birine ayırabilen bir sistem kurmak.

📊 Model Karşılaştırması: Aynı veri seti üzerinde klasik makine öğrenmesi ve derin öğrenme yaklaşımlarının performansını (Accuracy, Precision, F1-Score) sistematik olarak karşılaştırmak.

🤖 Çeşitli Yaklaşımlar: Görüntü verisi için hem ham piksel kullanan (CNN) hem de özellik vektörü (Feature Vector) kullanan (KNN, SVM vb.) modelleri test etmek.

📒 Raporlama: Hangi modelin hangi durumda daha başarılı olduğuna dair detaylı bir analiz raporu sunmak.

🛠️ Teknoloji Mimarisi
Ana Dil: Python

Veri Analizi ve İşleme: Pandas, NumPy, OpenCV, Matplotlib

Veri Yükleme: TensorFlow/Keras (ImageDataGenerator veya tf.data)

Klasik Modeller: Scikit-learn

Logistic Regression

Linear Regression

K-Nearest Neighbors (KNN)

Decision Tree & Random Forest

Support Vector Machine (SVM)

Derin Öğrenme Modelleri: TensorFlow / Keras

Multi-Layer Perceptron (MLP)

Convolutional Neural Network (CNN)

📁 Veri Seti
Bu projede, Ocular Disease Recognition (ODIR-5K) veri seti kullanılmaktadır.

full_of.csv dosyası hasta yaşı, cinsiyeti, görüntü dosya yolları (Left-Fundus, Right-Fundus) ve teşhis etiketlerini içerir.

Görüntüler, farklı teşhislere (N, D, G, C, A, H, M, O) göre sınıflandırılacaktır.

🤖 Karşılaştırılan Modeller
Projemiz, ham görüntü verilerini işlemek için iki temel yaklaşımı karşılaştırmaktadır:

Klasik MÖ Yaklaşımı (Özellik Vektörü ile):

Görüntülerden çıkarılan özellik vektörleri (Feature Vectors) kullanılır.

Modeller: Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, GBM, LDA

Uçtan Uca Derin Öğrenme Yaklaşımı (Ham Piksel ile):

Görüntüler doğrudan girdi olarak verilir.

Modeller: Multi-Layer Perceptron (MLP) (Düzleştirilmiş piksellerle), Convolutional Neural Network (CNN) (2D piksellerle)

🚀 Kurulum ve Çalıştırma
Projenin mevcut (veritabanı ve betik) kısmını çalıştırmak için:

1. Projeyi Klonlayın
Bash

git clone https://github.com/[kullanici_adiniz]/[proje_adiniz].git
cd [proje_adiniz]
2. (Öneri) Sanal Ortam Oluşturun
Bash

# Python sanal ortamını oluştur ve aktive et
python -m venv venv
source venv/bin/activate  # (Windows için: venv\Scripts\activate)
3. Bağımlılıkları Yükleyin
Proje betiklerinin ihtiyaç duyduğu Python kütüphanelerini yükleyin.

Bash

pip install -r requirements.txt
4. Veri Setini İndirin
Kaggle'dan ODIR-5K veri setini indirin. Görüntü klasörünü (preprocessed_images veya benzeri) data/ klasörü altına taşıyın. full_of.csv dosyasının data/ altında olduğundan emin olun.

5. EDA Notebook'unu Çalıştırın
Bash

jupyter notebook notebooks/EDA.ipynb
6. Modelleri Eğitin
(Not: train.py betiği tamamlandığında kullanılacaktır)

Bash

# Tüm modelleri eğitmek için
python scripts/train.py --model all

# Sadece CNN modelini eğitmek için
python scripts/train.py --model cnn
👥 Ekip
Berk Ülker

Duygu Akman

Ali Emre Yenihayat
