# Kalp Hastalığı Tahmini - Topluluk Öğrenmesi Projesi

Bu proje, "Heart Disease Prediction" veri seti üzerinde farklı Topluluk Öğrenmesi (Ensemble Learning) algoritmalarını kullanarak kalp hastalığı riskini tahmin etmeyi amaçlar. Proje kapsamında Bagging, Boosting ve Stacking yöntemleri karşılaştırılmış; PCA ve Hiperparametre Optimizasyonu ile modellerin başarısı artırılmıştır.

## Proje Özellikleri

- **Üç Temel Algoritma:**
  - **Random Forest** (Bagging)
  - **AdaBoost** (Boosting)
  - **Stacking** (Hibrit: RF, AdaBoost ve SVC birleşimi)
- **Gelişmiş Analizler:**
  - **PCA (Temel Bileşenler Analizi):** Veri boyutunu indirgeyerek gürültüyü azaltma.
  - **GridSearchCV:** Hiperparametre optimizasyonu ile en iyi model performansını yakalama.
  - **Cross-Validation:** 5-Katlı Çapraz Doğrulama ile overfitting (aşırı öğrenme) kontrolü.
- **Kapsamlı Görselleştirme:** Karmaşıklık matrisleri, özellik önem dereceleri, PCA varyans analizi ve metrik karşılaştırma grafikleri.

## Dosya Yapısı

- `main_proje.py`: Projenin tüm analiz süreçlerini yöneten ana dosya.
- `data_preprocessing.py`: Veri temizleme, ölçeklendirme ve PCA modülü.
- `visualization.py`: Tüm grafiklerin üretimini sağlayan görselleştirme modülü.
- `Heart_Disease_Prediction.csv`: Kullanılan kalp hastalığı veri seti.
- `final_sonuclar.txt`: Tüm modellerin Accuracy, Precision, Recall ve F1 skorlarını içeren tablo.

## Özet Sonuçlar

Proje sonunda en yüksek başarıya **%90.74** doğruluk oranı ile **AdaBoost** (PCA sonrası) ulaşmıştır. **Stacking** ve **Random Forest** modelleri de %88.89 performans sergileyerek oldukça güvenilir sonuçlar vermiştir.

![PCA Başarı Analizi](pca_başarı_analizi.png)

## Detaylı Analiz Görselleri

### 1. Performans Karşılaştırması (Tüm Metrikler)

![Metrik Karşılaştırma](tüm_metrikler_karşılaştırma.png)

### 2. PCA Açıklanan Varyans

![PCA Varyans](pca_varyans_analizi.png)

### 3. Hata Matrisleri (Confusion Matrices - Advanced)

![Karmaşıklık Matrisleri](karmaşıklık_matrisleri_advanced.png)

### 4. Öznitelik Önem Dereceleri (Örnek: Random Forest)

![Özellik Önemi](özellik_önemi_random_forest_baseline.png)

Hastalık teşhisinde en belirleyici faktörler olarak **Talasemi (Thal)**, **Büyük Damar Sayısı (Ca)** ve **Göğüs Ağrısı Tipi (CP)** öne çıkmıştır.
