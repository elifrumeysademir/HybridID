# HybridID - S5: CNN Model Eğitimi (Sunum Notları)

## 1. Sistemin Amacı
HybridID projemizin S5 aşamasında hedefimiz, dijital görüntülerin gerçek (real) mi yoksa sahte (fake/manipüle edilmiş) mi olduğunu tespit edebilen güçlü bir yapay zeka modeli geliştirmekti. Bu aşamada görüntülerin piksel tabanlı yapısal bozulmalarını analiz etmek için **Derin Öğrenme (CNN)** yöntemini kullandık.

## 2. Model Seçimi ve Mimari
Sıfırdan bir model eğitmek yerine, Google tarafından ImageNet veriseti üzerinde eğitilmiş rüştünü ispatlamış olan **ResNet50** mimarisini tercih ettik (Transfer Learning).
- **Taban Model (Base):** ResNet50 (Ağırlıklar donduruldu, özellik çıkarıcı olarak kullanıldı).
- **Özelleştirilmiş Sınıflandırıcı (Head):** ResNet50'nin sonuna 256 nöronlu bir Dense layer ve aşırı öğrenmeyi (overfitting) engellemek için %50 Dropout ekledik.

## 3. Eğitim Stratejisi (2 Aşamalı Eğitim)
Büyük veri seti ve kompleks model mimarisi sebebiyle eğitimi **Google Colab (T4 GPU)** üzerinde, iki aşamalı (Phase 1 & Phase 2) bir yaklaşımla gerçekleştirdik:
1. **Faz 1 (Head Eğitimi):** ResNet50'nin ana gövdesi donduruldu, sadece bizim eklediğimiz son katman eğitildi.
2. **Faz 2 (Fine-Tuning):** Modelin son blokları (Layer 143 ve sonrası) çözülerek, model ağırlıklarının kendi fake/real veri setimize daha iyi adapte olması sağlandı.

## 4. Elde Edilen İlk Sonuçlar ve Metrikler
Yapılan test veri seti değerlendirmesi sonucunda elde edilen güncel performans metriklerimiz:
- **Doğruluk (Accuracy):** %68
- **ROC-AUC Skoru:** 0.7698
- **Sahte Sınıfı F1-Skoru:** %70
- **Gerçek Sınıfı F1-Skoru:** %66

*(Not: Confusion matrix raporları ve test eğrileri `models/` klasöründe yer almaktadır ve sunumda grafik olarak gösterilebilir).*

## 5. İyileştirme Adımı: Hiperparametre Optimizasyonu
%68'lik doğruluk oranını production seviyesine (%90+) çıkarmak için manuel deneme-yanılma yerine **KerasTuner** entegrasyonu sağladık. 
- Optimum Nöron sayısı, Öğrenme Oranı (Learning Rate) ve L2 Regülarizasyonu gibi değerleri makine öğrenmesi algoritmalarıyla taramaya başladık.
- Buradan elde edeceğimiz konfigürasyon ile nihai modeli oluşturup S6 aşamasındaki Hibrit entegrasyona dahil edeceğiz.
