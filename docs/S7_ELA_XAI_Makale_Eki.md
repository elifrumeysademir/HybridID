# Hybrid-ID Projesi: ELA ve XAI Metodolojisi (Makale Eki)

Bu rapor, Hybrid-ID sistemindeki piksel hata seviyesi analizinin (ELA) ve derin öğrenme modelinin kararlarını şeffaflaştıran Açıklanabilir Yapay Zeka (XAI) entegrasyonunun teknik altyapısını özetlemektedir.

## 1. Error Level Analysis (ELA) Bölümü
Piksel Hata Seviyesi Analizi, görsele sonradan eklenen dijital montajları (kopyala-yapıştır) tespit etmek için tasarlanmıştır.

* **Teknik Altyapı:** Görüntü işleme süreçleri için **PIL (Pillow)** ve **NumPy**; matematiksel hesaplamalar ve kümeleme için **SciPy (`ndimage`)** kütüphaneleri kullanılmıştır. 
* **Yöntem (Recompression):** Görsel bellekte standart bir kalite oranıyla (Quality=90) yeniden JPEG formatında sıkıştırılır. Orijinal piksel dizilimi ile yeniden sıkıştırılmış dizilim arasındaki piksel farkları (`ImageChops.difference`) hesaplanarak görünürlüğünü artırmak adına 10 kat amplifikasyon (parlaklık artırımı) uygulanır.
* **Adaptif Eşikleme (Adaptive Thresholding):** Matris haline getirilen hata değerleri üzerinden, görselin kendi karakteristiğine göre dinamik bir eşik değeri (Ortalama + 1.5 * Standart Sapma) belirlenir.
* **Kümeler ve Gürültü Filtreleme:** Bu eşiğin üzerinde kalan piksel yığınları SciPy kütüphanesi yardımıyla etiketlenerek şüpheli bölgeler (bounding box) oluşturulur. Yanlış alarmları önlemek için 100 pikselden küçük kümeler (noise) filtrelenir.
* **Sonuç:** Orijinal arka plan ile sonradan eklenen objenin farklı JPEG sıkıştırma geçmişleri (compression history) olduğu için, eklenen objeler sistemde yüksek hata seviyesine sahip kırmızı/sıcak bölgeler olarak başarıyla yakalanır.

---

## 2. Açıklanabilir Yapay Zeka (XAI) ve Grad-CAM Bölümü
CNN (ResNet50) modelinin "Kara Kutu" (Black-Box) problemini aşmak ve modelin tahmin yaparken tam olarak hangi dokuya veya desene odaklandığını kanıtlamak amacıyla **Grad-CAM (Gradient-weighted Class Activation Mapping)** algoritması entegre edilmiştir.

* **Teknik Altyapı:** Model içi gradyan takibi için **TensorFlow/Keras (`tf.GradientTape`)**; ısı haritası ve piksel bindirme (overlay) işlemleri için **OpenCV (`cv2`)** kullanılmıştır.
* **Matematiksel Süreç (Backpropagation):** Model sahtelik tahmini yaptıktan sonra, ağın en son konvolüsyonel katmanı olan `conv5_block3_out` katmanına doğru türev/gradyan hesaplaması yapılır. 
* **Ağırlıklandırma ve Aktivasyon:** Çıkarılan gradyanların kanal bazında (Global Average Pooling) ortalaması alınarak her bir "Feature Map" (Öznitelik Haritası) için önem ağırlıkları (importance weights) bulunur. Haritalar bu ağırlıklarla çarpılıp toplanır. Sadece hedef sınıfı (FAKE tahmini) olumlu yönde etkileyen pikselleri izole etmek için aktivasyon matrisine **ReLU** uygulanır.
* **Görselleştirme (Overlay):** Matematiksel olarak elde edilen bu şüphe matrisi, orijinal fotoğraf çözünürlüğüne çekilir. OpenCV üzerinden **JET renk haritası (colormap)** ile renklendirilip, Alpha Blending yöntemiyle (saydamlık: 0.45) orijinal görselin üzerine dijital bir kanıt (heatmap) olarak basılır. 
