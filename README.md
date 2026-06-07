# Hybrid-ID: Hibrit Görüntü Doğrulama Sistemi

Üretken yapay zekâ modelleri tarafından oluşturulan görselleri
ve dijital manipülasyonları tespit etmek için geliştirilmiş çok katmanlı,
açıklanabilir bir hibrit analiz sistemi.

<br>

## Proje Hakkında

Günümüzde GAN ve difüzyon tabanlı modellerin ürettiği görseller gerçek
fotoğraflardan ayırt edilemez hale gelmiştir. Hybrid-ID bu sorunu tek bir
yönteme bağlı kalmadan, üç farklı analiz katmanını birleştirerek çözer:
metadata analizi, piksel düzeyinde sıkıştırma incelemesi (ELA) ve derin
öğrenme tabanlı sınıflandırma (ResNet50). Karar süreci Grad-CAM ısı
haritalarıyla şeffaf biçimde kullanıcıya sunulur.

<br>

## Mimari
H(x) = 0.20 × Sm  +  0.35 × Se  +  0.45 × Sc
| Katman | Yöntem | Ağırlık |
|---|---|---|
| Metadata Analizi | EXIF dijital iz incelemesi | 0.20 |
| ELA | Piksel sıkıştırma tutarsızlığı | 0.35 |
| CNN (ResNet50) | Derin öğrenme sınıflandırması | 0.45 |

H(x) > 0.40 → **FAKE** &nbsp;&nbsp;|&nbsp;&nbsp; H(x) ≤ 0.40 → **REAL**

<br>

## Özellikler

- Metadata silinmiş görsellerde bile ELA ve CNN üzerinden çalışmaya devam eder
- Sahtelik türünü sınıflandırır: **AI Üretimi** veya **Dijital Montaj**
- Grad-CAM ile şüpheli bölgeler ısı haritası olarak işaretlenir
- Güven seviyesi skorlanır: YÜKSEK / ORTA / DÜŞÜK
- Analiz sonucu PDF raporu olarak dışa aktarılır
- Görüntü başına ortalama işlem süresi ~2.22 saniye

<br>

## Performans

16.000 görüntülük veri seti — 8.000 gerçek, 8.000 AI üretimi

| Yöntem | Doğruluk | F1 | AUC |
|---|---|---|---|
| Metadata | %49.97 | 0.000 | 0.500 |
| ELA | %50.09 | 0.624 | 0.503 |
| CNN (ResNet50) | %68.67 | 0.733 | 0.770 |
| CNN + ELA | %70.38 | 0.741 | 0.786 |
| **Hybrid-ID** | **%72.44** | **0.728** | **0.804** |

<br>

## Teknolojiler

`Python` `PyTorch` `ResNet50` `OpenCV` `Grad-CAM`
`ELA` `EXIF Metadata` `Flask` `NumPy` `SciPy` `Pillow`

<br>
