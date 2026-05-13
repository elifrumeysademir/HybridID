import os
import io
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI olmayan ortamlarda render için
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageEnhance, ImageChops
from scipy import ndimage

# ──────────────────────────────────────────────────────────────────────────────
# Sabitler
# ──────────────────────────────────────────────────────────────────────────────
ELA_QUALITY       = 90    # Yeniden sıkıştırma kalitesi (standart ELA eşiği)
ELA_AMPLIFY       = 10    # Fark haritası amplifikasyon çarpanı
REGION_THRESHOLD  = 1.5   # Şüpheli bölge eşiği: mean + k * std
MIN_REGION_AREA   = 100   # Piksel cinsinden minimum bölge alanı (gürültü filtresi)
MAX_ANALYSIS_SIZE = 1024  # Büyük görseller bu boyuta küçültülür (performans)
SCORE_THRESHOLD   = 0.50  # is_manipulation_detected için güven skoru eşiği
OUTPUT_DIR        = os.path.join(os.path.dirname(__file__), "..", "outputs", "ela_heatmaps")


# ──────────────────────────────────────────────────────────────────────────────
# 1. ELA Hesaplama Motoru
# ──────────────────────────────────────────────────────────────────────────────
def perform_ela(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Görseli JPEG formatında yeniden sıkıştırır ve hata haritasını döner. [cite: S3-P0]

    Args:
        image: Orijinal PIL Image nesnesi.

    Returns:
        ela_image  : Amplifikasyon uygulanmış ELA görseli (PIL Image).
        error_array: Gri tonlamalı hata haritası (numpy array, 0–255).
    """
    # RGB'ye çevir (RGBA, P modu gibi formatları normalize et)
    original = image.convert("RGB")

    # Yeniden sıkıştırma: bellek üzerinde JPEG encode → decode
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=ELA_QUALITY)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    # Piksel farkı
    diff = ImageChops.difference(original, recompressed)

    # Amplifikasyon: farkı görünür kıl
    enhancer = ImageEnhance.Brightness(diff)
    ela_image = enhancer.enhance(ELA_AMPLIFY)

    # Numpy array — gri ton ortalaması
    diff_array = np.array(diff, dtype=np.float32)
    error_array = diff_array.mean(axis=2)  # (H, W)

    return ela_image, error_array


# ──────────────────────────────────────────────────────────────────────────────
# 2. Bölge Tespiti
# ──────────────────────────────────────────────────────────────────────────────
def detect_manipulation_regions(error_array: np.ndarray) -> tuple[bool, float, list[dict]]:
    """
    ELA hata haritasında manipülasyon bölgelerini koordinat bazlı tespit eder. [cite: S3-P1]

    Args:
        error_array: `perform_ela()` çıktısı, (H, W) float array.

    Returns:
        is_detected  : Manipülasyon tespit edildi mi?
        ela_score    : 0.0–1.0 arası normalleştirilmiş güven skoru.
        regions      : Her bölge için {'x', 'y', 'w', 'h', 'score'} sözlükleri listesi.
    """
    mean_val = error_array.mean()
    std_val  = error_array.std()

    # Adaptif eşik: global ortalama + k * standart sapma
    threshold = mean_val + REGION_THRESHOLD * std_val

    # ELA skoru: eşiğin üzerindeki piksellerin oranı (normalleştirilmiş)
    above_threshold = (error_array > threshold)
    ratio = above_threshold.sum() / error_array.size

    # Skoru 0–1 arasına sıkıştır
    # (Önceden ratio * 20.0 idi, çok hassastı. Şimdi ratio * 5.0 yaparak toleransı artırdık)
    ela_score = float(min(ratio * 5.0, 1.0))  # %20 eşik → skor=1.0

    # Bağlı bölge etiketleme (scipy)
    labeled_array, num_features = ndimage.label(above_threshold)

    regions = []
    for region_id in range(1, num_features + 1):
        region_mask = labeled_array == region_id

        # Küçük gürültü bölgelerini filtrele
        if region_mask.sum() < MIN_REGION_AREA:
            continue

        # Bounding box koordinatları
        rows = np.any(region_mask, axis=1)
        cols = np.any(region_mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Bölge içi ortalama hata → bölge skoru
        region_mean = float(error_array[region_mask].mean())
        region_score = float(min(region_mean / 255.0, 1.0))

        regions.append({
            "x": int(x_min),
            "y": int(y_min),
            "w": int(x_max - x_min),
            "h": int(y_max - y_min),
            "score": round(region_score, 4)
        })

    # En yüksek skorlu bölgeler önce gelsin
    regions.sort(key=lambda r: r["score"], reverse=True)

    is_detected = ela_score > SCORE_THRESHOLD

    return is_detected, round(ela_score, 4), regions


# ──────────────────────────────────────────────────────────────────────────────
# 3. Isı Haritası Görselleştirme
# ──────────────────────────────────────────────────────────────────────────────
def save_ela_heatmap(
    original_image: Image.Image,
    ela_image: Image.Image,
    error_array: np.ndarray,
    regions: list[dict],
    file_path: str
) -> str | None:
    """
    ELA ısı haritasını 3 panelli olarak kaydeder. [cite: S3-P0]

    Panel 1 — Orijinal görsel
    Panel 2 — ELA fark haritası (hot colormap)
    Panel 3 — Orijinal + şüpheli bölge bounding box'ları

    Args:
        original_image : Orijinal PIL Image.
        ela_image      : `perform_ela()` çıktısı PIL Image.
        error_array    : `perform_ela()` çıktısı numpy array.
        regions        : `detect_manipulation_regions()` çıktısı bölge listesi.
        file_path      : Analiz edilen orijinal görselin yolu (çıktı adı için).

    Returns:
        Kaydedilen PNG dosyasının mutlak yolu. Hata durumunda None.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.abspath(
        os.path.join(OUTPUT_DIR, f"{base_name}_ela.png")
    )

    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("#1a1a2e")

        # ── Panel 1: Orijinal ──────────────────────────────────────────────
        axes[0].imshow(original_image)
        axes[0].set_title("Orijinal Görsel", color="white", fontsize=13, fontweight="bold")
        axes[0].axis("off")

        # ── Panel 2: ELA Isı Haritası ─────────────────────────────────────
        im = axes[1].imshow(error_array, cmap="hot", vmin=0, vmax=error_array.max())
        axes[1].set_title("ELA Hata Haritası", color="white", fontsize=13, fontweight="bold")
        axes[1].axis("off")
        cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        cbar.set_label("Hata Seviyesi", color="white", fontsize=9)

        # ── Panel 3: Tespitler ────────────────────────────────────────────
        axes[2].imshow(original_image)
        axes[2].set_title(
            f"Şüpheli Bölgeler ({len(regions)} alan)",
            color="white", fontsize=13, fontweight="bold"
        )
        axes[2].axis("off")

        for region in regions:
            rect = patches.Rectangle(
                (region["x"], region["y"]),
                region["w"],
                region["h"],
                linewidth=2,
                edgecolor="#ff4444",
                facecolor="none",
                alpha=0.85
            )
            axes[2].add_patch(rect)
            axes[2].text(
                region["x"] + 2,
                region["y"] - 5,
                f"{region['score']:.2f}",
                color="#ff4444",
                fontsize=8,
                fontweight="bold"
            )

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_edgecolor("#333366")

        plt.suptitle(
            f"Hybrid-ID ELA Analizi — {os.path.basename(file_path)}",
            color="white", fontsize=15, fontweight="bold", y=1.01
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        return output_path

    except Exception as e:
        print(f"Hata: Isı haritası kaydedilemedi → {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 4. Ana Analiz Fonksiyonu
# ──────────────────────────────────────────────────────────────────────────────
def analyze_ela(file_path: str, save_heatmap: bool = True) -> dict:
    """
    ELA analizi yapar ve S6 entegrasyonuna uygun JSON sözlüğü döner. [cite: S3-P0]

    `metadata_analyzer.analyze_metadata()` ile aynı yapı kullanılır; böylece
    S6 hibrit skorlama katmanı her iki modülü tutarlı şekilde okuyabilir.

    Args:
        file_path    : Analiz edilecek görsel dosyasının yolu (.jpg, .jpeg veya .png).
        save_heatmap : True ise ısı haritası PNG olarak kaydedilir.

    Returns:
        {
            "success"               : bool,
            "error_message"         : str | None,
            "file_path"             : str,
            "is_manipulation_detected": bool,
            "ela_score"             : float,   # 0.0–1.0
            "suspicious_regions"    : list[dict],
            "ela_heatmap_path"      : str | None
        }
    """
    result = {
        "success":                   False,
        "error_message":             None,
        "file_path":                 file_path,
        "is_manipulation_detected":  False,
        "ela_score":                 0.0,
        "suspicious_regions":        [],
        "ela_heatmap_path":          None,
    }

    # ── Dosya kontrolü ────────────────────────────────────────────────────────
    if not os.path.exists(file_path):
        msg = f"Dosya bulunamadı → {file_path}"
        print(f"Hata: {msg}")
        result["error_message"] = msg
        return result

    valid_extensions = {".jpg", ".jpeg", ".png"}
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in valid_extensions:
        msg = f"Desteklenmeyen format ({ext}). Kabul: .jpg, .jpeg, .png"
        print(f"Hata: {msg}")
        result["error_message"] = msg
        return result

    # ── Analiz ───────────────────────────────────────────────────────────────
    try:
        image = Image.open(file_path)

        # PNG uyarısı: ELA JPEG'e özeldir, PNG sonuçları daha az güvenilir
        if ext.lower() == ".png":
            print("Uyarı: PNG görseli algılandı. ELA JPEG'e özel bir tekniktir; "
                  "sonuçlar daha az kesin olabilir.")

        # Büyük görseller küçültülür (performans optimizasyonu)
        orig_w, orig_h = image.size
        if max(orig_w, orig_h) > MAX_ANALYSIS_SIZE:
            scale = MAX_ANALYSIS_SIZE / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            print(f"  Not: Görsel analiz için {orig_w}x{orig_h} → {new_w}x{new_h} boyutuna ölçeklendirildi.")

        print(f"\n--- Hybrid-ID ELA Analiz Raporu: {file_path} ---")

        ela_image, error_array = perform_ela(image)
        is_detected, ela_score, regions = detect_manipulation_regions(error_array)

        result["is_manipulation_detected"] = is_detected
        result["ela_score"]                = ela_score
        result["suspicious_regions"]       = regions
        result["success"]                  = True

        # ── Konsol çıktısı ────────────────────────────────────────────────
        print(f"\n  ELA Skoru        : {ela_score:.4f}  (eşik: {SCORE_THRESHOLD})")
        print(f"  Şüpheli Bölge    : {len(regions)} adet")

        if is_detected:
            print("\n  [!] Manipülasyon İzine Rastlandı [!]")
            for i, r in enumerate(regions[:5], 1):  # İlk 5 bölgeyi göster
                print(f"      Bölge {i}: x={r['x']}, y={r['y']}, "
                      f"genişlik={r['w']}, yükseklik={r['h']}, "
                      f"skor={r['score']:.4f}")
        else:
            print("\n  [+] Temiz: Anlamlı manipülasyon bölgesi tespit edilmedi.")

        # ── Isı haritası kaydet ───────────────────────────────────────────
        if save_heatmap:
            heatmap_path = save_ela_heatmap(
                image, ela_image, error_array, regions, file_path
            )
            result["ela_heatmap_path"] = heatmap_path
            if heatmap_path:
                print(f"\n  Isı haritası kaydedildi → {heatmap_path}")
            else:
                print("\n  Uyarı: Isı haritası kaydedilemedi.")

        print("-------------------------------------------\n")
        return result

    except Exception as e:
        msg = f"Beklenmeyen hata: {e}"
        print(f"Hata: {msg}")
        result["error_message"] = msg
        return result


# ──────────────────────────────────────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Hybrid-ID ELA Analiz Aracı (Sprint 3)")
    print("=" * 55)
    print(f"  Desteklenen formatlar : .jpg, .jpeg, .png")
    print(f"  Isı haritası çıktısı  : outputs/ela_heatmaps/")
    print("=" * 55)

    while True:
        user_input = input("\nDosya yolunu girin (Çıkış: 'q'): ").strip().strip("\"'")

        if user_input.lower() in ("q", "exit", "quit"):
            print("Program kapatılıyor...")
            break

        if not user_input:
            continue

        result_dict = analyze_ela(user_input)

        if result_dict["success"]:
            status = "MANİPÜLE" if result_dict["is_manipulation_detected"] else "TEMİZ"
            print(f"[BAŞARILI] Karar: {status} | "
                  f"Skor: {result_dict['ela_score']:.4f} | "
                  f"Bölge: {len(result_dict['suspicious_regions'])} | "
                  f"JSON çıktısı S6 entegrasyonuna hazır.")
        else:
            print(f"[BAŞARISIZ] Neden: {result_dict.get('error_message')}")
