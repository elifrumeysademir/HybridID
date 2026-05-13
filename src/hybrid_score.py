"""
HybridID - S6: Hibrit Confidence Score Sistemi
===============================================
Görevler:
  - S6/P0: EXIF + ELA + CNN katmanlarının entegrasyonu
  - S6/P0: Hibrit Confidence Score algoritmasının yazılması

Kullanım (standalone):
  python src/hybrid_score.py

Import (Streamlit/pipeline):
  from hybrid_score import run_full_analysis, load_cnn_model
"""

import os
import sys
import numpy as np

# src/ içinden çalıştırılabilmesi için proje kökünü path'e ekle
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Ağırlıklar & Sabitler
# ─────────────────────────────────────────────────────────────────────────────
W_METADATA = 0.15   # EXIF/AI imza katmanı ağırlığı (Düşürüldü)
W_ELA      = 0.30   # ELA görsel analiz katmanı ağırlığı (Düşürüldü)
W_CNN      = 0.55   # CNN derin öğrenme katmanı ağırlığı (Artırıldı)

VERDICT_THRESHOLD = 0.40  # HybridScore > bu değer → FAKE (Daha hassas yapıldı)

MODEL_PATH = os.path.join(_ROOT_DIR, "models", "resnet50_hybridid.h5")
IMG_SIZE   = (224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CNN Modeli Yükleme
# ─────────────────────────────────────────────────────────────────────────────
def load_cnn_model(model_path: str = MODEL_PATH):
    """
    Eğitilmiş ResNet50 modelini diskten yükler.

    Args:
        model_path: .h5 model dosyasının yolu.

    Returns:
        Yüklenmiş Keras modeli veya None (dosya yoksa).
    """
    import tensorflow as tf
    from tensorflow import keras

    if not os.path.exists(model_path):
        print(f"Uyarı: CNN model dosyası bulunamadı → {model_path}")
        return None

    # GPU bellek büyümesini sınırla (gereksiz OOM hatalarından kaçın)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model = keras.models.load_model(model_path)
    print(f"[CNN] Model yüklendi → {model_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. CNN Skoru
# ─────────────────────────────────────────────────────────────────────────────
def compute_cnn_score(image_path: str, model) -> float:
    """
    ResNet50 ile görsel üzerinde FAKE olasılığı tahmin eder.

    CNN class_indices: {'fake': 0, 'real': 1}
    Model çıktısı: sigmoid → 'real' sınıfı olasılığı
    Dolayısıyla FAKE skoru = 1 - model_output

    Args:
        image_path: Görsel dosyası yolu.
        model     : Yüklenmiş Keras modeli.

    Returns:
        fake_score: 0.0–1.0 arası FAKE olasılığı.
    """
    if model is None:
        return 0.5  # Model yoksa nötr skor

    from tensorflow.keras.applications.resnet50 import preprocess_input

    try:
        img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

        raw_output = float(model.predict(arr, verbose=0)[0][0])
        # Veri setindeki klasör isimleri yüzünden (örneğin 'Gerçek'=0, 'Sahte'=1)
        # model doğrudan sahtelik (FAKE) oranını veriyor.
        # Bu yüzden 1'den çıkarmayı iptal ettik.
        fake_score = raw_output
        return round(fake_score, 4)

    except Exception as e:
        print(f"Uyarı: CNN skoru hesaplanamadı → {e}")
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metadata Skoru (normalize)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_metadata_score(meta_result: dict) -> float:
    """
    metadata_analyzer.analyze_metadata() çıktısından 0-1 arası skor türetir.

    - Analiz başarısızsa veya metadata yoksa → 0.0 (bilgisiz, FAKE lehine yorum yok)
    - AI imzası bulunduysa → 1.0
    - Temiz metadata → 0.0
    """
    if not meta_result.get("success", False):
        return 0.0
    return 1.0 if meta_result.get("is_ai_detected", False) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hibrit Skor Hesabı
# ─────────────────────────────────────────────────────────────────────────────
def compute_hybrid_score(
    meta_result: dict,
    ela_result: dict,
    cnn_score: float,
) -> dict:
    """
    Üç katmandan gelen skorları ağırlıklı olarak birleştirir.

    Args:
        meta_result: `analyze_metadata()` çıktısı.
        ela_result : `analyze_ela()` çıktısı.
        cnn_score  : `compute_cnn_score()` çıktısı (0–1, FAKE olasılığı).

    Returns:
        {
            "hybrid_score" : float,      # 0.0–1.0
            "verdict"      : str,        # "FAKE" | "REAL"
            "confidence"   : str,        # "YÜKSEK" | "ORTA" | "DÜŞÜK"
            "layer_scores" : dict,       # metadata / ela / cnn
            "weights"      : dict,
        }
    """
    meta_score = _compute_metadata_score(meta_result)
    ela_score  = float(ela_result.get("ela_score", 0.0))
    # ela_score zaten 0-1 arasında normalize edilmiş (ela_analyzer)

    # --- YENİ EKLENEN YAPAY ZEKA KURALI ---
    # Eğer fotoğrafta kamera bilgisi (Make/Model/Tarih) yoksa VE 
    # ELA skoru sıfıra çok yakınsa (fotoğrafta kamera gürültüsü veya montaj izi yoksa, kusursuzsa)
    # Bu %99 ihtimalle doğrudan kaydedilmiş bir AI fotoğrafıdır.
    meta_dict = meta_result.get("extracted_metadata", {})
    has_camera_info = "Image Make" in meta_dict or "Image Model" in meta_dict or "EXIF DateTimeOriginal" in meta_dict

    if not has_camera_info and ela_score < 0.15:
        # CNN modelini dinlemeden skoru doğrudan yapay zeka şüphesiyle artır
        cnn_score = max(cnn_score, 0.85)

    hybrid = (
        W_METADATA * meta_score
        + W_ELA    * ela_score
        + W_CNN    * cnn_score
    )
    hybrid = round(float(np.clip(hybrid, 0.0, 1.0)), 4)

    verdict = "FAKE" if hybrid > VERDICT_THRESHOLD else "REAL"

    # Güven seviyesi: skorun eşikten uzaklığına göre
    distance = abs(hybrid - VERDICT_THRESHOLD)
    if distance >= 0.25:
        confidence = "YÜKSEK"
    elif distance >= 0.12:
        confidence = "ORTA"
    else:
        confidence = "DÜŞÜK"

    # Sahtelik türünü tahmin etme (AI mi yoksa Montaj mı?)
    manipulation_type = None
    if verdict == "FAKE":
        if meta_result.get("is_ai_detected", False):
            manipulation_type = "YAPAY ZEKA (AI) ÜRETİMİ"
        elif cnn_score > ela_score + 0.15:
            # CNN çok şüphelendiyse ama piksellerde montaj izi yoksa -> AI
            manipulation_type = "YAPAY ZEKA (AI) ÜRETİMİ"
        elif ela_score > cnn_score + 0.15:
            # ELA'da yüksek montaj izi varsa -> Photoshop/Montaj
            manipulation_type = "DİJİTAL MONTAJ (PHOTOSHOP)"
        else:
            manipulation_type = "MANİPÜLE / SAHTE"

    return {
        "hybrid_score": hybrid,
        "verdict"     : verdict,
        "manipulation_type": manipulation_type,
        "confidence"  : confidence,
        "layer_scores": {
            "metadata": round(meta_score, 4),
            "ela"     : round(ela_score,  4),
            "cnn"     : round(cnn_score,  4),
        },
        "weights": {
            "metadata": W_METADATA,
            "ela"     : W_ELA,
            "cnn"     : W_CNN,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tam Analiz (Pipeline Girişi)
# ─────────────────────────────────────────────────────────────────────────────
def run_full_analysis(image_path: str, model=None) -> dict:
    """
    Bir görsel üzerinde tüm analiz katmanlarını çalıştırır ve
    birleşik sonuç sözlüğü döner. S7 Streamlit arayüzü bu fonksiyonu çağırır.

    Args:
        image_path: Analiz edilecek görsel dosyası yolu.
        model     : Önceden yüklenmiş Keras modeli (None ise içerde yüklenir).

    Returns:
        {
            "success"           : bool,
            "error_message"     : str | None,
            "file_path"         : str,
            "verdict"           : "FAKE" | "REAL",
            "hybrid_score"      : float,
            "confidence"        : str,
            "layer_scores"      : dict,
            "weights"           : dict,
            "metadata_result"   : dict,
            "ela_result"        : dict,
            "cnn_score"         : float,
            "ela_heatmap_path"  : str | None,
            "gradcam_path"      : str | None,   # xai_engine tarafından doldurulur
        }
    """
    from metadata_analyzer import analyze_metadata
    from ela_analyzer      import analyze_ela

    result = {
        "success"          : False,
        "error_message"    : None,
        "file_path"        : image_path,
        "verdict"          : None,
        "hybrid_score"     : None,
        "confidence"       : None,
        "layer_scores"     : {},
        "weights"          : {"metadata": W_METADATA, "ela": W_ELA, "cnn": W_CNN},
        "metadata_result"  : {},
        "ela_result"       : {},
        "cnn_score"        : None,
        "ela_heatmap_path" : None,
        "gradcam_path"     : None,
    }

    if not os.path.exists(image_path):
        result["error_message"] = f"Dosya bulunamadı → {image_path}"
        return result

    print("\n" + "=" * 60)
    print(f"  HybridID — Tam Analiz Başlatıldı")
    print(f"  Görsel: {os.path.basename(image_path)}")
    print("=" * 60)

    # ── Katman 1: Metadata ────────────────────────────────────────────────────
    print("\n[1/3] Metadata analizi...")
    meta_result = analyze_metadata(image_path)
    result["metadata_result"] = meta_result

    # ── Katman 2: ELA ─────────────────────────────────────────────────────────
    print("[2/3] ELA analizi...")
    ela_result = analyze_ela(image_path, save_heatmap=True)
    result["ela_result"]       = ela_result
    result["ela_heatmap_path"] = ela_result.get("ela_heatmap_path")

    # ── Katman 3: CNN ─────────────────────────────────────────────────────────
    print("[3/3] CNN tahmini...")
    if model is None:
        model = load_cnn_model()
    cnn_score = compute_cnn_score(image_path, model)
    result["cnn_score"] = cnn_score

    # ── Hibrit Birleşim ───────────────────────────────────────────────────────
    print("\n[Hibrit] Skorlar birleştiriliyor...")
    hybrid = compute_hybrid_score(meta_result, ela_result, cnn_score)

    result["verdict"]      = hybrid["verdict"]
    result["hybrid_score"] = hybrid["hybrid_score"]
    result["confidence"]   = hybrid["confidence"]
    result["layer_scores"] = hybrid["layer_scores"]
    result["weights"]      = hybrid["weights"]
    result["success"]      = True

    # ── Özet ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  KARAR        : {result['verdict']}  ({result['confidence']} güven)")
    print(f"  Hybrid Skor  : {result['hybrid_score']:.4f}  (eşik: {VERDICT_THRESHOLD})")
    print(f"  Katman Skoru : meta={result['layer_scores']['metadata']:.2f}  "
          f"ela={result['layer_scores']['ela']:.2f}  "
          f"cnn={result['layer_scores']['cnn']:.2f}")
    print("=" * 60 + "\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  HybridID — Hibrit Analiz Aracı (Sprint 6)")
    print("=" * 60)

    _model = load_cnn_model()

    while True:
        user_input = input("\nDosya yolunu girin (Çıkış: 'q'): ").strip().strip("\"'")

        if user_input.lower() in ("q", "exit", "quit"):
            print("Program kapatılıyor...")
            break

        if not user_input:
            continue

        r = run_full_analysis(user_input, model=_model)

        if r["success"]:
            print(f"[TAMAMLANDI] {r['verdict']} | Skor: {r['hybrid_score']:.4f} "
                  f"| Güven: {r['confidence']}")
        else:
            print(f"[HATA] {r['error_message']}")
