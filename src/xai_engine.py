"""
HybridID - S6/P1: XAI — Grad-CAM Entegrasyonu
===============================================
Görev:
  Eğitilmiş ResNet50 modelinin son konvolüsyon katmanı üzerinden
  Gradient-weighted Class Activation Map (Grad-CAM) üretir ve
  orijinal görsel ile süperpozisyon (overlay) oluşturur.

Referans: Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from DNNs"

Kullanım (standalone):
  python src/xai_engine.py

Import (Streamlit/pipeline):
  from xai_engine import save_gradcam
"""

import os
import sys
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

# src/ ve proje kökünü path'e ekle
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────────────────────
LAST_CONV_LAYER = "conv5_block3_out"   # ResNet50 son konvolüsyon katmanı
IMG_SIZE        = (224, 224)
ALPHA_HEATMAP   = 0.45                 # Süperpozisyon ısı haritası saydamlığı
OUTPUT_DIR      = os.path.join(_ROOT_DIR, "outputs", "gradcam")
MODEL_PATH      = os.path.join(_ROOT_DIR, "models", "resnet50_hybridid.h5")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Görsel Ön İşleme
# ─────────────────────────────────────────────────────────────────────────────
def _preprocess(image_path: str) -> tuple[np.ndarray, Image.Image]:
    """
    Görseli ResNet50 için hazırlar.

    Returns:
        arr          : (1, 224, 224, 3) float32 array (preprocess_input uygulanmış)
        original_pil : Orijinal PIL Image (overlay için)
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input

    original_pil = Image.open(image_path).convert("RGB")
    resized      = original_pil.resize(IMG_SIZE)
    arr          = np.array(resized, dtype=np.float32)
    arr          = preprocess_input(arr)
    return np.expand_dims(arr, axis=0), original_pil


# ─────────────────────────────────────────────────────────────────────────────
# 2. Grad-CAM Hesabı
# ─────────────────────────────────────────────────────────────────────────────
def _find_last_conv_layer(model):
    """
    Model hiyerarşisinde (iç içe Functional modeller dahil) son konvolüsyon
    katmanını bulur ve (katman, ara_model) çiftini döner.

    HybridID mimarisi:
        input_image → resnet50 (Functional) → gap → bn → dense → dropout → output

    Grad-CAM için resnet50 içindeki 'conv5_block3_out' katmanına erişmek gerekir.
    Bu katmanın çıktısını açığa çıkarmak için resnet50'nin alt modeli kullanılır.
    """
    import tensorflow as tf

    # Önce doğrudan dene
    try:
        layer = model.get_layer(LAST_CONV_LAYER)
        return layer, model
    except ValueError:
        pass

    # İç içe Functional (sub-model) içinde ara
    for layer in model.layers:
        if hasattr(layer, "layers"):           # sub-model (Functional)
            try:
                inner = layer.get_layer(LAST_CONV_LAYER)
                # Sub-model için çıktıyı içeren yeni bir model kur
                sub_model = tf.keras.Model(
                    inputs=layer.input,
                    outputs=[inner.output, layer.output],
                    name="grad_submodel"
                )
                return inner, sub_model
            except ValueError:
                # LAST_CONV_LAYER bu sub-model'de yok; son Conv2D'yi seç
                for inner in reversed(layer.layers):
                    if "conv" in inner.name.lower():
                        sub_model = tf.keras.Model(
                            inputs=layer.input,
                            outputs=[inner.output, layer.output],
                            name="grad_submodel_fallback"
                        )
                        print(f"Uyarı: '{LAST_CONV_LAYER}' bulunamadı, "
                              f"'{inner.name}' kullanılıyor.")
                        return inner, sub_model

    raise RuntimeError("Hiçbir konvolüsyon katmanı bulunamadı.")


def get_gradcam_heatmap(model, image_array: np.ndarray) -> np.ndarray:
    """
    GradientTape ile Grad-CAM ısı haritası üretir.

    İç içe sarmalı ResNet50 mimarisini destekler:
      HybridID model → resnet50 (sub-model) → conv5_block3_out

    Adımlar:
      1. Son conv katmanı ve sub-model bulunur.
      2. GradientTape ile son conv çıktısına göre gradyan hesaplanır.
      3. Gradyanlar kanallar bazında ortalanır (önem ağırlıkları).
      4. Feature map'ler ağırlıklı toplanır → ReLU → normalize.

    Args:
        model       : Yüklenmiş Keras modeli.
        image_array : (1, H, W, 3) preprocess_input uygulanmış array.

    Returns:
        heatmap: (H, W) float32 array, 0.0–1.0 arası.
    """
    import tensorflow as tf

    # ── ResNet50 sub-modelini ve son conv katmanını bul ───────────────────────
    resnet_sub = None
    for _layer in model.layers:
        if hasattr(_layer, "layers") and _layer.name == "resnet50":
            resnet_sub = _layer
            break

    if resnet_sub is not None:
        last_conv = None
        try:
            last_conv = resnet_sub.get_layer(LAST_CONV_LAYER)
        except ValueError:
            for lyr in reversed(resnet_sub.layers):
                if "conv" in lyr.name.lower():
                    last_conv = lyr
                    print(f"Uyarı: '{LAST_CONV_LAYER}' bulunamadı → '{lyr.name}'")
                    break
        if last_conv is None:
            raise RuntimeError("Konvolüsyon katmanı bulunamadı.")

        # Sub-model için: resnet_input → [conv_output, resnet_output]
        sub_grad_model = tf.keras.Model(
            inputs=resnet_sub.inputs,
            outputs=[last_conv.output, resnet_sub.output],
            name="sub_grad_model"
        )
        use_sub = True
    else:
        last_conv = None
        try:
            last_conv = model.get_layer(LAST_CONV_LAYER)
        except ValueError:
            for lyr in reversed(model.layers):
                if "conv" in lyr.name.lower():
                    last_conv = lyr
                    break
        sub_grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output],
            name="flat_grad_model"
        )
        use_sub = False

    # ── GradientTape: input'u izle, conv_outputs üzerinden gradyan al ────────
    inputs_cast = tf.cast(image_array, tf.float32)
    inputs_var  = tf.Variable(inputs_cast, trainable=True, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs_var)
        if use_sub:
            conv_outputs, _resnet_out = sub_grad_model(inputs_var, training=False)
            # Dış modeli tam zincir üzerinde çalıştır
            # Gap → BN → Dense → Output katmanlarını elle geç
            x = model.get_layer("gap")(conv_outputs)
            x = model.get_layer("bn_head")(x, training=False)
            x = model.get_layer("dense_head")(x)
            x = model.get_layer("dropout_head")(x, training=False)
            predictions = model.get_layer("output")(x)
        else:
            conv_outputs, predictions = sub_grad_model(inputs_var, training=False)
        loss = predictions[:, 0]

    # Gradyanları conv_outputs ile zincirle
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        # Son çare: inputs_var'a göre gradyan al, conv çıktısını yeniden hesapla
        with tf.GradientTape() as tape2:
            tape2.watch(inputs_var)
            if use_sub:
                conv_outputs, _ = sub_grad_model(inputs_var, training=False)
            else:
                conv_outputs, _ = sub_grad_model(inputs_var, training=False)
            loss2 = tf.reduce_mean(conv_outputs)
        grads = tape2.gradient(loss2, conv_outputs)
        if grads is None:
            raise RuntimeError("Gradyan hesaplanamadı.")


    # Kanallar bazında global ortalama → önem ağırlıkları
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (filters,)

    # Feature map'leri ağırlıkla
    conv_outputs = conv_outputs[0]                         # (h, w, filters)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)                          # (h, w)

    # ReLU + normalize
    heatmap = tf.nn.relu(heatmap).numpy()
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap.astype(np.float32)



# ─────────────────────────────────────────────────────────────────────────────
# 3. Görsel Üzerine Bindirme (Overlay)
# ─────────────────────────────────────────────────────────────────────────────
def overlay_gradcam(original_image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    """
    Grad-CAM ısı haritasını orijinal görsel üzerine bindirir.

    Args:
        original_image : Orijinal PIL Image (herhangi boyut).
        heatmap        : (H, W) float32 Grad-CAM haritası.

    Returns:
        overlay_pil: Isı haritası bindirilmiş PIL Image.
    """
    # Haritayı orijinal görsel boyutuna büyüt
    orig_w, orig_h = original_image.size
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (orig_w, orig_h))

    # JET colormap uygula (BGR → RGB)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Orijinal görsel numpy array
    original_arr = np.array(original_image.convert("RGB"), dtype=np.float32)

    # Alpha blend: heatmap + original
    overlay = (ALPHA_HEATMAP * heatmap_colored.astype(np.float32)
               + (1 - ALPHA_HEATMAP) * original_arr)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Görsel Kaydetme
# ─────────────────────────────────────────────────────────────────────────────
def save_gradcam(image_path: str, model) -> str | None:
    """
    Grad-CAM hesaplar, 2 panelli figür oluşturur (orijinal + overlay) ve
    PNG olarak kaydeder.

    Args:
        image_path : Analiz edilecek görsel yolu.
        model      : Yüklenmiş Keras modeli.

    Returns:
        Kaydedilen PNG'nin mutlak yolu. Hata durumunda None.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_name   = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.abspath(os.path.join(OUTPUT_DIR, f"{base_name}_gradcam.png"))

    try:
        image_array, original_pil = _preprocess(image_path)
        heatmap                   = get_gradcam_heatmap(model, image_array)
        overlay_pil               = overlay_gradcam(original_pil, heatmap)

        # ── 3 panelli figür: Orijinal | Grad-CAM Haritası | Overlay ─────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("#1a1a2e")

        # Panel 1 — Orijinal
        axes[0].imshow(original_pil)
        axes[0].set_title("Orijinal Görsel", color="white", fontsize=13, fontweight="bold")
        axes[0].axis("off")

        # Panel 2 — Saf Grad-CAM haritası (büyütülmüş)
        orig_w, orig_h = original_pil.size
        heatmap_big = cv2.resize(np.uint8(255 * heatmap), (orig_w, orig_h))
        im = axes[1].imshow(heatmap_big, cmap="jet", vmin=0, vmax=255)
        axes[1].set_title("Grad-CAM Aktivasyon Haritası", color="white", fontsize=13, fontweight="bold")
        axes[1].axis("off")
        cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        cbar.set_label("Aktivasyon Seviyesi", color="white", fontsize=9)

        # Panel 3 — Overlay
        axes[2].imshow(overlay_pil)
        axes[2].set_title("Grad-CAM Süperpozisyon", color="white", fontsize=13, fontweight="bold")
        axes[2].axis("off")

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_edgecolor("#333366")

        plt.suptitle(
            f"Hybrid-ID Grad-CAM (XAI) — {os.path.basename(image_path)}",
            color="white", fontsize=15, fontweight="bold", y=1.01
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"[XAI] Grad-CAM kaydedildi → {output_path}")
        return output_path

    except Exception as e:
        print(f"Hata: Grad-CAM oluşturulamadı → {e}")
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tensorflow as tf
    from tensorflow import keras

    print("=" * 60)
    print("  HybridID — Grad-CAM XAI Motoru (Sprint 6/P1)")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model bulunamadı → {MODEL_PATH}")
        sys.exit(1)

    print(f"Model yükleniyor: {MODEL_PATH}")
    _model = keras.models.load_model(MODEL_PATH)
    print("Model hazır.\n")

    while True:
        user_input = input("Dosya yolunu girin (Çıkış: 'q'): ").strip().strip("\"'")

        if user_input.lower() in ("q", "exit", "quit"):
            print("Program kapatılıyor...")
            break

        if not user_input:
            continue

        path = save_gradcam(user_input, _model)
        if path:
            print(f"[OK] → {path}")
        else:
            print("[HATA] Grad-CAM oluşturulamadı.")
