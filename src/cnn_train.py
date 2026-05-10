"""
HybridID - S5: CNN Model Eğitimi (ResNet50)
=============================================
Görevler:
  - S5/P0: Model seçimi (ResNet50) ve mimari kurulumu
  - S5/P0: Modelin eğitilmesi ve başarı metriklerinin ölçümü
  - S5/P1: Hiperparametre tuning işlemleri

Kullanım:
  python src/cnn_train.py

Çıktılar:
  models/resnet50_hybridid.h5     -> Eğitilmiş model (en iyi checkpoint)
  models/training_history.json    -> Epoch bazlı eğitim geçmişi
  models/evaluation_report.json   -> Test seti sonuçları (accuracy, precision, recall, F1, AUC)
  models/confusion_matrix.png     -> Confusion matrix görseli
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # GUI olmayan ortamlar için

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

# ─────────────────────────────────────────────
# 0. YAPILANDIRMA (Hiperparametreler)
# ─────────────────────────────────────────────

CONFIG = {
    # Veri
    "dataset_dir": "/Users/ufuk/Desktop/processed",   # train / val / test alt klasörleri buradan
    "img_size": (224, 224),
    "batch_size": 32,
    "num_classes": 2,
    "class_names": ["fake", "real"],   # keras flow_from_directory alfabetik sırayla alır

    # Mimari
    "base_model": "ResNet50",
    "dropout_rate": 0.5,
    "l2_lambda": 1e-4,
    "dense_units": 256,

    # Eğitim - Faz 1 (sadece head katmanları)
    "phase1_epochs": 10,
    "phase1_lr": 1e-3,

    # Eğitim - Faz 2 (fine-tuning: son bloklar açılır)
    "phase2_epochs": 30,
    "phase2_lr": 1e-5,
    "unfreeze_from_layer": 143,       # ResNet50'nin son "conv5_block" başlangıcı

    # Erken durdurma
    "patience": 8,

    # Çıktı
    "model_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
    "model_name": "resnet50_hybridid.h5",
}

os.makedirs(CONFIG["model_dir"], exist_ok=True)

print("=" * 60)
print("  HybridID — S5 CNN Eğitim Süreci Başlatıldı")
print("=" * 60)
print(f"TensorFlow sürümü : {tf.__version__}")
print(f"GPU kullanılabilir : {tf.config.list_physical_devices('GPU')}")


# ─────────────────────────────────────────────
# 1. VERİ YÜKLEYİCİLER  (ImageDataGenerator)
# ─────────────────────────────────────────────

def build_generators(cfg):
    """
    Train, val ve test için keras ImageDataGenerator oluşturur.
    Train setinde agresif augmentation uygulanır.
    ResNet50 ön işleme (preprocess_input) entegre edilmiştir.
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input

    # --- Train: augmentation ---
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.85, 1.15],
        shear_range=5,
    )

    # --- Val & Test: sadece normalize ---
    eval_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(cfg["dataset_dir"], "train"),
        target_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        class_mode="binary",
        shuffle=True,
        seed=42,
    )

    val_gen = eval_datagen.flow_from_directory(
        os.path.join(cfg["dataset_dir"], "val"),
        target_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        class_mode="binary",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        os.path.join(cfg["dataset_dir"], "test"),
        target_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        class_mode="binary",
        shuffle=False,
    )

    print(f"\n[Veri] Sınıf indeksleri  : {train_gen.class_indices}")
    print(f"[Veri] Train örnekleri  : {train_gen.samples}")
    print(f"[Veri] Val örnekleri    : {val_gen.samples}")
    print(f"[Veri] Test örnekleri   : {test_gen.samples}\n")

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────
# 2. MODEL MİMARİSİ  (ResNet50 + Custom Head)
# ─────────────────────────────────────────────

def build_model(cfg):
    """
    Transfer learning: ImageNet ağırlıklı ResNet50 + özel sınıflandırıcı katmanı.

    Mimari:
        ResNet50 (base, dondurulmuş)
        → GlobalAveragePooling2D
        → BatchNormalization
        → Dense(256, relu, L2)
        → Dropout(0.5)
        → Dense(1, sigmoid)          ← binary classification (fake/real)
    """
    # --- Base model ---
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*cfg["img_size"], 3),
    )
    base.trainable = False   # Faz-1: dondurulmuş

    # --- Custom head ---
    inputs = keras.Input(shape=(*cfg["img_size"], 3), name="input_image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dense(
        cfg["dense_units"],
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg["l2_lambda"]),
        name="dense_head",
    )(x)
    x = layers.Dropout(cfg["dropout_rate"], name="dropout_head")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="HybridID_ResNet50")

    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"[Model] Toplam parametre      : {total_params:,}")
    print(f"[Model] Eğitilebilir parametre: {trainable_params:,}")
    print(f"[Model] Dondurulmuş katman    : {base.name}\n")

    return model, base


# ─────────────────────────────────────────────
# 3. CALLBACK'LER
# ─────────────────────────────────────────────

def build_callbacks(cfg, phase: int):
    model_path = os.path.join(cfg["model_dir"], cfg["model_name"])

    return [
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=cfg["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ─────────────────────────────────────────────
# 4. EĞİTİM
# ─────────────────────────────────────────────

def train(cfg):
    train_gen, val_gen, test_gen = build_generators(cfg)
    model, base_model = build_model(cfg)

    history_all = {"phase1": {}, "phase2": {}}

    # ── FAZ 1: Sadece head eğitimi ──────────────────
    print("─" * 50)
    print("  FAZ 1: Head katmanı eğitimi (base dondurulmuş)")
    print("─" * 50)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["phase1_lr"]),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    t0 = time.time()
    h1 = model.fit(
        train_gen,
        epochs=cfg["phase1_epochs"],
        validation_data=val_gen,
        callbacks=build_callbacks(cfg, phase=1),
        verbose=1,
    )
    print(f"  Faz-1 süresi: {(time.time()-t0)/60:.1f} dakika\n")
    history_all["phase1"] = {k: [float(v) for v in vals] for k, vals in h1.history.items()}

    # ── FAZ 2: Fine-tuning (son ResNet50 blokları açılıyor) ─
    print("─" * 50)
    print(f"  FAZ 2: Fine-tuning (katman {cfg['unfreeze_from_layer']}+ açılıyor)")
    print("─" * 50)

    base_model.trainable = True
    for layer in base_model.layers[:cfg["unfreeze_from_layer"]]:
        layer.trainable = False

    trainable_now = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Açılan ResNet50 katmanı sayısı: {trainable_now}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["phase2_lr"]),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    t0 = time.time()
    h2 = model.fit(
        train_gen,
        epochs=cfg["phase2_epochs"],
        validation_data=val_gen,
        callbacks=build_callbacks(cfg, phase=2),
        verbose=1,
    )
    print(f"  Faz-2 süresi: {(time.time()-t0)/60:.1f} dakika\n")
    history_all["phase2"] = {k: [float(v) for v in vals] for k, vals in h2.history.items()}

    return model, test_gen, history_all


# ─────────────────────────────────────────────
# 5. DEĞERLENDİRME & METRİKLER
# ─────────────────────────────────────────────

def evaluate(model, test_gen, history_all, cfg):
    print("=" * 60)
    print("  TEST SETİ DEĞERLENDİRMESİ")
    print("=" * 60)

    # --- Tahminler ---
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = (y_prob > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    # --- Metrikler ---
    report = classification_report(y_true, y_pred, target_names=cfg["class_names"], output_dict=True)
    auc = roc_auc_score(y_true, y_prob)

    print("\nSınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=cfg["class_names"]))
    print(f"ROC-AUC Skoru : {auc:.4f}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg["class_names"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("HybridID — Confusion Matrix (Test Seti)", fontsize=13)
    plt.tight_layout()
    cm_path = os.path.join(cfg["model_dir"], "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[Grafik] Confusion matrix kaydedildi → {cm_path}")

    # --- Eğitim eğrileri ---
    _plot_history(history_all, cfg)

    # --- JSON raporu ---
    eval_report = {
        "test_accuracy"  : float(report["accuracy"]),
        "roc_auc"        : float(auc),
        "fake_precision" : float(report["fake"]["precision"]),
        "fake_recall"    : float(report["fake"]["recall"]),
        "fake_f1"        : float(report["fake"]["f1-score"]),
        "real_precision" : float(report["real"]["precision"]),
        "real_recall"    : float(report["real"]["recall"]),
        "real_f1"        : float(report["real"]["f1-score"]),
        "config"         : cfg,
    }

    report_path = os.path.join(cfg["model_dir"], "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)
    print(f"[Rapor] Değerlendirme raporu kaydedildi → {report_path}")

    history_path = os.path.join(cfg["model_dir"], "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_all, f, indent=2, ensure_ascii=False)
    print(f"[Rapor] Eğitim geçmişi kaydedildi → {history_path}")

    return eval_report


def _plot_history(history_all, cfg):
    """Faz 1 + Faz 2 eğitim eğrilerini birleştirip çizer."""
    def concat(key):
        p1 = history_all["phase1"].get(key, [])
        p2 = history_all["phase2"].get(key, [])
        return p1 + p2

    acc      = concat("accuracy")
    val_acc  = concat("val_accuracy")
    loss     = concat("loss")
    val_loss = concat("val_loss")
    auc      = concat("auc")
    val_auc  = concat("val_auc")

    epochs = range(1, len(acc) + 1)
    phase1_end = len(history_all["phase1"].get("accuracy", []))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("HybridID — Eğitim Eğrileri (Faz1 + Faz2)", fontsize=14)

    # --- Accuracy ---
    ax = axes[0]
    ax.plot(epochs, acc,     "b-o", markersize=3, label="Train Accuracy")
    ax.plot(epochs, val_acc, "r-o", markersize=3, label="Val Accuracy")
    if phase1_end:
        ax.axvline(phase1_end + 0.5, color="gray", linestyle="--", alpha=0.7, label="Faz 1 / 2 sınırı")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    # --- Loss ---
    ax = axes[1]
    ax.plot(epochs, loss,     "b-o", markersize=3, label="Train Loss")
    ax.plot(epochs, val_loss, "r-o", markersize=3, label="Val Loss")
    if phase1_end:
        ax.axvline(phase1_end + 0.5, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    # --- AUC ---
    ax = axes[2]
    ax.plot(epochs, auc,     "b-o", markersize=3, label="Train AUC")
    ax.plot(epochs, val_auc, "r-o", markersize=3, label="Val AUC")
    if phase1_end:
        ax.axvline(phase1_end + 0.5, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("ROC-AUC"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(cfg["model_dir"], "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Grafik] Eğitim eğrileri kaydedildi → {plot_path}")


# ─────────────────────────────────────────────
# 6. ANA AKIŞ
# ─────────────────────────────────────────────

if __name__ == "__main__":
    total_start = time.time()

    model, test_gen, history_all = train(CONFIG)
    eval_report = evaluate(model, test_gen, history_all, CONFIG)

    total_min = (time.time() - total_start) / 60
    print("\n" + "=" * 60)
    print(f"  EĞİTİM TAMAMLANDI  ({total_min:.1f} dakika)")
    print("=" * 60)
    print(f"  Test Accuracy : {eval_report['test_accuracy']:.4f}")
    print(f"  ROC-AUC       : {eval_report['roc_auc']:.4f}")
    print(f"  Model kayıt   : {os.path.join(CONFIG['model_dir'], CONFIG['model_name'])}")
    print("=" * 60)
