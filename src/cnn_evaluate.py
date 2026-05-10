import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

import tensorflow as tf
from tensorflow import keras

# Import the configuration and generator builder from the existing training script
from cnn_train import CONFIG, build_generators

def evaluate_only():
    print("=" * 60)
    print("  HybridID — CNN Model Değerlendirme (Evaluation)")
    print("=" * 60)
    
    model_path = os.path.join(CONFIG["model_dir"], CONFIG["model_name"])
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı! Yol: {model_path}")
        return
        
    print(f"Model yükleniyor: {model_path}")
    # Load the model
    model = keras.models.load_model(model_path)
    print("Model başarıyla yüklendi.")
    
    # We only need the test_gen
    _, _, test_gen = build_generators(CONFIG)
    
    print("\nTest seti üzerinde tahminler yapılıyor...")
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = (y_prob > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    
    print("\nMetrikler hesaplanıyor...")
    report = classification_report(y_true, y_pred, target_names=CONFIG["class_names"], output_dict=True)
    auc = roc_auc_score(y_true, y_prob)

    print("\nSınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=CONFIG["class_names"]))
    print(f"ROC-AUC Skoru : {auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CONFIG["class_names"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("HybridID — Confusion Matrix (Test Seti)", fontsize=13)
    plt.tight_layout()
    cm_path = os.path.join(CONFIG["model_dir"], "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[Grafik] Confusion matrix kaydedildi → {cm_path}")

    # Evaluation Report JSON
    eval_report = {
        "test_accuracy"  : float(report["accuracy"]),
        "roc_auc"        : float(auc),
        "fake_precision" : float(report["fake"]["precision"]),
        "fake_recall"    : float(report["fake"]["recall"]),
        "fake_f1"        : float(report["fake"]["f1-score"]),
        "real_precision" : float(report["real"]["precision"]),
        "real_recall"    : float(report["real"]["recall"]),
        "real_f1"        : float(report["real"]["f1-score"]),
    }

    report_path = os.path.join(CONFIG["model_dir"], "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)
    print(f"[Rapor] Değerlendirme raporu kaydedildi → {report_path}")
    print("=" * 60)
    print("  DEĞERLENDİRME TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_only()
