import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import ResNet50
import keras_tuner as kt

# Import the existing configuration and generator builder
from cnn_train import CONFIG, build_generators

def build_tunable_model(hp):
    """
    KerasTuner için model mimarisini oluşturur.
    Faz-1 (Sadece Head Katmanı) için arama yapacağız.
    """
    # 1. Base model her zaman dondurulmuş (Frozen)
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*CONFIG["img_size"], 3),
    )
    base.trainable = False

    # 2. Aranacak Hiperparametreler (Search Space)
    hp_dense_units = hp.Choice('dense_units', values=[128, 256, 512])
    hp_dropout_rate = hp.Choice('dropout_rate', values=[0.3, 0.5, 0.7])
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])
    hp_l2_lambda = hp.Choice('l2_lambda', values=[1e-4, 1e-5])

    # 3. Mimarinin inşası
    inputs = keras.Input(shape=(*CONFIG["img_size"], 3), name="input_image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    
    x = layers.Dense(
        hp_dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(hp_l2_lambda),
        name="dense_head",
    )(x)
    
    x = layers.Dropout(hp_dropout_rate, name="dropout_head")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="HybridID_Tune_ResNet50")

    # 4. Derleme
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    
    return model

def tune_hyperparameters():
    print("=" * 60)
    print("  HybridID — S5 CNN Hyperparameter Tuning (KerasTuner)")
    print("=" * 60)
    
    # Veri yükleyicileri mevcut cnn_train modülünden alıyoruz
    train_gen, val_gen, _ = build_generators(CONFIG)
    
    # Ayarlayıcı (Tuner) kurulumu: RandomSearch
    tuner = kt.RandomSearch(
        build_tunable_model,
        objective='val_accuracy',
        max_trials=10,  # Rastgele 10 farklı kombinasyon denenecek
        executions_per_trial=1, # Her kombinasyon 1 kez eğitilecek
        directory=os.path.join(CONFIG["model_dir"], "tuning"),
        project_name='resnet50_head_tune'
    )
    
    tuner.search_space_summary()
    
    # Çok beklememek için val_loss iyileşmezse erkenden durdur
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    
    print("\n[Tuning] Arama başlatılıyor...\n")
    # Tuning işlemi Faz-1 üzerinden yürütüleceği için 10 epoch yeterlidir
    tuner.search(
        train_gen,
        validation_data=val_gen,
        epochs=10, 
        callbacks=[early_stop]
    )
    
    print("\n" + "=" * 60)
    print("  TUNING (ARAMA) İŞLEMİ TAMAMLANDI")
    print("=" * 60)
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"🌟 EN İYİ HİPERPARAMETRELER:")
    print(f"   - Dense Units  : {best_hps.get('dense_units')}")
    print(f"   - Dropout Rate : {best_hps.get('dropout_rate')}")
    print(f"   - Learning Rate: {best_hps.get('learning_rate')}")
    print(f"   - L2 Lambda    : {best_hps.get('l2_lambda')}")
    
    print("\nLütfen cnn_train.py dosyasındaki CONFIG kısmını bu değerlerle güncelleyin ve tam eğitimi tekrar başlatın.")

if __name__ == "__main__":
    tune_hyperparameters()
