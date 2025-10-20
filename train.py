import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

# 1. VERİ HAZIRLAMA
def prepare_dataset(processed_data_dir, original_data_dir, img_size=224, validation_split=0.2):
    """Dataset'i yükle ve train/val'e ayır"""
    processed_data_dir = Path(processed_data_dir)
    original_data_dir = Path(original_data_dir)

    # Sınıf isimlerini orijinal veri dizininden al
    class_names = sorted([d.name for d in original_data_dir.iterdir() if d.is_dir()])
    print(f"Bulunan sınıflar: {class_names}")

    # Her sınıftan örnek sayısını say (işlenmiş veriden)
    print("\nİşlenmiş veri seti dağılımı:")
    for class_name in class_names:
        train_path = processed_data_dir / 'train' / class_name
        val_path = processed_data_dir / 'val' / class_name
        train_count = len(list(train_path.glob('*.jpg')) + list(train_path.glob('*.png')) + list(train_path.glob('*.jpeg')))
        val_count = len(list(val_path.glob('*.jpg')) + list(val_path.glob('*.png')) + list(val_path.glob('*.jpeg')))
        print(f"  {class_name}: Train {train_count}, Val {val_count}")


    # Data augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # Train set
    train_generator = train_datagen.flow_from_directory(
        processed_data_dir / 'train',  # İşlenmiş verinin train klasörü
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    # Validation set
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # Val sette sadece rescale

    val_generator = val_datagen.flow_from_directory(
        processed_data_dir / 'val', # İşlenmiş verinin val klasörü
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )


    return train_generator, val_generator, class_names

# 2. MODEL OLUŞTURMA
def create_model(num_classes, img_size=224):
    """Global Average Pooling kullanan model (CAM için gerekli)"""
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )

    # İlk katmanları dondur
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)

    # CAM için Global Average Pooling kullan
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs)
    return model, base

# 3. GRAD-CAM İMPLEMENTASYONU
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Grad-CAM ısı haritası oluştur"""
    # Model için preprocessing
    # img_array already preprocessed by flow_from_directory (rescale)
    # keras.applications.efficientnet.preprocess_input(img_array)

    grad_model = keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

# 4. OPENCV İLE GÖRSELLEŞTIRME
def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Isı haritasını görüntü üzerine bindirin"""
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # Görüntüyü birleştir
    superimposed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    return superimposed, heatmap

def mark_disease_region(img, heatmap, threshold=0.6):
    """Hastalık bölgesini bbox ve kontur ile işaretleyin"""
    img_marked = img.copy()

    # Isı haritasını resize et
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Eşikleme uygula
    _, binary = cv2.threshold(
        (heatmap_resized * 255).astype(np.uint8),
        int(threshold * 255),
        255,
        cv2.THRESH_BINARY
    )

    # Morfolojik işlemler (gürültüyü azalt)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Konturları bul
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Konturları büyüklüğe göre sırala ve en büyük 3'ünü al
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    # Her konturu işaretle
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 100:  # Çok küçük alanları yoksay
            # Bounding box çiz
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_marked, (x, y), (x+w, y+h), (0, 255, 0), 3)

            # Konturu çiz
            cv2.drawContours(img_marked, [contour], -1, (255, 0, 0), 2)

            # Alan bilgisini yaz
            area_pct = (cv2.contourArea(contour) / (img.shape[0] * img.shape[1])) * 100
            cv2.putText(img_marked, f'{area_pct:.1f}%', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img_marked

# 5. TAM PIPELINE
def process_image(image_path, model, class_names, img_size=224):
    """Tam işleme pipeline'ı"""
    # Görüntüyü yükle
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Model için hazırla
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)

    # Tahmin yap (preprocessing burada yapılacak)
    img_preprocessed = keras.applications.efficientnet.preprocess_input(img_array.copy())
    predictions = model.predict(img_preprocessed, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = predictions[0][pred_class]

    # Top 3 tahmini al
    top3_idx = np.argsort(predictions[0])[-3:][::-1]
    top3_predictions = [(class_names[i], predictions[0][i]) for i in top3_idx]

    # Grad-CAM hesapla
    last_conv_layer = 'top_activation'
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_class)

    # Görselleştir
    superimposed, heatmap_colored = overlay_heatmap_on_image(img_rgb, heatmap)
    marked = mark_disease_region(img_rgb, heatmap, threshold=0.5)

    return {
        'original': img_rgb,
        'heatmap_overlay': superimposed,
        'marked': marked,
        'prediction': class_names[pred_class],
        'confidence': confidence,
        'top3': top3_predictions,
        'heatmap': heatmap
    }

# 6. GÖRSELLEŞTIRME
def visualize_results(results, save_path=None):
    """Sonuçları göster"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(results['original'])
    axes[0].set_title('Orijinal Görüntü', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(results['heatmap_overlay'])
    axes[1].set_title('Grad-CAM Isı Haritası', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Top 3 tahmini göster
    top3_text = '\n'.join([f"{name}: {conf:.1%}" for name, conf in results['top3']])
    axes[2].imshow(results['marked'])
    axes[2].set_title(f"İşaretlenmiş Bölge\n{results['prediction']} ({results['confidence']:.1%})\n\nTop 3:\n{top3_text}",
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sonuç kaydedildi: {save_path}")

    plt.show()

# 7. EĞİTİM FONKSİYONU
def train_model(processed_data_dir, original_data_dir, epochs=30, img_size=224):
    """Model eğitimi"""
    print("=" * 60)
    print("VERİ SETİ HAZIRLANIYOR...")
    print("=" * 60)

    # Veri setini hazırla
    train_gen, val_gen, class_names = prepare_dataset(processed_data_dir, original_data_dir, img_size)
    num_classes = len(class_names)

    print(f"\nToplam sınıf sayısı: {num_classes}")
    print(f"Train örnekleri: {train_gen.samples}")
    print(f"Validation örnekleri: {val_gen.samples}")

    # Model oluştur
    print("\n" + "=" * 60)
    print("MODEL OLUŞTURULUYOR...")
    print("=" * 60)
    model, base = create_model(num_classes, img_size)
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # İlk eğitim (frozen base)
    print("\n" + "=" * 60)
    print("PHASE 1: TRANSFER LEARNING (Frozen Base)")
    print("=" * 60)
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs // 2,
        callbacks=callbacks,
        verbose=1
    )

    # Fine-tuning
    print("\n" + "=" * 60)
    print("PHASE 2: FINE-TUNING (Unfrozen Base)")
    print("=" * 60)
    base.trainable = True

    # Son 50 katmanı unfreeze et
    for layer in base.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs // 2,
        callbacks=callbacks,
        verbose=1
    )

    # Son modeli kaydet
    model.save('skin_disease_model_final.h5')
    print(f"\nModel kaydedildi: skin_disease_model_final.h5")

    # Sınıf isimlerini kaydet
    with open('class_names.txt', 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Sınıf isimleri kaydedildi: class_names.txt")

    # Eğitim grafiklerini çiz
    plot_training_history(history1, history2)

    return model, class_names

# 8. EĞİTİM GRAFİKLERİ
def plot_training_history(history1, history2):
    """Eğitim geçmişini görselleştir"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history1.history['accuracy'], label='Train (Phase 1)')
    axes[0].plot(history1.history['val_accuracy'], label='Val (Phase 1)')
    axes[0].plot(range(len(history1.history['accuracy']),
                      len(history1.history['accuracy']) + len(history2.history['accuracy'])),
                history2.history['accuracy'], label='Train (Phase 2)')
    axes[0].plot(range(len(history1.history['val_accuracy']),
                      len(history1.history['val_accuracy']) + len(history2.history['val_accuracy'])),
                history2.history['val_accuracy'], label='Val (Phase 2)')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history1.history['loss'], label='Train (Phase 1)')
    axes[1].plot(history1.history['val_loss'], label='Val (Phase 1)')
    axes[1].plot(range(len(history1.history['loss']),
                      len(history1.history['loss']) + len(history2.history['loss'])),
                history2.history['loss'], label='Train (Phase 2)')
    axes[1].plot(range(len(history1.history['val_loss']),
                      len(history1.history['val_loss']) + len(history2.history['val_loss'])),
                history2.history['val_loss'], label='Val (Phase 2)')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()

# 9. TOPLU TEST
def batch_test(test_dir, model, class_names, num_images=5):
    """Birden fazla görüntüyü test et"""
    test_images = list(Path(test_dir).rglob('*.jpg')) + \
                  list(Path(test_dir).rglob('*.png')) + \
                  list(Path(test_dir).rglob('*.jpeg'))

    test_images = test_images[:num_images]

    for img_path in test_images:
        print(f"\n{'='*60}")
        print(f"Test: {img_path.name}")
        print(f"{'='*60}")

        results = process_image(img_path, model, class_names)
        visualize_results(results, save_path=f'result_{img_path.stem}.png')

# =============================================================================
# KULLANIM KILAVUZU
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║     CİLT HASTALIĞI SINIFLANDIRMA VE LOKALİZASYON SİSTEMİ         ║
    ╚═══════════════════════════════════════════════════════════════════╝

    KULLANIM ADIMLARI:

    1️⃣  MODEL EĞİTİMİ:
        model, class_names = train_model('/content/dataset/processed', '/content/dataset/IMG_CLASSES', epochs=30) # İşlenmiş ve orijinal veri klasörlerini kullan

    2️⃣  TEK GÖRÜNTÜ TESTİ:
        results = process_image('test.jpg', model, class_names)
        visualize_results(results)

    3️⃣  TOPLU TEST:
        batch_test('/content/dataset/processed/val', model, class_names, num_images=10) # İşlenmiş validation setini kullan

    4️⃣  MODELİ YÜKLEME (sonraki kullanımlar için):
        model = keras.models.load_model('skin_disease_model_final.h5')
        with open('class_names.txt', 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]

    5️⃣  SONUÇLARI KAYDETME:
        results = process_image('test.jpg', model, class_names)
        cv2.imwrite('marked_result.jpg', cv2.cvtColor(results['marked'], cv2.COLOR_RGB2BGR))
    """)

    # OTOMATIK ÇALIŞTIRMA
    SOURCE_DATA_DIR = '/content/dataset/IMG_CLASSES' # Orijinal veri klasörü
    PROCESSED_DATA_DIR = '/content/dataset/processed' # İşlenmiş veri klasörü

    if os.path.exists(PROCESSED_DATA_DIR):
        print(f"\n✅ İşlenmiş veri seti bulundu: {PROCESSED_DATA_DIR}")
        print("\n🚀 Eğitim başlatılıyor...\n")

        # Modeli eğit - İşlenmiş ve orijinal veri klasörlerini kullan
        model, class_names = train_model(PROCESSED_DATA_DIR, SOURCE_DATA_DIR, epochs=30)

        print("\n" + "="*60)
        print("✅ EĞİTİM TAMAMLANDI!")
        print("="*60)
        print(f"\nKaydedilen dosyalar:")
        print("  📁 skin_disease_model_final.h5")
        print("  📁 best_model.h5")
        print("  📁 class_names.txt")
        print("  📁 training_history.png")

    else:
        print(f"\n❌ İşlenmiş veri seti bulunamadı: {PROCESSED_DATA_DIR}")
        print("Lütfen ön işleme kodunun tamamlandığından ve 'processed' klasörünün oluştuğundan emin olun.")