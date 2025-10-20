import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json

class SkinDiseasePreprocessor:
    """Cilt hastalığı görüntülerini ön işleme sınıfı"""
    
    def __init__(self, source_dir, output_dir, img_size=224, validation_split=0.2):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.validation_split = validation_split
        self.stats = {}
        
    def analyze_dataset(self):
        """Veri setini analiz et"""
        print("=" * 70)
        print("VERİ SETİ ANALİZİ")
        print("=" * 70)
        
        classes = sorted([d.name for d in self.source_dir.iterdir() if d.is_dir()])
        
        total_images = 0
        class_distribution = {}
        
        for class_name in classes:
            class_path = self.source_dir / class_name
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            images = []
            for ext in extensions:
                images.extend(list(class_path.glob(ext)))
            
            count = len(images)
            class_distribution[class_name] = count
            total_images += count
            
            print(f"  📁 {class_name:20s} : {count:5d} görüntü")
        
        print("-" * 70)
        print(f"  📊 Toplam Sınıf        : {len(classes)}")
        print(f"  📊 Toplam Görüntü      : {total_images}")
        print(f"  📊 Ortalama/Sınıf      : {total_images//len(classes)}")
        print(f"  📊 Min Görüntü         : {min(class_distribution.values())} ({min(class_distribution, key=class_distribution.get)})")
        print(f"  📊 Max Görüntü         : {max(class_distribution.values())} ({max(class_distribution, key=class_distribution.get)})")
        print("=" * 70)
        
        # Dağılımı görselleştir
        self.plot_class_distribution(class_distribution)
        
        self.stats['classes'] = classes
        self.stats['distribution'] = class_distribution
        self.stats['total'] = total_images
        
        return class_distribution
    
    def plot_class_distribution(self, distribution):
        """Sınıf dağılımını görselleştir"""
        plt.figure(figsize=(12, 6))
        classes = list(distribution.keys())
        counts = list(distribution.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        bars = plt.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
        
        # Bar üzerine sayıları yaz
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.xlabel('Hastalık Sınıfı', fontsize=12, fontweight='bold')
        plt.ylabel('Görüntü Sayısı', fontsize=12, fontweight='bold')
        plt.title('Sınıf Dağılımı', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
        print("\n✅ Sınıf dağılım grafiği kaydedildi: class_distribution.png")
        plt.close()
    
    def check_image_quality(self, img_path):
        """Görüntü kalitesini kontrol et"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return False, "Okunamadı"
            
            # Boyut kontrolü
            h, w = img.shape[:2]
            if h < 50 or w < 50:
                return False, "Çok küçük"
            
            # Boş görüntü kontrolü
            if np.mean(img) < 5 or np.mean(img) > 250:
                return False, "Çok karanlık/aydınlık"
            
            return True, "OK"
        except Exception as e:
            return False, str(e)
    
    def remove_hair(self, image):
        """Kıl giderme algoritması (opsiyonel)"""
        # Gray scale dönüşüm
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Black hat morfolojik işlem
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold
        _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Inpaint ile kılları kaldır
        result = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
        
        return result
    
    def enhance_contrast(self, image):
        """Kontrast iyileştirme - CLAHE"""
        # LAB color space'e dönüştür
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE uygula (sadece L kanalına)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Geri birleştir
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def preprocess_image(self, img_path, remove_hair_flag=False):
        """Tek bir görüntüyü ön işle"""
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        # 1. Kıl giderme (opsiyonel)
        if remove_hair_flag:
            img = self.remove_hair(img)
        
        # 2. Kontrast iyileştirme
        img = self.enhance_contrast(img)
        
        # 3. Resize (aspect ratio korunarak)
        h, w = img.shape[:2]
        if h > w:
            new_h = self.img_size
            new_w = int(w * (self.img_size / h))
        else:
            new_w = self.img_size
            new_h = int(h * (self.img_size / w))
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 4. Padding (kare yapmak için)
        delta_w = self.img_size - new_w
        delta_h = self.img_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # 5. Gürültü azaltma
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        return img
    
    def create_splits(self, class_name, images, train_dir, val_dir):
        """Train ve validation split oluştur"""
        # Karıştır ve böl
        train_imgs, val_imgs = train_test_split(
            images, 
            test_size=self.validation_split, 
            random_state=42,
            shuffle=True
        )
        
        # Train klasörü
        train_class_dir = train_dir / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation klasörü
        val_class_dir = val_dir / class_name
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        return train_imgs, val_imgs, train_class_dir, val_class_dir
    
    def process_all(self, remove_hair=False, show_samples=True):
        """Tüm veri setini işle"""
        print("\n" + "=" * 70)
        print("VERİ ÖN İŞLEME BAŞLATILIYOR")
        print("=" * 70)
        
        # Çıktı klasörlerini oluştur
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        
        # Varsa temizle
        if self.output_dir.exists():
            response = input(f"\n⚠️  '{self.output_dir}' klasörü mevcut. Silinsin mi? (e/h): ")
            if response.lower() == 'e':
                shutil.rmtree(self.output_dir)
                print("✅ Eski klasör silindi.")
            else:
                print("❌ İşlem iptal edildi.")
                return
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # İstatistikler
        total_processed = 0
        total_failed = 0
        class_stats = {}
        
        classes = sorted([d.name for d in self.source_dir.iterdir() if d.is_dir()])
        
        for class_name in classes:
            print(f"\n{'='*70}")
            print(f"İşleniyor: {class_name}")
            print(f"{'='*70}")
            
            class_path = self.source_dir / class_name
            
            # Tüm görüntüleri al
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            images = []
            for ext in extensions:
                images.extend(list(class_path.glob(ext)))
            
            # Kalite kontrolü
            print(f"  🔍 Kalite kontrolü yapılıyor...")
            valid_images = []
            failed_images = []
            
            for img_path in tqdm(images, desc="  Kontrol", ncols=70):
                is_valid, reason = self.check_image_quality(img_path)
                if is_valid:
                    valid_images.append(img_path)
                else:
                    failed_images.append((img_path, reason))
            
            print(f"  ✅ Geçerli: {len(valid_images)}")
            print(f"  ❌ Geçersiz: {len(failed_images)}")
            
            if failed_images:
                print(f"  ⚠️  İlk 5 hatalı görüntü:")
                for img_path, reason in failed_images[:5]:
                    print(f"      - {img_path.name}: {reason}")
            
            # Train/Val split
            train_imgs, val_imgs, train_class_dir, val_class_dir = \
                self.create_splits(class_name, valid_images, train_dir, val_dir)
            
            # Train görüntülerini işle
            train_success = 0
            print(f"\n  📦 Train seti işleniyor...")
            for img_path in tqdm(train_imgs, desc="  Train", ncols=70):
                processed = self.preprocess_image(img_path, remove_hair)
                if processed is not None:
                    output_path = train_class_dir / img_path.name
                    cv2.imwrite(str(output_path), processed)
                    train_success += 1
                else:
                    total_failed += 1
            
            # Validation görüntülerini işle
            val_success = 0
            print(f"  📦 Validation seti işleniyor...")
            for img_path in tqdm(val_imgs, desc="  Val  ", ncols=70):
                processed = self.preprocess_image(img_path, remove_hair)
                if processed is not None:
                    output_path = val_class_dir / img_path.name
                    cv2.imwrite(str(output_path), processed)
                    val_success += 1
                else:
                    total_failed += 1
            
            class_stats[class_name] = {
                'total': len(images),
                'valid': len(valid_images),
                'invalid': len(failed_images),
                'train': train_success,
                'val': val_success
            }
            
            total_processed += (train_success + val_success)
            
            print(f"  ✅ Train: {train_success} | Val: {val_success}")
        
        # Özet rapor
        self.print_summary(class_stats, total_processed, total_failed)
        
        # İstatistikleri kaydet
        self.save_stats(class_stats)
        
        # Örnek görüntüler göster
        if show_samples:
            self.show_samples(train_dir, classes)
    
    def print_summary(self, class_stats, total_processed, total_failed):
        """Özet rapor yazdır"""
        print("\n" + "=" * 70)
        print("İŞLEM RAPORU")
        print("=" * 70)
        
        print(f"\n{'Sınıf':<20} {'Toplam':<10} {'Geçerli':<10} {'Train':<10} {'Val':<10}")
        print("-" * 70)
        
        for class_name, stats in class_stats.items():
            print(f"{class_name:<20} {stats['total']:<10} {stats['valid']:<10} "
                  f"{stats['train']:<10} {stats['val']:<10}")
        
        print("=" * 70)
        print(f"✅ Başarıyla İşlenen  : {total_processed}")
        print(f"❌ Başarısız         : {total_failed}")
        print(f"📁 Çıktı Klasörü     : {self.output_dir}")
        print("=" * 70)
    
    def save_stats(self, class_stats):
        """İstatistikleri JSON olarak kaydet"""
        stats_path = self.output_dir / 'preprocessing_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(class_stats, f, indent=2, ensure_ascii=False)
        print(f"\n💾 İstatistikler kaydedildi: {stats_path}")
    
    def show_samples(self, train_dir, classes, samples_per_class=3):
        """Her sınıftan örnek görüntüler göster"""
        print(f"\n📸 Örnek görüntüler gösteriliyor...")
        
        fig, axes = plt.subplots(len(classes), samples_per_class, 
                                figsize=(15, 3*len(classes)))
        
        if len(classes) == 1:
            axes = axes.reshape(1, -1)
        
        for i, class_name in enumerate(classes):
            class_dir = train_dir / class_name
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            for j in range(min(samples_per_class, len(images))):
                img = cv2.imread(str(images[j]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                
                if j == 0:
                    axes[i, j].set_title(f"{class_name}", 
                                        fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_images.png', dpi=150, bbox_inches='tight')
        print(f"✅ Örnek görüntüler kaydedildi: {self.output_dir / 'sample_images.png'}")
        plt.close()

# =============================================================================
# KULLANIM
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║          CİLT HASTALIĞI VERİ ÖN İŞLEME SİSTEMİ                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    Bu script şunları yapar:
    ✅ Görüntü kalite kontrolü
    ✅ Kontrast iyileştirme (CLAHE)
    ✅ Kıl giderme (opsiyonel)
    ✅ Gürültü azaltma
    ✅ Standart boyutlandırma
    ✅ Train/Val split (%80/%20)
    ✅ Detaylı istatistik raporlama
    """)
    
    # Parametreler
    SOURCE_DIR = '/content/dataset/IMG_CLASSES'
    OUTPUT_DIR = '/content/dataset/processed'
    IMG_SIZE = 224
    VALIDATION_SPLIT = 0.2
    REMOVE_HAIR = False  # True yaparsanız kıl giderme aktif olur
    
    # Preprocessor oluştur
    preprocessor = SkinDiseasePreprocessor(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        img_size=IMG_SIZE,
        validation_split=VALIDATION_SPLIT
    )
    
    # Veri setini analiz et
    preprocessor.analyze_dataset()
    
    # Kullanıcıya sor
    print("\n" + "="*70)
    response = input("Ön işleme başlatılsın mı? (e/h): ")
    
    if response.lower() == 'e':
        # Kıl giderme seçeneği
        hair_response = input("Kıl giderme algoritması kullanılsın mı? (e/h): ")
        REMOVE_HAIR = (hair_response.lower() == 'e')
        
        # İşleme başlat
        preprocessor.process_all(remove_hair=REMOVE_HAIR, show_samples=True)
        
        print("\n🎉 Ön işleme tamamlandı!")
        print(f"\n📁 İşlenmiş veriler: {OUTPUT_DIR}")
        print("   ├── train/")
        print("   └── val/")
        print("\nArtık model eğitimine başlayabilirsiniz:")
        print(f"   model, class_names = train_model('{OUTPUT_DIR}/train')")
    else:
        print("❌ İşlem iptal edildi.")