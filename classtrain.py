import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import shutil

# Configuration
# Use relative paths from project root
DATA_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'processedCLASSIFIC')
MODEL_PATH = 'model'
IMG_SIZE = 224
BATCH_SIZE = 64  
EPOCHS = 100
LEARNING_RATE = 0.01  

class SkinDiseaseTrainer:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.class_names = None
        self.history = None
        
        os.makedirs(model_path, exist_ok=True)
    
    def prepare_train_val_split(self):
        """Train/Val klasörlerini ayır"""
        train_dir = os.path.join(self.data_path, 'train')
        val_dir = os.path.join(self.data_path, 'val')
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            print("✅ Using existing train/val directories")
            return train_dir, val_dir
        
        print("📂 Creating train/val split (80/20)...")
        
        all_items = os.listdir(self.data_path)
        class_folders = [
            d for d in all_items 
            if os.path.isdir(os.path.join(self.data_path, d)) 
            and d not in ['train', 'val']
        ]
        
        if len(class_folders) == 0:
            raise FileNotFoundError(f"No class folders found in {self.data_path}")
        
        print(f"✅ Found {len(class_folders)} class folders")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        for class_name in class_folders:
            class_path = os.path.join(self.data_path, class_name)
            
            image_files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))
            ]
            
            if len(image_files) == 0:
                print(f"⚠️  No images in {class_name}, skipping...")
                continue
            
            split_idx = int(len(image_files) * 0.8)
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]
            
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)
            
            for img_file in train_files:
                src = os.path.join(class_path, img_file)
                dst = os.path.join(train_class_dir, img_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            
            for img_file in val_files:
                src = os.path.join(class_path, img_file)
                dst = os.path.join(val_class_dir, img_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            
            print(f"  ✓ {class_name}: {len(train_files)} train, {len(val_files)} val")
        
        return train_dir, val_dir
    
    def load_data_generators(self):
        """Data generators yükle"""
        print("\n📁 Loading data from directories...")
        
        train_dir, val_dir = self.prepare_train_val_split()
        
        self.class_names = sorted([
            d for d in os.listdir(train_dir) 
            if os.path.isdir(os.path.join(train_dir, d))
        ])
        
        print(f"\n✅ Found {len(self.class_names)} classes:")
        for i, class_name in enumerate(self.class_names, 1):
            print(f"   {i}. {class_name}")
        
        # ✅ MobileNetV2 preprocessing
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.3,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            shear_range=0.15
        )
        
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.mobilenet_v2.preprocess_input
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"\n📊 Data Statistics:")
        print(f"   ✓ Training samples: {train_generator.samples}")
        print(f"   ✓ Validation samples: {val_generator.samples}")
        print(f"   ✓ Number of classes: {len(self.class_names)}")
        print(f"   ✓ Image size: {IMG_SIZE}x{IMG_SIZE}")
        print(f"   ✓ Batch size: {BATCH_SIZE}")
        
        return train_generator, val_generator
    
    def build_model(self, num_classes):
        """✅ MobileNetV2 ile güçlü model"""
        print("\n🏗️  Building MobileNetV2 model...")
        
        try:
            base_model = keras.applications.MobileNetV2(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
            print("   ✓ Loaded with ImageNet weights")
        except Exception as e:
            print(f"   ⚠️  Could not load ImageNet weights: {e}")
            base_model = keras.applications.MobileNetV2(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights=None
            )
        
        # ✅ İlk 100 layer freeze
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        
        x = base_model(inputs, training=True)
        x = layers.GlobalAveragePooling2D()(x)
        
        # ✅ Daha basit ama etkili head
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        return model, base_model
    
    def lr_schedule(self, epoch):
        """Learning rate schedule"""
        lr = LEARNING_RATE
        if epoch > 20:
            lr = 0.005
        if epoch > 40:
            lr = 0.001
        if epoch > 60:
            lr = 0.0005
        if epoch > 80:
            lr = 0.0001
        return lr
    
    def train(self, train_generator, val_generator):
        """Model eğit"""
        num_classes = len(self.class_names)
        
        self.model, base_model = self.build_model(num_classes)
        
        print(f"\n📊 Model Architecture:")
        print(f"   ✓ Total parameters: {self.model.count_params():,}")
        
        print("\n" + "="*70)
        print("🚀 TRAINING STARTED")
        print("="*70)
        
        # ✅ Daha agresif learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            ModelCheckpoint(
                os.path.join(self.model_path, 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,  # ✅ Daha uzun patience
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            LearningRateScheduler(self.lr_schedule)  # ✅ YENİ
        ]
        
        steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
        validation_steps = max(1, val_generator.samples // BATCH_SIZE)
        
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # Model kaydet
        final_path = os.path.join(self.model_path, 'skin_disease_model.keras')
        self.model.save(final_path)
        print(f"\n✅ Final model saved to: {final_path}")
        
        # Class names kaydet
        class_names_path = os.path.join(self.model_path, 'class_names.txt')
        with open(class_names_path, 'w', encoding='utf-8') as f:
            for name in self.class_names:
                f.write(f"{name}\n")
        print(f"✅ Class names saved to: {class_names_path}")
        
        self.history = history.history
        
        return self.history
    
    def plot_training_history(self):
        """Eğitim geçmişini çiz"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Training History - MobileNetV2', fontsize=16, fontweight='bold')
        
        # Accuracy plot
        axes[0].plot(self.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'training_history.png'), dpi=150, bbox_inches='tight')
        print(f"💾 Training history saved to {self.model_path}/training_history.png")
        plt.show()

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║     SKIN DISEASE CLASSIFICATION TRAINING v4.0 - MobileNetV2       ║
    ║              (Optimized for FAST & BETTER Convergence)            ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Model: MobileNetV2 (daha hızlı)")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Initial LR: {LEARNING_RATE}")
    print(f"   With LR Schedule: ✅")
    print(f"   Data path: {DATA_PATH}")
    
    trainer = SkinDiseaseTrainer(DATA_PATH, MODEL_PATH)
    
    try:
        train_gen, val_gen = trainer.load_data_generators()
        trainer.train(train_gen, val_gen)
        trainer.plot_training_history()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"📁 Models saved in: {MODEL_PATH}/")
        print(f"   ✓ best_model.keras")
        print(f"   ✓ skin_disease_model.keras")
        print(f"   ✓ class_names.txt")
        print(f"   ✓ training_history.png")
        print("\n🚀 Ready for testing with test.py!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()