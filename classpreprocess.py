import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 224
BATCH_PROCESS_SIZE = 50  # ✅ EKLENDI: Her 50 resimdde belleği temizle
# Use relative paths from project root
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'CLASSIFIC')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'processedCLASSIFIC')

class SkinDiseasePreprocessor:
    def __init__(self, dataset_path, img_size=224):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.class_names = []
        
    def preprocess_image(self, img):
        """Apply preprocessing steps to a single image"""
        try:
            # Check if image is valid
            if img is None or img.size == 0:
                return None
            
            # Resize image to 224x224
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            
            # Convert to RGB if needed
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            try:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.CLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except:
                pass  # Skip CLAHE if it fails
            
            # Light denoising (faster than full denoising)
            try:
                img = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
            except:
                pass  # Skip denoising if it fails
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None
    
    def load_and_preprocess_images(self):
        """Load images from dataset and preprocess them in batches"""
        images = []
        labels = []
        
        # Get all class folders
        try:
            class_folders = sorted([f for f in os.listdir(self.dataset_path) 
                                  if os.path.isdir(os.path.join(self.dataset_path, f))])
        except Exception as e:
            print(f"Error reading dataset path: {e}")
            print(f"Please check if the path exists: {self.dataset_path}")
            return None, None
        
        if len(class_folders) == 0:
            print(f"No class folders found in {self.dataset_path}")
            return None, None
        
        self.class_names = class_folders
        print(f"\n✓ Found {len(class_folders)} classes")
        print("="*60)
        
        total_images = 0
        failed_images = 0
        
        for class_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(self.dataset_path, class_name)
            print(f"\n[{class_idx + 1}/{len(class_folders)}] Processing: {class_name}")
            
            try:
                image_files = [f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            except Exception as e:
                print(f"  ⚠️  Error reading folder: {e}")
                continue
            
            if len(image_files) == 0:
                print(f"  ⚠️  No images found in {class_name}")
                continue
            
            print(f"  Found {len(image_files)} images")
            
            # Process images with progress bar
            class_images = []
            class_labels = []
            
            for img_file in tqdm(image_files, desc="  Progress", leave=False):
                try:
                    img_path = os.path.join(class_path, img_file)
                    
                    # Read image
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        failed_images += 1
                        continue
                    
                    # Preprocess image
                    processed_img = self.preprocess_image(img)
                    
                    if processed_img is not None:
                        class_images.append(processed_img)
                        class_labels.append(class_idx)
                        total_images += 1
                    else:
                        failed_images += 1
                    
                    # Clear memory periodically
                    if len(class_images) % BATCH_PROCESS_SIZE == 0:
                        images.extend(class_images)
                        labels.extend(class_labels)
                        class_images = []
                        class_labels = []
                        gc.collect()
                        
                except Exception as e:
                    failed_images += 1
                    continue
            
            # Add remaining images
            if class_images:
                images.extend(class_images)
                labels.extend(class_labels)
            
            print(f"  ✓ Successfully processed: {len([l for l in labels if l == class_idx])} images")
            
            # Clear memory
            gc.collect()
        
        print("\n" + "="*60)
        print(f"✓ Total images processed: {total_images}")
        if failed_images > 0:
            print(f"⚠️  Failed images: {failed_images}")
        print("="*60)
        
        if total_images == 0:
            print("\n❌ No images were successfully processed!")
            return None, None
        
        # Convert to numpy arrays
        print("\nConverting to numpy arrays...")
        try:
            images = np.array(images, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)
        except MemoryError:
            print("❌ Memory error! Try reducing image size or processing fewer images.")
            return None, None
        
        return images, labels
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data to disk as folder structure (for train.py compatibility)"""
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        print("\n💾 Saving processed data...")
        
        try:
            # Create class folders
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = os.path.join(OUTPUT_PATH, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Save training images
                train_indices = np.where(y_train == class_idx)[0]
                for i, idx in enumerate(train_indices):
                    img = (X_train[idx] * 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(class_dir, f'train_{i:05d}.jpg'), img_bgr)
                
                # Save test images
                test_indices = np.where(y_test == class_idx)[0]
                for i, idx in enumerate(test_indices):
                    img = (X_test[idx] * 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(class_dir, f'test_{i:05d}.jpg'), img_bgr)
                
                print(f"  ✓ Saved {class_name}: {len(train_indices)} train + {len(test_indices)} test images")
            
            # Save class names for reference
            class_names_file = os.path.join(OUTPUT_PATH, 'class_names.txt')
            with open(class_names_file, 'w', encoding='utf-8') as f:
                for name in self.class_names:
                    f.write(f"{name}\n")
            print(f"  ✓ Saved class_names.txt")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


def print_dataset_info(X_train, X_test, y_train, y_test, class_names):
    """Print comprehensive dataset information"""
    print("\n" + "="*60)
    print("📊 DATASET INFORMATION")
    print("="*60)
    
    print(f"\n🔢 Dataset Split:")
    print(f"   Training samples:   {X_train.shape[0]:,}")
    print(f"   Test samples:       {X_test.shape[0]:,}")
    print(f"   Total samples:      {X_train.shape[0] + X_test.shape[0]:,}")
    
    print(f"\n📐 Data Shape:")
    print(f"   Image shape:        {X_train.shape[1:]}")
    print(f"   Image size:         {X_train.shape[1]}x{X_train.shape[2]} pixels")
    print(f"   Channels:           {X_train.shape[3]}")
    
    print(f"\n🏷️  Classes:")
    print(f"   Number of classes:  {len(class_names)}")
    
    print("\n📈 Class Distribution:")
    print("-" * 60)
    print(f"{'Class Name':<40} {'Train':<8} {'Test':<8} {'Total':<8}")
    print("-" * 60)
    
    for idx, class_name in enumerate(class_names):
        train_count = np.sum(y_train == idx)
        test_count = np.sum(y_test == idx)
        total_count = train_count + test_count
        print(f"{class_name[:38]:<40} {train_count:<8} {test_count:<8} {total_count:<8}")
    
    print("-" * 60)
    
    print(f"\n💾 Memory Usage:")
    print(f"   Training data:      {X_train.nbytes / (1024**2):.2f} MB")
    print(f"   Test data:          {X_test.nbytes / (1024**2):.2f} MB")
    print(f"   Total:              {(X_train.nbytes + X_test.nbytes) / (1024**2):.2f} MB")
    
    print("\n" + "="*60)


def main():
    print("="*60)
    print("🔬 SKIN DISEASE DATASET PREPROCESSING")
    print("="*60)
    
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ Error: Dataset path not found!")
        print(f"   Expected path: {DATASET_PATH}")
        print(f"   Current directory: {os.getcwd()}")
        print("\nPlease check:")
        print("   1. The dataset folder exists")
        print("   2. The path in the script is correct")
        return
    
    # Initialize preprocessor
    print(f"\n⚙️  Configuration:")
    print(f"   Image size:         {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Dataset path:       {DATASET_PATH}")
    print(f"   Output path:        {OUTPUT_PATH}")
    print(f"   Batch size:         {BATCH_PROCESS_SIZE}")
    
    preprocessor = SkinDiseasePreprocessor(DATASET_PATH, IMG_SIZE)
    
    # Load and preprocess images
    print("\n" + "="*60)
    print("🖼️  LOADING AND PREPROCESSING IMAGES")
    print("="*60)
    
    X, y = preprocessor.load_and_preprocess_images()
    
    if X is None or y is None:
        print("\n❌ Preprocessing failed!")
        return
    
    if len(X) == 0:
        print("\n❌ No images were processed!")
        return
    
    # Split into train and test sets
    print("\n📊 Splitting dataset into train/test...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        print("  ✓ Dataset split complete (80% train, 20% test)")
    except Exception as e:
        print(f"❌ Error splitting dataset: {e}")
        return
    
    # Print dataset information
    print_dataset_info(X_train, X_test, y_train, y_test, preprocessor.class_names)
    
    # Save processed data
    success = preprocessor.save_processed_data(X_train, X_test, y_train, y_test)
    
    if success:
        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETE!")
        print("="*60)
        print(f"\n📁 Processed data saved to: {OUTPUT_PATH}/")
        print("\nFolder structure created:")
        for class_name in preprocessor.class_names:
            print(f"   ✓ {class_name}/")
        print("\n🚀 You can now run train.py to train the model!")
        print("="*60)
    else:
        print("\n❌ Failed to save processed data!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()