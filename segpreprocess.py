"""
Segmentation Dataset Preprocessing
- Image resizing (640x640 → 320x320)
- CLAHE for lesion detail enhancement
- Denoising
- Hair removal
- Mask visualization
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

class SegmentationPreprocessor:
    def __init__(self, input_path="segdataset", output_path="segdataset_processed"):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.target_size = (320, 320)  # Yarı boyut
        self.splits = ['train', 'valid', 'test']
        
    def preprocess_all(self):
        """Complete preprocessing pipeline"""
        print("=" * 80)
        print("🔧 SEGMENTATION DATASET PREPROCESSING")
        print("=" * 80)
        print(f"\n📂 Input: {self.input_path}")
        print(f"📂 Output: {self.output_path}")
        print(f"📏 Target size: {self.target_size}")
        print()
        
        # Create output directories
        self.create_output_structure()
        
        # Process each split
        for split in self.splits:
            print(f"\n{'='*80}")
            print(f"Processing {split.upper()} split...")
            print(f"{'='*80}")
            self.process_split(split)
        
        # Show examples
        print(f"\n{'='*80}")
        print("📊 Creating visualization examples...")
        print(f"{'='*80}")
        self.visualize_examples()
        
        print("\n" + "=" * 80)
        print("✅ PREPROCESSING COMPLETE!")
        print("=" * 80)
    
    def create_output_structure(self):
        """Create output directory structure"""
        for split in self.splits:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        print("✅ Output directories created")
    
    def remove_hair(self, img):
        """Remove hair from skin images using morphological operations"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create kernel for hair detection (thin lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        
        # Black hat transform (detect dark hair on lighter background)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold to create binary mask of hair
        _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Inpaint to remove hair
        result = cv2.inpaint(img, hair_mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        
        return result
    
    def apply_clahe(self, img):
        """Apply CLAHE for contrast enhancement"""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge channels
        lab_clahe = cv2.merge([l_clahe, a, b])
        
        # Convert back to BGR
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return result
    
    def denoise(self, img):
        """Apply denoising"""
        # Non-local means denoising (preserves edges)
        result = cv2.fastNlMeansDenoisingColored(
            img, 
            None, 
            h=10,           # Filter strength
            hColor=10,      # Color component filter strength
            templateWindowSize=7, 
            searchWindowSize=21
        )
        return result
    
    def preprocess_image(self, img):
        """Apply all preprocessing steps to image"""
        # 1. Remove hair
        img = self.remove_hair(img)
        
        # 2. Denoise
        img = self.denoise(img)
        
        # 3. Apply CLAHE
        img = self.apply_clahe(img)
        
        # 4. Resize
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        
        return img
    
    def process_split(self, split):
        """Process one split (train/valid/test)"""
        input_images_path = self.input_path / split / 'images'
        input_labels_path = self.input_path / split / 'labels'
        output_images_path = self.output_path / split / 'images'
        output_labels_path = self.output_path / split / 'labels'
        
        # Get image files
        image_files = list(input_images_path.glob('*.*'))
        
        print(f"\n📸 Processing {len(image_files)} images...")
        
        processed_count = 0
        error_count = 0
        
        for img_path in tqdm(image_files, desc=f"   {split}"):
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"   ⚠️  Could not read: {img_path.name}")
                    error_count += 1
                    continue
                
                # Preprocess image
                processed_img = self.preprocess_image(img)
                
                # Save processed image
                output_img_path = output_images_path / img_path.name
                cv2.imwrite(str(output_img_path), processed_img)
                
                # Copy corresponding label file (no processing needed for labels in segmentation)
                label_path = input_labels_path / f"{img_path.stem}.txt"
                if label_path.exists():
                    output_label_path = output_labels_path / f"{img_path.stem}.txt"
                    shutil.copy(str(label_path), str(output_label_path))
                else:
                    print(f"   ⚠️  Label not found for: {img_path.name}")
                    error_count += 1
                    continue
                
                processed_count += 1
                
            except Exception as e:
                print(f"   ❌ Error processing {img_path.name}: {e}")
                error_count += 1
        
        print(f"\n   ✅ Successfully processed: {processed_count}/{len(image_files)}")
        if error_count > 0:
            print(f"   ⚠️  Errors: {error_count}")
    
    def visualize_examples(self):
        """Create visualization comparing original vs processed images with masks"""
        # Take 6 random samples from train
        train_images_path = self.input_path / 'train' / 'images'
        train_labels_path = self.input_path / 'train' / 'labels'
        processed_images_path = self.output_path / 'train' / 'images'
        
        image_files = list(train_images_path.glob('*.*'))[:6]
        
        fig, axes = plt.subplots(6, 4, figsize=(16, 24))
        fig.suptitle('Preprocessing Examples: Original → Processed → Original Mask → Processed Mask', 
                     fontsize=16, fontweight='bold')
        
        for idx, img_path in enumerate(image_files):
            # Read original image
            orig_img = cv2.imread(str(img_path))
            orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            # Read processed image
            processed_img_path = processed_images_path / img_path.name
            processed_img = cv2.imread(str(processed_img_path))
            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            
            # Read label and create mask
            label_path = train_labels_path / f"{img_path.stem}.txt"
            
            # Original mask (640x640)
            orig_mask = self.create_mask_from_label(label_path, (640, 640))
            
            # Processed mask (320x320)
            processed_mask = self.create_mask_from_label(label_path, self.target_size)
            
            # Plot
            axes[idx, 0].imshow(orig_img_rgb)
            axes[idx, 0].set_title(f'Original ({orig_img.shape[1]}x{orig_img.shape[0]})', fontsize=10)
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(processed_img_rgb)
            axes[idx, 1].set_title(f'Processed ({processed_img.shape[1]}x{processed_img.shape[0]})', fontsize=10)
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(orig_mask, cmap='gray')
            axes[idx, 2].set_title(f'Mask ({orig_mask.shape[1]}x{orig_mask.shape[0]})', fontsize=10)
            axes[idx, 2].axis('off')
            
            axes[idx, 3].imshow(processed_mask, cmap='gray')
            axes[idx, 3].set_title(f'Mask ({processed_mask.shape[1]}x{processed_mask.shape[0]})', fontsize=10)
            axes[idx, 3].axis('off')
        
        plt.tight_layout()
        output_viz_path = 'preprocessing_examples.png'
        plt.savefig(output_viz_path, dpi=200, bbox_inches='tight')
        print(f"\n✅ Visualization saved: {output_viz_path}")
        
        # Create detailed comparison for one image
        self.create_detailed_comparison(image_files[0])
    
    def create_mask_from_label(self, label_path, img_size):
        """Create binary mask from YOLO segmentation label"""
        mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        
        if not label_path.exists():
            return mask
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # Parse coordinates (normalized)
            coords = list(map(float, parts[1:]))
            
            # Check if it's polygon segmentation or bbox
            if len(coords) > 4:
                # Polygon segmentation
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * img_size[0])
                    y = int(coords[i+1] * img_size[1])
                    points.append([x, y])
                
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
            else:
                # Bounding box (x_center, y_center, width, height)
                x_center, y_center, width, height = coords[:4]
                
                # Convert to pixel coordinates
                x_center = int(x_center * img_size[0])
                y_center = int(y_center * img_size[1])
                w = int(width * img_size[0])
                h = int(height * img_size[1])
                
                # Draw rectangle
                x1 = int(x_center - w/2)
                y1 = int(y_center - h/2)
                x2 = int(x_center + w/2)
                y2 = int(y_center + h/2)
                
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        return mask
    
    def create_detailed_comparison(self, img_path):
        """Create detailed step-by-step preprocessing visualization"""
        # Read original
        orig_img = cv2.imread(str(img_path))
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Step by step
        step1 = self.remove_hair(orig_img)
        step1_rgb = cv2.cvtColor(step1, cv2.COLOR_BGR2RGB)
        
        step2 = self.denoise(step1)
        step2_rgb = cv2.cvtColor(step2, cv2.COLOR_BGR2RGB)
        
        step3 = self.apply_clahe(step2)
        step3_rgb = cv2.cvtColor(step3, cv2.COLOR_BGR2RGB)
        
        step4 = cv2.resize(step3, self.target_size, interpolation=cv2.INTER_AREA)
        step4_rgb = cv2.cvtColor(step4, cv2.COLOR_BGR2RGB)
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Detailed Preprocessing Pipeline', fontsize=16, fontweight='bold')
        
        axes[0, 0].imshow(orig_img_rgb)
        axes[0, 0].set_title('1. Original Image\n(640x640)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(step1_rgb)
        axes[0, 1].set_title('2. After Hair Removal', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(step2_rgb)
        axes[0, 2].set_title('3. After Denoising', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(step3_rgb)
        axes[1, 0].set_title('4. After CLAHE', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(step4_rgb)
        axes[1, 1].set_title('5. After Resize\n(320x320)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Mask comparison
        train_labels_path = self.input_path / 'train' / 'labels'
        label_path = train_labels_path / f"{img_path.stem}.txt"
        
        orig_mask = self.create_mask_from_label(label_path, (640, 640))
        processed_mask = self.create_mask_from_label(label_path, self.target_size)
        
        # Combine masks for comparison
        mask_comparison = np.hstack([
            cv2.resize(orig_mask, (320, 320)),
            processed_mask
        ])
        
        axes[1, 2].imshow(mask_comparison, cmap='gray')
        axes[1, 2].set_title('6. Masks: Original (left) vs Processed (right)', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('preprocessing_detailed.png', dpi=200, bbox_inches='tight')
        print(f"✅ Detailed visualization saved: preprocessing_detailed.png")


def main():
    preprocessor = SegmentationPreprocessor(
        input_path="segdataset",
        output_path="segdataset_processed"
    )
    preprocessor.preprocess_all()


if __name__ == "__main__":
    main()
