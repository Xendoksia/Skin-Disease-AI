"""
Segmentation Dataset Analysis Tool
Analyzes segdataset folder structure, images, and labels
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

class SegmentationDatasetAnalyzer:
    def __init__(self, dataset_path="segdataset"):
        self.dataset_path = Path(dataset_path)
        self.splits = ['train', 'valid', 'test']
        self.results = {}
        
    def analyze_all(self):
        """Run complete analysis"""
        print("=" * 80)
        print("🔍 SEGMENTATION DATASET ANALYSIS")
        print("=" * 80)
        
        # 1. Count files
        print("\n📊 1. FILE COUNT ANALYSIS")
        self.analyze_file_counts()
        
        # 2. Image dimensions
        print("\n📐 2. IMAGE DIMENSIONS ANALYSIS")
        self.analyze_image_dimensions()
        
        # 3. Label analysis
        print("\n🏷️  3. LABEL ANALYSIS")
        self.analyze_labels()
        
        # 4. Image properties
        print("\n🎨 4. IMAGE PROPERTIES ANALYSIS")
        self.analyze_image_properties()
        
        # 5. Matching analysis
        print("\n🔗 5. IMAGE-LABEL MATCHING")
        self.analyze_matching()
        
        # 6. Summary
        print("\n📝 6. SUMMARY")
        self.print_summary()
        
        # 7. Save report
        self.save_report()
        
        # 8. Create visualizations
        print("\n📊 7. CREATING VISUALIZATIONS")
        self.create_visualizations()
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 80)
    
    def analyze_file_counts(self):
        """Count files in each split"""
        self.results['file_counts'] = {}
        
        for split in self.splits:
            images_path = self.dataset_path / split / 'images'
            labels_path = self.dataset_path / split / 'labels'
            
            if not images_path.exists() or not labels_path.exists():
                print(f"   ⚠️  {split}: Not found!")
                continue
            
            num_images = len(list(images_path.glob('*.*')))
            num_labels = len(list(labels_path.glob('*.txt')))
            
            self.results['file_counts'][split] = {
                'images': num_images,
                'labels': num_labels,
                'matched': num_images == num_labels
            }
            
            status = "✅" if num_images == num_labels else "⚠️"
            print(f"   {status} {split.upper():<8} - Images: {num_images:>5} | Labels: {num_labels:>5}")
        
        total_images = sum(d['images'] for d in self.results['file_counts'].values())
        total_labels = sum(d['labels'] for d in self.results['file_counts'].values())
        print(f"\n   📦 TOTAL: {total_images} images, {total_labels} labels")
    
    def analyze_image_dimensions(self):
        """Analyze image dimensions"""
        self.results['dimensions'] = {}
        
        for split in self.splits:
            images_path = self.dataset_path / split / 'images'
            
            if not images_path.exists():
                continue
            
            print(f"\n   Analyzing {split} images...")
            
            dimensions = []
            aspect_ratios = []
            file_sizes = []
            
            image_files = list(images_path.glob('*.*'))[:500]  # Sample 500 for speed
            
            for img_path in tqdm(image_files, desc=f"   {split}", leave=False):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        dimensions.append((w, h))
                        aspect_ratios.append(w / h)
                        file_sizes.append(img_path.stat().st_size / 1024)  # KB
                except Exception as e:
                    print(f"   ⚠️  Error reading {img_path.name}: {e}")
            
            if dimensions:
                dim_counter = Counter(dimensions)
                most_common_dim = dim_counter.most_common(1)[0]
                
                widths = [d[0] for d in dimensions]
                heights = [d[1] for d in dimensions]
                
                self.results['dimensions'][split] = {
                    'count': len(dimensions),
                    'unique_dimensions': len(dim_counter),
                    'most_common': most_common_dim[0],
                    'most_common_count': most_common_dim[1],
                    'width_range': (min(widths), max(widths)),
                    'height_range': (min(heights), max(heights)),
                    'width_mean': np.mean(widths),
                    'height_mean': np.mean(heights),
                    'aspect_ratio_mean': np.mean(aspect_ratios),
                    'aspect_ratio_std': np.std(aspect_ratios),
                    'file_size_mean_kb': np.mean(file_sizes),
                    'file_size_range_kb': (min(file_sizes), max(file_sizes)),
                    'all_same': len(dim_counter) == 1
                }
                
                print(f"   {split.upper()}:")
                print(f"      • Unique dimensions: {len(dim_counter)}")
                print(f"      • Most common: {most_common_dim[0]} ({most_common_dim[1]} images)")
                print(f"      • Width range: {min(widths)} - {max(widths)} (avg: {np.mean(widths):.1f})")
                print(f"      • Height range: {min(heights)} - {max(heights)} (avg: {np.mean(heights):.1f})")
                print(f"      • Aspect ratio: {np.mean(aspect_ratios):.3f} ± {np.std(aspect_ratios):.3f}")
                print(f"      • File size: {np.mean(file_sizes):.1f} KB (range: {min(file_sizes):.1f} - {max(file_sizes):.1f})")
                
                if len(dim_counter) == 1:
                    print(f"      ✅ All images have SAME dimensions!")
                else:
                    print(f"      ⚠️  Images have DIFFERENT dimensions!")
    
    def analyze_labels(self):
        """Analyze YOLO format labels"""
        self.results['labels'] = {}
        
        for split in self.splits:
            labels_path = self.dataset_path / split / 'labels'
            
            if not labels_path.exists():
                continue
            
            print(f"\n   Analyzing {split} labels...")
            
            class_counts = defaultdict(int)
            bbox_widths = []
            bbox_heights = []
            bbox_areas = []
            objects_per_image = []
            empty_labels = 0
            
            label_files = list(labels_path.glob('*.txt'))
            
            for label_path in tqdm(label_files, desc=f"   {split}", leave=False):
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    if not lines:
                        empty_labels += 1
                        objects_per_image.append(0)
                        continue
                    
                    objects_per_image.append(len(lines))
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            class_counts[class_id] += 1
                            bbox_widths.append(width)
                            bbox_heights.append(height)
                            bbox_areas.append(width * height)
                
                except Exception as e:
                    print(f"   ⚠️  Error reading {label_path.name}: {e}")
            
            if objects_per_image:
                self.results['labels'][split] = {
                    'total_labels': len(label_files),
                    'empty_labels': empty_labels,
                    'total_objects': sum(objects_per_image),
                    'objects_per_image_mean': np.mean(objects_per_image),
                    'objects_per_image_max': max(objects_per_image),
                    'class_distribution': dict(class_counts),
                    'num_classes': len(class_counts),
                    'bbox_width_mean': np.mean(bbox_widths) if bbox_widths else 0,
                    'bbox_height_mean': np.mean(bbox_heights) if bbox_heights else 0,
                    'bbox_area_mean': np.mean(bbox_areas) if bbox_areas else 0,
                    'bbox_width_range': (min(bbox_widths), max(bbox_widths)) if bbox_widths else (0, 0),
                    'bbox_height_range': (min(bbox_heights), max(bbox_heights)) if bbox_heights else (0, 0),
                }
                
                print(f"   {split.upper()}:")
                print(f"      • Total label files: {len(label_files)}")
                print(f"      • Empty labels: {empty_labels}")
                print(f"      • Total objects: {sum(objects_per_image)}")
                print(f"      • Objects per image: {np.mean(objects_per_image):.2f} (max: {max(objects_per_image)})")
                print(f"      • Number of classes: {len(class_counts)}")
                print(f"      • Class distribution: {dict(sorted(class_counts.items()))}")
                print(f"      • Bbox width: {np.mean(bbox_widths):.3f} (range: {min(bbox_widths):.3f} - {max(bbox_widths):.3f})")
                print(f"      • Bbox height: {np.mean(bbox_heights):.3f} (range: {min(bbox_heights):.3f} - {max(bbox_heights):.3f})")
    
    def analyze_image_properties(self):
        """Analyze color, brightness, contrast"""
        self.results['properties'] = {}
        
        for split in self.splits:
            images_path = self.dataset_path / split / 'images'
            
            if not images_path.exists():
                continue
            
            print(f"\n   Analyzing {split} image properties...")
            
            brightness_vals = []
            contrast_vals = []
            color_channels = {'B': [], 'G': [], 'R': []}
            
            image_files = list(images_path.glob('*.*'))[:200]  # Sample 200
            
            for img_path in tqdm(image_files, desc=f"   {split}", leave=False):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Brightness (average pixel value)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        brightness_vals.append(np.mean(gray))
                        contrast_vals.append(np.std(gray))
                        
                        # Color channels
                        for i, channel in enumerate(['B', 'G', 'R']):
                            color_channels[channel].append(np.mean(img[:, :, i]))
                
                except Exception as e:
                    pass
            
            if brightness_vals:
                self.results['properties'][split] = {
                    'brightness_mean': np.mean(brightness_vals),
                    'brightness_std': np.std(brightness_vals),
                    'contrast_mean': np.mean(contrast_vals),
                    'contrast_std': np.std(contrast_vals),
                    'color_means': {ch: np.mean(vals) for ch, vals in color_channels.items()}
                }
                
                print(f"   {split.upper()}:")
                print(f"      • Brightness: {np.mean(brightness_vals):.2f} ± {np.std(brightness_vals):.2f}")
                print(f"      • Contrast: {np.mean(contrast_vals):.2f} ± {np.std(contrast_vals):.2f}")
                print(f"      • Color (BGR): {np.mean(color_channels['B']):.1f}, {np.mean(color_channels['G']):.1f}, {np.mean(color_channels['R']):.1f}")
    
    def analyze_matching(self):
        """Check if image and label files match"""
        self.results['matching'] = {}
        
        for split in self.splits:
            images_path = self.dataset_path / split / 'images'
            labels_path = self.dataset_path / split / 'labels'
            
            if not images_path.exists() or not labels_path.exists():
                continue
            
            image_files = {p.stem for p in images_path.glob('*.*')}
            label_files = {p.stem for p in labels_path.glob('*.txt')}
            
            matched = image_files & label_files
            images_without_labels = image_files - label_files
            labels_without_images = label_files - image_files
            
            self.results['matching'][split] = {
                'matched': len(matched),
                'images_without_labels': len(images_without_labels),
                'labels_without_images': len(labels_without_images),
                'match_percentage': len(matched) / len(image_files) * 100 if image_files else 0
            }
            
            print(f"\n   {split.upper()}:")
            print(f"      • Matched pairs: {len(matched)}")
            print(f"      • Images without labels: {len(images_without_labels)}")
            print(f"      • Labels without images: {len(labels_without_images)}")
            print(f"      • Match percentage: {len(matched) / len(image_files) * 100:.1f}%" if image_files else "      • N/A")
            
            if images_without_labels:
                print(f"      ⚠️  Found {len(images_without_labels)} images without labels!")
            if labels_without_images:
                print(f"      ⚠️  Found {len(labels_without_images)} labels without images!")
    
    def print_summary(self):
        """Print summary statistics"""
        total_images = sum(d['images'] for d in self.results.get('file_counts', {}).values())
        total_labels = sum(d['labels'] for d in self.results.get('file_counts', {}).values())
        total_objects = sum(d.get('total_objects', 0) for d in self.results.get('labels', {}).values())
        
        print(f"\n   📊 OVERALL STATISTICS:")
        print(f"      • Total images: {total_images}")
        print(f"      • Total labels: {total_labels}")
        print(f"      • Total objects: {total_objects}")
        
        # Check dimension consistency
        dimensions_data = self.results.get('dimensions', {})
        all_same = all(d.get('all_same', False) for d in dimensions_data.values())
        
        if all_same:
            print(f"      ✅ All images have the SAME dimensions!")
        else:
            print(f"      ⚠️  Images have DIFFERENT dimensions across splits!")
        
        # Check matching
        matching_data = self.results.get('matching', {})
        all_matched = all(d.get('match_percentage', 0) == 100 for d in matching_data.values())
        
        if all_matched:
            print(f"      ✅ All images have matching labels!")
        else:
            print(f"      ⚠️  Some images/labels don't match!")
    
    def save_report(self):
        """Save analysis report to JSON"""
        output_file = 'segmentation_analysis_report.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\n   💾 Report saved to: {output_file}")
    
    def create_visualizations(self):
        """Create visualization plots"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Segmentation Dataset Analysis', fontsize=16, fontweight='bold')
            
            # 1. File counts
            if 'file_counts' in self.results:
                ax = axes[0, 0]
                splits = list(self.results['file_counts'].keys())
                images = [self.results['file_counts'][s]['images'] for s in splits]
                labels = [self.results['file_counts'][s]['labels'] for s in splits]
                
                x = np.arange(len(splits))
                width = 0.35
                ax.bar(x - width/2, images, width, label='Images', color='skyblue')
                ax.bar(x + width/2, labels, width, label='Labels', color='lightcoral')
                ax.set_xlabel('Split')
                ax.set_ylabel('Count')
                ax.set_title('File Counts per Split')
                ax.set_xticks(x)
                ax.set_xticklabels(splits)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
            
            # 2. Dimension distribution
            if 'dimensions' in self.results:
                ax = axes[0, 1]
                for split in self.results['dimensions'].keys():
                    data = self.results['dimensions'][split]
                    ax.scatter(data['width_mean'], data['height_mean'], s=200, label=split, alpha=0.7)
                ax.set_xlabel('Width (pixels)')
                ax.set_ylabel('Height (pixels)')
                ax.set_title('Image Dimensions')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 3. Class distribution
            if 'labels' in self.results:
                ax = axes[0, 2]
                for split in self.results['labels'].keys():
                    class_dist = self.results['labels'][split].get('class_distribution', {})
                    if class_dist:
                        classes = sorted(class_dist.keys())
                        counts = [class_dist[c] for c in classes]
                        ax.bar(classes, counts, alpha=0.6, label=split)
                ax.set_xlabel('Class ID')
                ax.set_ylabel('Count')
                ax.set_title('Class Distribution')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
            
            # 4. Objects per image
            if 'labels' in self.results:
                ax = axes[1, 0]
                splits = list(self.results['labels'].keys())
                obj_per_img = [self.results['labels'][s]['objects_per_image_mean'] for s in splits]
                ax.bar(splits, obj_per_img, color='lightgreen')
                ax.set_xlabel('Split')
                ax.set_ylabel('Objects per Image')
                ax.set_title('Average Objects per Image')
                ax.grid(axis='y', alpha=0.3)
            
            # 5. Brightness & Contrast
            if 'properties' in self.results:
                ax = axes[1, 1]
                splits = list(self.results['properties'].keys())
                brightness = [self.results['properties'][s]['brightness_mean'] for s in splits]
                contrast = [self.results['properties'][s]['contrast_mean'] for s in splits]
                
                x = np.arange(len(splits))
                width = 0.35
                ax.bar(x - width/2, brightness, width, label='Brightness', color='gold')
                ax.bar(x + width/2, contrast, width, label='Contrast', color='orange')
                ax.set_xlabel('Split')
                ax.set_ylabel('Value')
                ax.set_title('Brightness & Contrast')
                ax.set_xticks(x)
                ax.set_xticklabels(splits)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
            
            # 6. Matching percentage
            if 'matching' in self.results:
                ax = axes[1, 2]
                splits = list(self.results['matching'].keys())
                match_pct = [self.results['matching'][s]['match_percentage'] for s in splits]
                colors = ['green' if p == 100 else 'orange' for p in match_pct]
                ax.bar(splits, match_pct, color=colors)
                ax.set_xlabel('Split')
                ax.set_ylabel('Match %')
                ax.set_title('Image-Label Matching')
                ax.set_ylim([0, 105])
                ax.axhline(y=100, color='red', linestyle='--', alpha=0.5)
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('segmentation_analysis.png', dpi=300, bbox_inches='tight')
            print(f"   📊 Visualization saved to: segmentation_analysis.png")
            
        except Exception as e:
            print(f"   ⚠️  Could not create visualizations: {e}")


def main():
    analyzer = SegmentationDatasetAnalyzer("segdataset")
    analyzer.analyze_all()


if __name__ == "__main__":
    main()
