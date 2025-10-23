"""
Segmentation Training with U-Net
Model: U-Net with various backbones (ResNet, EfficientNet, MobileNet)
Dataset: Skin lesion segmentation (YOLO format)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.metrics import jaccard_score
import json
from datetime import datetime

class SegmentationDataset(Dataset):
    """Custom dataset for skin lesion segmentation"""
    
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob('*.*')))
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Read image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read label and create mask
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        mask = self.create_mask_from_label(label_path, image.shape[:2])
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to tensor if not already (albumentations with ToTensorV2 returns tensor)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        else:
            mask = mask.float()
        
        # Add channel dimension if needed
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask
    
    def create_mask_from_label(self, label_path, img_shape):
        """Create binary mask from YOLO segmentation label"""
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if not label_path.exists():
            return mask
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            coords = list(map(float, parts[1:]))
            
            if len(coords) > 4:
                # Polygon segmentation
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i+1] * h)
                    points.append([x, y])
                
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)
            else:
                # Bounding box
                x_center, y_center, width, height = coords[:4]
                x_center = int(x_center * w)
                y_center = int(y_center * h)
                box_w = int(width * w)
                box_h = int(height * h)
                
                x1 = int(x_center - box_w/2)
                y1 = int(y_center - box_h/2)
                x2 = int(x_center + box_w/2)
                y2 = int(y_center + box_h/2)
                
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
        
        return mask


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combination of BCE and Dice loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class SegmentationTrainer:
    """Training manager for segmentation model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = self.create_model()
        
        # Create dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders()
        
        # Loss and optimizer
        self.criterion = CombinedLoss(bce_weight=0.3, dice_weight=0.7)  # Dice'a daha çok ağırlık
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler - Cosine annealing (daha iyi)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # İlk restart 10 epoch sonra
            T_mult=2,  # Her restart'ta period 2x olsun
            eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_dice': [],
            'lr': []
        }
        
        # Best metrics
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
    
    def create_model(self):
        """Create medical segmentation model"""
        print(f"\nCreating {self.config['architecture']} model with {self.config['backbone']} backbone...")
        
        # Select architecture
        if self.config['architecture'] == 'unet':
            model = smp.Unet(
                encoder_name=self.config['backbone'],
                encoder_weights='imagenet',
                in_channels=3,
                classes=1,
                activation=None
            )
        elif self.config['architecture'] == 'unet++':
            model = smp.UnetPlusPlus(
                encoder_name=self.config['backbone'],
                encoder_weights='imagenet',
                in_channels=3,
                classes=1,
                activation=None
            )
        elif self.config['architecture'] == 'manet':
            model = smp.MAnet(  # Mobile Attention U-Net
                encoder_name=self.config['backbone'],
                encoder_weights='imagenet',
                in_channels=3,
                classes=1,
                activation=None
            )
        elif self.config['architecture'] == 'deeplabv3+':
            model = smp.DeepLabV3Plus(
                encoder_name=self.config['backbone'],
                encoder_weights='imagenet',
                in_channels=3,
                classes=1,
                activation=None
            )
        else:
            raise ValueError(f"Unknown architecture: {self.config['architecture']}")
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        return model
    
    def create_dataloaders(self):
        """Create training and validation dataloaders"""
        print(f"\nCreating dataloaders...")
        
        # Training augmentation - STRONGER for better generalization
        # NOTE: CLAHE, denoising, hair removal already applied in preprocessing
        train_transform = A.Compose([
            # Geometric transformations - DÜZELTİLDİ
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),  # Arttırıldı
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=45, p=0.7),  # Daha agresif
            A.ElasticTransform(alpha=1, sigma=50, p=0.4),  # Arttırıldı
            
            # Color & lighting - AZALTILDI (segmentasyon için renk çok önemli değil)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            
            # Normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation transform (no augmentation)
        val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Create datasets
        train_dataset = SegmentationDataset(
            images_dir=f"{self.config['data_path']}/train/images",
            labels_dir=f"{self.config['data_path']}/train/labels",
            transform=train_transform
        )
        
        val_dataset = SegmentationDataset(
            images_dir=f"{self.config['data_path']}/valid/images",
            labels_dir=f"{self.config['data_path']}/valid/labels",
            transform=val_transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        print(f"   Batch size: {self.config['batch_size']}")
        
        return train_loader, val_loader
    
    def calculate_metrics(self, predictions, targets):
        """Calculate IoU and Dice metrics"""
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).float()
        
        # Flatten
        preds_flat = predictions.cpu().numpy().flatten()
        targets_flat = targets.cpu().numpy().flatten()
        
        # IoU (Jaccard)
        iou = jaccard_score(targets_flat, preds_flat, average='binary', zero_division=0)
        
        # Dice coefficient
        intersection = (preds_flat * targets_flat).sum()
        dice = (2. * intersection) / (preds_flat.sum() + targets_flat.sum() + 1e-8)
        
        return iou, dice
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        total_iou = 0
        total_dice = 0
        
        progress_bar = tqdm(self.val_loader, desc='Validation')
        
        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Calculate metrics
            iou, dice = self.calculate_metrics(outputs, masks)
            
            total_loss += loss.item()
            total_iou += iou
            total_dice += dice
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss/len(progress_bar):.4f}',
                'iou': f'{total_iou/len(progress_bar):.4f}'
            })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_iou = total_iou / len(self.val_loader)
        avg_dice = total_dice / len(self.val_loader)
        
        return avg_loss, avg_iou, avg_dice
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_iou': self.best_val_iou,
            'history': self.history,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'checkpoints' / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"   Best model saved! (IoU: {self.best_val_iou:.4f})")
    
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # IoU
        axes[0, 1].plot(self.history['val_iou'], label='Val IoU', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('Validation IoU (Jaccard Index)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Dice
        axes[1, 0].plot(self.history['val_dice'], label='Val Dice', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].set_title('Validation Dice Coefficient')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.history['lr'], label='Learning Rate', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=200, bbox_inches='tight')
        print(f"\nTraining history saved: {self.output_dir / 'training_history.png'}")
    
    def final_evaluation(self):
        """Comprehensive evaluation on test set"""
        # Load best model
        checkpoint_path = self.output_dir / 'checkpoints' / 'best_model.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        
        # Create test dataloader
        test_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        test_dataset = SegmentationDataset(
            images_dir=Path(self.config['data_path']) / 'test' / 'images',
            labels_dir=Path(self.config['data_path']) / 'test' / 'labels',
            transform=test_transform
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        print(f"Test samples: {len(test_dataset)}")
        
        # Evaluate
        self.model.eval()
        
        all_iou = []
        all_dice = []
        all_precision = []
        all_recall = []
        
        sample_images = []
        sample_masks = []
        sample_preds = []
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc='Testing')):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Predict
                outputs = self.model(images)
                preds = torch.sigmoid(outputs) > 0.5
                
                # Calculate metrics for each sample
                for i in range(images.shape[0]):
                    pred = preds[i].float()
                    mask = masks[i]
                    
                    # IoU
                    intersection = (pred * mask).sum()
                    union = pred.sum() + mask.sum() - intersection
                    iou = (intersection / (union + 1e-8)).item()
                    all_iou.append(iou)
                    
                    # Dice
                    dice = (2 * intersection / (pred.sum() + mask.sum() + 1e-8)).item()
                    all_dice.append(dice)
                    
                    # Precision & Recall
                    tp = (pred * mask).sum().item()
                    fp = (pred * (1 - mask)).sum().item()
                    fn = ((1 - pred) * mask).sum().item()
                    
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    
                    all_precision.append(precision)
                    all_recall.append(recall)
                    
                    # Save samples for visualization
                    if len(sample_images) < 16:
                        sample_images.append(images[i].cpu())
                        sample_masks.append(masks[i].cpu())
                        sample_preds.append(preds[i].cpu())
        
        # Calculate statistics
        mean_iou = np.mean(all_iou)
        std_iou = np.std(all_iou)
        mean_dice = np.mean(all_dice)
        std_dice = np.std(all_dice)
        mean_precision = np.mean(all_precision)
        mean_recall = np.mean(all_recall)
        
        # Print results
        print(f"\nTest Set Results (n={len(test_dataset)}):")
        print(f"   IoU:       {mean_iou:.4f} ± {std_iou:.4f}")
        print(f"   Dice:      {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"   Precision: {mean_precision:.4f}")
        print(f"   Recall:    {mean_recall:.4f}")
        
        # Save metrics to JSON
        metrics = {
            'test_samples': len(test_dataset),
            'iou_mean': float(mean_iou),
            'iou_std': float(std_iou),
            'dice_mean': float(mean_dice),
            'dice_std': float(std_dice),
            'precision': float(mean_precision),
            'recall': float(mean_recall),
            'iou_per_sample': [float(x) for x in all_iou],
            'dice_per_sample': [float(x) for x in all_dice]
        }
        
        with open(self.output_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved: {self.output_dir / 'test_metrics.json'}")
        
        # Visualize predictions
        self.visualize_predictions(sample_images, sample_masks, sample_preds)
        
        # Plot metric distributions
        self.plot_metric_distributions(all_iou, all_dice)
    
    def visualize_predictions(self, images, masks, preds):
        """Visualize sample predictions"""
        n_samples = len(images)
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(n_samples):
            # Denormalize image
            img = images[idx].permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            mask = masks[idx].squeeze().numpy()
            pred = preds[idx].squeeze().numpy()
            
            # Image
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title('Input Image')
            axes[idx, 0].axis('off')
            
            # Ground Truth
            axes[idx, 1].imshow(img)
            axes[idx, 1].imshow(mask, alpha=0.5, cmap='Reds')
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')
            
            # Prediction
            axes[idx, 2].imshow(img)
            axes[idx, 2].imshow(pred, alpha=0.5, cmap='Greens')
            axes[idx, 2].set_title('Prediction')
            axes[idx, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_predictions.png', dpi=150, bbox_inches='tight')
        print(f"Predictions saved: {self.output_dir / 'test_predictions.png'}")
        plt.close()
    
    def plot_metric_distributions(self, iou_scores, dice_scores):
        """Plot distribution of metrics across test set"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # IoU distribution
        axes[0].hist(iou_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(iou_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(iou_scores):.4f}')
        axes[0].set_xlabel('IoU Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('IoU Distribution on Test Set')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Dice distribution
        axes[1].hist(dice_scores, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(dice_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dice_scores):.4f}')
        axes[1].set_xlabel('Dice Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Dice Distribution on Test Set')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metric_distributions.png', dpi=150, bbox_inches='tight')
        print(f"Metric distributions saved: {self.output_dir / 'metric_distributions.png'}")
        plt.close()
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        start_time = datetime.now()
        
        for epoch in range(self.config['epochs']):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"{'='*80}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_iou, val_dice = self.validate()
            
            # Update scheduler (Cosine scheduler her epoch step alır)
            self.scheduler.step()
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_iou'].append(val_iou)
            self.history['val_dice'].append(val_dice)
            self.history['lr'].append(current_lr)    
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val IoU: {val_iou:.4f}")
            print(f"   Val Dice: {val_dice:.4f}")
            print(f"   Learning Rate: {current_lr:.6f}")
            # Save checkpoint
            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou
                self.best_val_loss = val_loss
            self.save_checkpoint(epoch, is_best=is_best)          
            # Early stopping check
            if current_lr < 1e-7:
                print("\nLearning rate too small. Stopping training.")
                break
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Duration: {duration}")
        print(f"Best Val IoU: {self.best_val_iou:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        
        # Plot history
        self.plot_history()
        
        # Final evaluation on test set
        print("\n" + "=" * 80)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 80)
        self.final_evaluation()
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Config saved: {self.output_dir / 'config.json'}")


def main():
    # Configuration
    config = {
        'data_path': 'segdataset_processed',
        'architecture': 'unet', 
        'backbone': 'resnet34',  # ResNet34 - daha güçlü encoder
        'epochs': 20,
        'batch_size': 8,  # Daha küçük batch - daha iyi gradient güncellemeleri
        'learning_rate': 1e-3,  # 10x daha yüksek - daha hızlı öğrenme
        'weight_decay': 1e-4,  # Daha güçlü regularization
        
        # System
        'num_workers': 4,
        'output_dir': 'segmentation_outputs',
    }
    
    print("=" * 80)
    print("SKIN LESION SEGMENTATION TRAINING")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    trainer = SegmentationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
