import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
import os
from pathlib import Path
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SkinDiseaseClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Skin Disease Classifier & Segmentation")
        self.root.geometry("1800x900")
        self.root.configure(bg='#f0f0f0')
        
        self.model = None
        self.seg_model = None
        self.class_names = None
        self.current_image = None
        self.current_image_path = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.setup_ui()
        self.load_model()
        self.load_segmentation_model()
        
    def setup_ui(self):
        """Setup the user interface"""
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🔬 Skin Disease Classification & Segmentation",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=20)
        
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        left_frame = tk.LabelFrame(
            main_frame,
            text="📷 Original Image",
            font=('Arial', 12, 'bold'),
            bg='white',
            relief=tk.RAISED,
            borderwidth=2
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_label = tk.Label(left_frame, bg='white', text="No image loaded\n\nPress 'Load Image' to start", 
                                   font=('Arial', 14), fg='gray')
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        middle_frame = tk.LabelFrame(
            main_frame,
            text="🔥 Grad-CAM Heat Map",
            font=('Arial', 12, 'bold'),
            bg='white',
            relief=tk.RAISED,
            borderwidth=2
        )
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.explanation_label = tk.Label(middle_frame, bg='white', text="Grad-CAM heat map\n\nKırmızı bölgeler = Yüksek aktivasyon", 
                                     font=('Arial', 12), fg='gray')
        self.explanation_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add segmentation frame
        seg_frame = tk.LabelFrame(
            main_frame,
            text="🎯 Lesion Segmentation",
            font=('Arial', 12, 'bold'),
            bg='white',
            relief=tk.RAISED,
            borderwidth=2
        )
        seg_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.segmentation_label = tk.Label(seg_frame, bg='white', text="U-Net Segmentation\n\nLezyon bölgesi maskelenir", 
                                     font=('Arial', 12), fg='gray')
        self.segmentation_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        button_frame = tk.LabelFrame(
            right_frame,
            text="🎮 Controls",
            font=('Arial', 12, 'bold'),
            bg='white',
            relief=tk.RAISED,
            borderwidth=2
        )
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.load_btn = tk.Button(
            button_frame,
            text="📁 Load Image",
            command=self.load_image,
            font=('Arial', 11, 'bold'),
            bg='#3498db',
            fg='white',
            relief=tk.RAISED,
            cursor='hand2',
            height=2
        )
        self.load_btn.pack(fill=tk.X, pady=5, padx=10)
        
        self.classify_btn = tk.Button(
            button_frame,
            text="🔍 Classify",
            command=self.classify_image,
            font=('Arial', 11, 'bold'),
            bg='#27ae60',
            fg='white',
            relief=tk.RAISED,
            cursor='hand2',
            height=2,
            state=tk.DISABLED
        )
        self.classify_btn.pack(fill=tk.X, pady=5, padx=10)
        
        self.camera_btn = tk.Button(
            button_frame,
            text="📷 Open Camera",
            command=self.open_camera,
            font=('Arial', 11, 'bold'),
            bg='#e74c3c',
            fg='white',
            relief=tk.RAISED,
            cursor='hand2',
            height=2
        )
        self.camera_btn.pack(fill=tk.X, pady=5, padx=10)
        
        results_frame = tk.LabelFrame(
            right_frame,
            text="📊 Classification Results",
            font=('Arial', 12, 'bold'),
            bg='white',
            relief=tk.RAISED,
            borderwidth=2
        )
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        pred_frame = tk.Frame(results_frame, bg='white')
        pred_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            pred_frame,
            text="🎯 Predicted Disease:",
            font=('Arial', 11, 'bold'),
            bg='white'
        ).pack(anchor=tk.W)
        
        self.prediction_label = tk.Label(
            pred_frame,
            text="---",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#e74c3c',
            wraplength=300,
            justify=tk.LEFT
        )
        self.prediction_label.pack(anchor=tk.W, pady=5)
        
        conf_frame = tk.Frame(results_frame, bg='white')
        conf_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            conf_frame,
            text="📈 Confidence:",
            font=('Arial', 11, 'bold'),
            bg='white'
        ).pack(anchor=tk.W)
        
        self.confidence_label = tk.Label(
            conf_frame,
            text="---",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#16a085'
        )
        self.confidence_label.pack(anchor=tk.W, pady=5)
        
        top_frame = tk.Frame(results_frame, bg='white')
        top_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(
            top_frame,
            text="🏆 Top 5 Predictions:",
            font=('Arial', 11, 'bold'),
            bg='white'
        ).pack(anchor=tk.W)
        
        scroll_frame = tk.Frame(top_frame, bg='white')
        scroll_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(scroll_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.predictions_text = tk.Text(
            scroll_frame,
            height=10,
            font=('Courier', 9),
            bg='#ecf0f1',
            relief=tk.SUNKEN,
            yscrollcommand=scrollbar.set
        )
        self.predictions_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.predictions_text.yview)
        
        self.status_label = tk.Label(
            self.root,
            text="🟢 Ready",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='white',
            anchor=tk.W
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
    def load_model(self):
        """Load the trained model and class names"""
        try:
            self.status_label.config(text="🔄 Loading model...")
            self.root.update()
            
            model_path = 'model/best_model.keras'
            class_names_path = 'model/class_names.txt'
            
            if not os.path.exists(model_path):
                model_path = 'model/skin_disease_model.keras'
            
            if not os.path.exists(model_path):
                self.status_label.config(text="❌ Error: Model not found!")
                messagebox.showerror("Error", f"Model not found!\n\nPlease train the model first using train.py")
                return
            
            if not os.path.exists(class_names_path):
                self.status_label.config(text="❌ Error: Class names file not found!")
                messagebox.showerror("Error", f"Class names file not found!")
                return
            
            print(f"Loading model from: {model_path}")
            self.model = keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
            
            print(f"Loading class names from: {class_names_path}")
            with open(class_names_path, 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"✅ Loaded {len(self.class_names)} classes")
            
            self.status_label.config(text=f"🟢 Ready - Model loaded! ({len(self.class_names)} classes)")
            messagebox.showinfo("Success", f"Model loaded successfully!\n{len(self.class_names)} disease classes found")
            
        except Exception as e:
            self.status_label.config(text=f"❌ Error loading model: {str(e)}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def load_segmentation_model(self):
        """Load the segmentation model"""
        try:
            print("\n🔄 Loading segmentation model...")
            
            seg_model_path = 'segmentation_outputs/checkpoints/best_model.pth'
            
            if not os.path.exists(seg_model_path):
                print("⚠️ Segmentation model not found, segmentation will be disabled")
                print(f"   Looking for: {os.path.abspath(seg_model_path)}")
                self.seg_model = None
                return
            
            # Load config
            config_path = 'segmentation_outputs/config.json'
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                architecture = config.get('architecture', 'unet')
                backbone = config.get('backbone', 'efficientnet-b0')
            else:
                architecture = 'unet'
                backbone = 'efficientnet-b0'
            
            print(f"   Architecture: {architecture}")
            print(f"   Backbone: {backbone}")
            
            # Create model
            if architecture == 'unet':
                self.seg_model = smp.Unet(
                    encoder_name=backbone,
                    encoder_weights=None,
                    in_channels=3,
                    classes=1,
                    activation=None
                )
            elif architecture == 'unet++':
                self.seg_model = smp.UnetPlusPlus(
                    encoder_name=backbone,
                    encoder_weights=None,
                    in_channels=3,
                    classes=1,
                    activation=None
                )
            else:
                self.seg_model = smp.Unet(
                    encoder_name=backbone,
                    encoder_weights=None,
                    in_channels=3,
                    classes=1,
                    activation=None
                )
            
            # Load weights
            checkpoint = torch.load(seg_model_path, map_location=self.device)
            self.seg_model.load_state_dict(checkpoint['model_state_dict'])
            self.seg_model.to(self.device)
            self.seg_model.eval()
            
            print(f"✅ Segmentation model loaded! (Epoch {checkpoint['epoch']+1})")
            best_iou = checkpoint.get('best_iou', checkpoint.get('best_val_iou', 0))
            print(f"   Best IoU: {best_iou:.4f}")
            
            messagebox.showinfo("Segmentation Model", 
                f"Segmentation model loaded!\n\nArchitecture: {architecture}\nBackbone: {backbone}\nBest IoU: {best_iou:.4f}")
            
        except Exception as e:
            print(f"⚠️ Could not load segmentation model: {e}")
            import traceback
            traceback.print_exc()
            self.seg_model = None
    
    def compute_gradcam(self, img):
        """🔥 Grad-CAM ile hastalık bölgesini tespit et"""
        try:
            print("🔥 Computing Grad-CAM...")
            
            # Orijinal resmi preprocess et
            img_array = self.preprocess_image(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Son convolutional layer'ı bul
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:  # Conv layer (batch, height, width, channels)
                    last_conv_layer = layer
                    break
            
            if last_conv_layer is None:
                print("⚠️ No convolutional layer found, using alternative method")
                return self.compute_attention_map(img)
            
            print(f"   Using layer: {last_conv_layer.name}")
            
            # Gradient modeli oluştur
            grad_model = tf.keras.models.Model(
                inputs=[self.model.inputs],
                outputs=[last_conv_layer.output, self.model.output]
            )
            
            # Forward pass ve gradient hesapla
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                predicted_class = tf.argmax(predictions[0])
                class_channel = predictions[:, predicted_class]
            
            # Gradient'ları al
            grads = tape.gradient(class_channel, conv_outputs)
            
            # Global average pooling
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Conv output ile ağırlıkları çarp
            conv_outputs = conv_outputs[0]
            pooled_grads = pooled_grads.numpy()
            conv_outputs = conv_outputs.numpy()
            
            for i in range(len(pooled_grads)):
                conv_outputs[:, :, i] *= pooled_grads[i]
            
            # Tüm kanalları ortala
            heatmap = np.mean(conv_outputs, axis=-1)
            
            # Normalize et (0-1 arası)
            heatmap = np.maximum(heatmap, 0)  # ReLU
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            print(f"   ✅ Grad-CAM computed! Predicted class: {int(predicted_class)}")
            
            return heatmap, int(predicted_class)
            
        except Exception as e:
            print(f"❌ Error computing Grad-CAM: {e}")
            import traceback
            traceback.print_exc()
            return self.compute_attention_map(img)
    
    def compute_attention_map(self, img):
        """🎯 Basit attention map (Grad-CAM başarısız olursa)"""
        try:
            print("   Using simple attention map...")
            
            # Preprocess
            img_array = self.preprocess_image(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Tahmin yap
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            
            # Basit saliency map
            h, w = 224, 224
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            
            # Gaussian-like attention (merkezden uzaklaştıkça azalan)
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            heatmap = np.exp(-distance / (h * 0.3))
            
            # Kenarları vurgula (cilt hastalıkları genelde kenar/yüzeyde)
            edges = cv2.Canny(cv2.resize(img, (224, 224)), 50, 150)
            edge_attention = cv2.GaussianBlur(edges.astype(np.float32), (21, 21), 0)
            edge_attention = edge_attention / (edge_attention.max() + 1e-8)
            
            # Birleştir
            heatmap = heatmap * 0.3 + edge_attention * 0.7
            heatmap = heatmap / (heatmap.max() + 1e-8)
            
            return heatmap, int(predicted_class)
            
        except Exception as e:
            print(f"Error in attention map: {e}")
            # En basit: uniform map
            heatmap = np.ones((224, 224)) * 0.5
            predictions = self.model.predict(np.expand_dims(self.preprocess_image(img), axis=0), verbose=0)
            predicted_class = np.argmax(predictions[0])
            return heatmap, int(predicted_class)
    
    def create_explanation_image(self, img, heatmap):
        """✅ Grad-CAM heatmap görselleştirmesi"""
        try:
            h, w = img.shape[:2]
            
            # Heatmap'i orijinal boyuta getir
            heatmap_resized = cv2.resize(heatmap, (w, h))
            
            # Heatmap'i 0-255 arası yap
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            
            # Jet colormap uygula (mavi=düşük, kırmızı=yüksek)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            # Orijinal resimle blend et
            superimposed = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
            
            print(f"   ✅ Grad-CAM visualization created")
            
            return superimposed
            
        except Exception as e:
            print(f"❌ Error creating Grad-CAM visualization: {e}")
            import traceback
            traceback.print_exc()
            return img
    
    def load_image(self):
        """Load an image from file"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.classify_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"🟡 Loaded: {os.path.basename(file_path)}")
    
    def display_image(self, image_path):
        """Display image in the UI"""
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("Error", "Could not read image file!")
            return
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_image = img.copy()
        
        h, w = img.shape[:2]
        max_size = 350
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.image_label.config(image=img_tk, text="")
        self.image_label.image = img_tk
    
    def preprocess_image(self, img):
        """Preprocess image"""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        img = cv2.resize(img, (224, 224))
        
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.CLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        except:
            pass
        
        try:
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        except:
            pass
        
        img = img.astype(np.float32)
        img = keras.applications.mobilenet_v2.preprocess_input(img)
        
        return img
    
    def classify_image(self):
        """Classify the current image"""
        if self.current_image is None or self.model is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        try:
            self.status_label.config(text="🔴 Classifying...")
            self.root.update()
            
            # Predict
            processed_img = self.preprocess_image(self.current_image)
            processed_img_tensor = np.expand_dims(processed_img, axis=0)
            
            predictions = self.model.predict(processed_img_tensor, verbose=0)[0]
            
            top_idx = np.argmax(predictions)
            top_class = self.class_names[top_idx]
            top_confidence = predictions[top_idx] * 100
            
            print(f"\n{'='*60}")
            print(f"🎯 Prediction: {top_class}")
            print(f"📈 Confidence: {top_confidence:.2f}%")
            print(f"{'='*60}\n")
            
            # ✅ Grad-CAM explanation
            self.status_label.config(text="🔥 Computing Grad-CAM (hastalık bölgesi tespit ediliyor)...")
            self.root.update()
            
            heatmap, predicted_class_idx = self.compute_gradcam(self.current_image)
            
            if heatmap is not None:
                print("✅ Creating Grad-CAM visualization...")
                
                # Display img'i resize et
                display_img = self.current_image.copy()
                h, w = display_img.shape[:2]
                max_size = 400
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    display_img = cv2.resize(display_img, (new_w, new_h))
                
                explanation_img = self.create_explanation_image(display_img, heatmap)
                
                if explanation_img is not None:
                    img_pil = Image.fromarray(explanation_img)
                    img_tk = ImageTk.PhotoImage(img_pil)
                    
                    self.explanation_label.config(image=img_tk, text="")
                    self.explanation_label.image = img_tk
                    print("✅ Grad-CAM visualization displayed!")
            
            # 🎯 Segmentation
            if self.seg_model is not None:
                self.status_label.config(text="🎯 Running segmentation (lezyon maskeleniyor)...")
                self.root.update()
                
                seg_mask = self.predict_segmentation(self.current_image)
                
                if seg_mask is not None:
                    print("✅ Creating segmentation visualization...")
                    
                    # Display img'i resize et
                    display_img = self.current_image.copy()
                    h, w = display_img.shape[:2]
                    max_size = 400
                    if max(h, w) > max_size:
                        scale = max_size / max(h, w)
                        new_w, new_h = int(w * scale), int(h * scale)
                        display_img = cv2.resize(display_img, (new_w, new_h))
                        seg_mask = cv2.resize(seg_mask, (new_w, new_h))
                    
                    seg_img = self.create_segmentation_overlay(display_img, seg_mask)
                    
                    if seg_img is not None:
                        img_pil = Image.fromarray(seg_img)
                        img_tk = ImageTk.PhotoImage(img_pil)
                        
                        self.segmentation_label.config(image=img_tk, text="")
                        self.segmentation_label.image = img_tk
                        print("✅ Segmentation visualization displayed!")
                else:
                    print("❌ Segmentation mask is None")
            else:
                print("⚠️ Segmentation model not loaded, skipping segmentation")
                self.segmentation_label.config(text="Segmentation model\nnot available\n\nTrain model first using\nsegtrain.py")
            
            # Update results
            self.prediction_label.config(text=top_class)
            self.confidence_label.config(text=f"{top_confidence:.2f}%")
            
            self.predictions_text.config(state=tk.NORMAL)
            self.predictions_text.delete(1.0, tk.END)
            top_5_idx = np.argsort(predictions)[-5:][::-1]
            
            for i, idx in enumerate(top_5_idx):
                class_name = self.class_names[idx]
                confidence = predictions[idx] * 100
                bar_length = int(confidence / 2.5)
                bar = "█" * bar_length
                
                self.predictions_text.insert(
                    tk.END,
                    f"{i+1}. {class_name}\n   {confidence:6.2f}% {bar}\n\n"
                )
            
            self.predictions_text.config(state=tk.DISABLED)
            
            self.status_label.config(text=f"🟢 Classification complete! Confidence: {top_confidence:.2f}%")
            
        except Exception as e:
            self.status_label.config(text=f"❌ Error: {str(e)}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Classification failed:\n{str(e)}")
    
    def predict_segmentation(self, img):
        """Predict lesion segmentation using U-Net"""
        try:
            print("🎯 Running U-Net segmentation...")
            
            # Preprocess for segmentation
            img_resized = cv2.resize(img, (320, 320))
            
            # Normalize
            transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            transformed = transform(image=img_resized)
            img_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.seg_model(img_tensor)
                mask = torch.sigmoid(output) > 0.5
                mask = mask[0, 0].cpu().numpy().astype(np.uint8)
            
            # Resize mask to original image size
            h, w = img.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            print(f"   ✅ Segmentation complete! Mask size: {mask.shape}")
            
            return mask
            
        except Exception as e:
            print(f"❌ Error in segmentation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_segmentation_overlay(self, img, mask):
        """Create segmentation overlay visualization"""
        try:
            # Create colored overlay
            overlay = img.copy()
            
            # Green overlay for lesion
            overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
            
            # Draw contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            
            # Add text with area
            total_pixels = np.sum(mask > 0)
            img_pixels = mask.shape[0] * mask.shape[1]
            percentage = (total_pixels / img_pixels) * 100
            
            text = f"Lesion Area: {percentage:.1f}%"
            cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            print(f"   ✅ Segmentation overlay created")
            
            return overlay
            
        except Exception as e:
            print(f"❌ Error creating segmentation overlay: {e}")
            return img
    
    def open_camera(self):
        """Open camera"""
        if self.model is None:
            messagebox.showwarning("Warning", "Model not loaded!")
            return
        
        camera_window = tk.Toplevel(self.root)
        camera_window.title("📷 Camera Capture")
        camera_window.geometry("800x700")
        camera_window.configure(bg='#2c3e50')
        
        cam_label = tk.Label(camera_window, bg='black')
        cam_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        info_label = tk.Label(
            camera_window,
            text="Press SPACE to capture | ESC to close",
            font=('Arial', 12, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        info_label.pack(pady=10)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            camera_window.destroy()
            messagebox.showerror("Error", "Cannot open camera!")
            return
        
        def update_camera():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_frame = cv2.resize(frame_rgb, (640, 480))
                
                img_pil = Image.fromarray(display_frame)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                cam_label.config(image=img_tk)
                cam_label.image = img_tk
                
                camera_window.current_frame = frame_rgb
            
            if camera_window.winfo_exists():
                camera_window.after(10, update_camera)
        
        def on_key(event):
            if event.keysym == 'space':
                if hasattr(camera_window, 'current_frame'):
                    self.current_image = camera_window.current_frame.copy()
                    self.display_image_from_array(self.current_image)
                    self.classify_btn.config(state=tk.NORMAL)
                    self.status_label.config(text="🟡 Image captured from camera")
                    camera_window.destroy()
                    cap.release()
            elif event.keysym == 'Escape':
                camera_window.destroy()
                cap.release()
        
        camera_window.bind('<KeyPress>', on_key)
        camera_window.protocol("WM_DELETE_WINDOW", lambda: [cap.release(), camera_window.destroy()])
        
        update_camera()
    
    def display_image_from_array(self, img_array):
        """Display image from array"""
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        self.current_image = img_array.copy()
        
        h, w = img_array.shape[:2]
        max_size = 350
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_array = cv2.resize(img_array, (new_w, new_h))
        
        img_pil = Image.fromarray(img_array)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.image_label.config(image=img_tk, text="")
        self.image_label.image = img_tk

def main():
    root = tk.Tk()
    app = SkinDiseaseClassifierUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()