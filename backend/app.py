from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os

# Try to import segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    print("⚠ Warning: segmentation_models_pytorch not installed. Segmentation will be disabled.")
    print("Install with: pip install segmentation-models-pytorch")
    SMP_AVAILABLE = False

app = Flask(__name__)
# Allow CORS for React development server
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:5174", "http://127.0.0.1:5175"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Model paths
CLASSIFICATION_MODEL_PATH = '../model/best_model.keras'
CLASS_NAMES_PATH = '../model/class_names.txt'
SEGMENTATION_MODEL_PATH = '../segmentation_outputs/checkpoints/best_model.pth'

# Load models
print("Loading models...")
try:
    classification_model = tf.keras.models.load_model(CLASSIFICATION_MODEL_PATH)
    print("[OK] Classification model loaded")
except Exception as e:
    print(f"[ERROR] Failed to load classification model: {e}")
    classification_model = None

try:
    with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"[OK] Class names loaded: {len(class_names)} classes")
except Exception as e:
    print(f"[ERROR] Failed to load class names: {e}")
    class_names = []

try:
    if not SMP_AVAILABLE:
        print("[WARNING] Segmentation disabled: segmentation_models_pytorch not installed")
        segmentation_model = None
    else:
        # Load segmentation checkpoint
        checkpoint = torch.load(SEGMENTATION_MODEL_PATH, map_location=torch.device('cpu'))
        
        # Check if it's a state dict or full model
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Loading segmentation model from checkpoint...")
            
            # Create UnetPlusPlus model with ResNet34 backbone (same as training)
            segmentation_model = smp.UnetPlusPlus(
                encoder_name='resnet34',  # ResNet34 encoder
                encoder_weights=None,      # Don't load pretrained weights
                in_channels=3,
                classes=1,
                activation=None
            )
            
            # Load the saved weights (with strict=False to ignore missing/unexpected keys)
            segmentation_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            segmentation_model.eval()
            print("[OK] Segmentation model loaded successfully (ResNet34 + UnetPlusPlus)")
            
        elif isinstance(checkpoint, dict):
            print("[WARNING] Checkpoint format not recognized. Available keys:", checkpoint.keys())
            segmentation_model = None
        else:
            # It's a full model (less common)
            segmentation_model = checkpoint
            segmentation_model.eval()
            print("[OK] Segmentation model loaded (full model)")
            
except Exception as e:
    print(f"[ERROR] Failed to load segmentation model: {e}")
    import traceback
    traceback.print_exc()
    segmentation_model = None

def preprocess_image_for_classification(image, target_size=(224, 224)):
    """Preprocess image for classification model"""
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_for_segmentation(image, target_size=(256, 256)):
    """Preprocess image for segmentation model"""
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0)
    return img_tensor

def generate_gradcam(model, img_array, class_idx):
    """Generate Grad-CAM heatmap"""
    try:
        # Get the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return None
        
        # Create a model that outputs the last conv layer and predictions
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        return heatmap
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None

def apply_heatmap_to_image(image, heatmap):
    """Apply heatmap overlay to original image"""
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.width, image.height))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Convert image to array
    img_array = np.array(image)
    
    # Superimpose heatmap
    superimposed = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    
    return Image.fromarray(superimposed)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'classification': classification_model is not None,
            'segmentation': segmentation_model is not None,
            'class_names': len(class_names) > 0
        }
    })

@app.route('/api/classify', methods=['POST'])
def classify_image():
    """Classification endpoint"""
    try:
        if classification_model is None:
            return jsonify({'error': 'Classification model not loaded. Please check model path.'}), 500
        
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        print(f"Received image: {file.filename}, content_type: {file.content_type}")
        
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        print(f"Image loaded: size={img.size}, mode={img.mode}")
        
        # Preprocess
        img_array = preprocess_image_for_classification(img)
        print(f"Image preprocessed: shape={img_array.shape}")
        
        # Predict
        predictions = classification_model.predict(img_array, verbose=0)
        print(f"Predictions: {predictions[0][:3]}...")
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        
        result = {
            'disease': class_names[np.argmax(predictions)] if class_names else f"Class {np.argmax(predictions)}",
            'confidence': float(np.max(predictions) * 100),
            'topPredictions': [
                {
                    'name': class_names[i] if i < len(class_names) else f"Class {i}",
                    'probability': float(predictions[0][i] * 100)
                }
                for i in top_indices
            ]
        }
        
        print(f"Result: {result['disease']} - {result['confidence']:.2f}%")
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in classify_image: {error_trace}")
        return jsonify({'error': str(e), 'trace': error_trace}), 500

@app.route('/api/segment', methods=['POST'])
def segment_image():
    """Segmentation endpoint"""
    try:
        if segmentation_model is None:
            return jsonify({'error': 'Segmentation model not loaded'}), 500
        
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        original_size = img.size
        
        # Preprocess
        img_tensor = preprocess_image_for_segmentation(img)
        
        # Predict
        with torch.no_grad():
            output = segmentation_model(img_tensor)
            mask = torch.sigmoid(output) > 0.5
            mask = mask.squeeze().cpu().numpy()
        
        # Resize mask to original size
        mask_resized = cv2.resize(mask.astype(np.uint8) * 255, original_size)
        
        # Create colored overlay
        img_array = np.array(img)
        overlay = np.zeros_like(img_array)
        overlay[mask_resized > 0] = [255, 0, 0]  # Red overlay
        
        # Blend
        result_img = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)
        result_pil = Image.fromarray(result_img)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        result_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return send_file(img_byte_arr, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/gradcam', methods=['POST'])
def gradcam_visualization():
    """Grad-CAM endpoint"""
    try:
        if classification_model is None:
            return jsonify({'error': 'Classification model not loaded'}), 500
        
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Preprocess
        img_array = preprocess_image_for_classification(img)
        
        # Get prediction
        predictions = classification_model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions)
        
        # Generate Grad-CAM
        heatmap = generate_gradcam(classification_model, img_array, class_idx)
        
        if heatmap is None:
            # If Grad-CAM fails, return original image
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            return send_file(img_byte_arr, mimetype='image/png')
        
        # Apply heatmap to image
        result_img = apply_heatmap_to_image(img, heatmap)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return send_file(img_byte_arr, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_complete():
    """Complete analysis endpoint - all models at once"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        
        # Classification
        classify_result = None
        if classification_model is not None:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_array = preprocess_image_for_classification(img)
            predictions = classification_model.predict(img_array, verbose=0)
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            
            classify_result = {
                'disease': class_names[np.argmax(predictions)] if class_names else f"Class {np.argmax(predictions)}",
                'confidence': float(np.max(predictions) * 100),
                'topPredictions': [
                    {
                        'name': class_names[i] if class_names else f"Class {i}",
                        'probability': float(predictions[0][i] * 100)
                    }
                    for i in top_indices
                ]
            }
        
        return jsonify({
            'classification': classify_result,
            'message': 'Analysis complete. Use /api/segment and /api/gradcam for images.'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Skin Disease Analysis API Server")
    print("="*50)
    print("\nAvailable endpoints:")
    print("  GET  /api/health     - Health check")
    print("  POST /api/classify   - Classification only")
    print("  POST /api/segment    - Segmentation only")
    print("  POST /api/gradcam    - Grad-CAM visualization")
    print("  POST /api/analyze    - Complete analysis")
    print("\n" + "="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
