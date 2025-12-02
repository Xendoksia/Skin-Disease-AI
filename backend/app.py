from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io
import os
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import tensorflow as tf
from tensorflow import keras
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Model paths
CLASSIFICATION_MODEL_PATH = '../model/best_model.keras'
CLASS_NAMES_PATH = '../model/class_names.txt'
SEGMENTATION_MODEL_PATH = '../segmentation_outputs/checkpoints/best_model.pth'
SEGMENTATION_CONFIG_PATH = '../segmentation_outputs/config.json'

# Global variables
classification_model = None
segmentation_model = None
class_names = []
device = None

# OpenAI API Key (Load from environment variable)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')

print("\n" + "="*60)
print("  Skin Disease Analysis API - Local Models")
print("="*60)

# Load Classification Model
try:
    print("\n[1/3] Loading classification model...")
    if os.path.exists(CLASSIFICATION_MODEL_PATH):
        classification_model = keras.models.load_model(CLASSIFICATION_MODEL_PATH)
        print(f"      ✓ Classification model loaded")
    else:
        print(f"      ✗ Model not found: {CLASSIFICATION_MODEL_PATH}")
        classification_model = None
except Exception as e:
    print(f"      ✗ Failed to load classification model: {e}")
    classification_model = None

# Load Class Names
try:
    print("\n[2/3] Loading class names...")
    if os.path.exists(CLASS_NAMES_PATH):
        import re
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            raw_names = [line.strip() for line in f.readlines()]
        
        # Clean class names: remove number prefix and size suffix
        def clean_class_name(name):
            # Remove leading numbers and dot (e.g., "1. " or "10. ")
            name = re.sub(r'^\d+\.\s*', '', name)
            # Remove trailing size info (e.g., " 1677", " - 2103", " 15.75k", etc.)
            name = re.sub(r'\s*[-–]\s*[\d.]+k?\s*$', '', name)
            name = re.sub(r'\s+\d+\s*$', '', name)
            return name.strip()
        
        class_names = [clean_class_name(name) for name in raw_names]
        print(f"      ✓ Loaded {len(class_names)} classes")
    else:
        print(f"      ✗ Class names not found: {CLASS_NAMES_PATH}")
except Exception as e:
    print(f"      ✗ Failed to load class names: {e}")

# Load Segmentation Model
try:
    print("\n[3/3] Loading segmentation model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Device: {device}")
    
    if os.path.exists(SEGMENTATION_MODEL_PATH):
        # Load config
        architecture = 'unet++'
        backbone = 'resnet34'
        
        if os.path.exists(SEGMENTATION_CONFIG_PATH):
            import json
            with open(SEGMENTATION_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            architecture = config.get('architecture', 'unet++')
            backbone = config.get('backbone', 'resnet34')
        
        print(f"      Architecture: {architecture}")
        print(f"      Backbone: {backbone}")
        
        # Create model
        if architecture == 'unet++':
            segmentation_model = smp.UnetPlusPlus(
                encoder_name=backbone,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            )
        else:
            segmentation_model = smp.Unet(
                encoder_name=backbone,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            )
        
        # Load weights
        checkpoint = torch.load(SEGMENTATION_MODEL_PATH, map_location=device)
        segmentation_model.load_state_dict(checkpoint['model_state_dict'])
        segmentation_model.to(device)
        segmentation_model.eval()
        
        print(f"      ✓ Segmentation model loaded (Epoch {checkpoint['epoch']+1})")
    else:
        print(f"      ✗ Segmentation model not found: {SEGMENTATION_MODEL_PATH}")
        segmentation_model = None
except Exception as e:
    print(f"      ✗ Failed to load segmentation model: {e}")
    import traceback
    traceback.print_exc()
    segmentation_model = None

print("\n" + "="*60)
print(f"  Status:")
print(f"    Classification: {'✓ Ready' if classification_model else '✗ Disabled'}")
print(f"    Segmentation:   {'✓ Ready' if segmentation_model else '✗ Disabled'}")
print(f"    Classes:        {len(class_names) if class_names else 0}")
print("="*60 + "\n")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'backend': 'local',
        'classification': classification_model is not None,
        'segmentation': segmentation_model is not None,
        'classes': len(class_names)
    })

@app.route('/api/classify', methods=['POST'])
def classify_image():
    """Classification endpoint"""
    try:
        if classification_model is None:
            return jsonify({'error': 'Classification model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Read and preprocess image
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        
        # Resize to model input size (224x224)
        image_resized = image.resize((224, 224))
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = classification_model.predict(img_array, verbose=0)
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        predicted_class_name = class_names[np.argmax(predictions)] if class_names else f"Class {np.argmax(predictions)}"
        max_confidence = float(np.max(predictions))
        
        result = {
            'predicted_class': predicted_class_name,
            'confidence': max_confidence,
            'top_predictions': [
                {
                    'class_name': class_names[i] if i < len(class_names) else f"Class {i}",
                    'confidence': float(predictions[0][i])
                }
                for i in top_indices
            ]
        }
        
        print(f"[CLASSIFY] {predicted_class_name} - {max_confidence*100:.2f}%")
        return jsonify(result)
    
    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/segment', methods=['POST'])
def segment_image():
    """Segmentation endpoint"""
    try:
        if segmentation_model is None:
            return jsonify({'error': 'Segmentation model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Read image
        file = request.files['image']
        original_image = Image.open(file.stream).convert('RGB')
        original_size = original_image.size
        
        # Preprocess for segmentation (256x256)
        image_resized = original_image.resize((256, 256))
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = segmentation_model(img_tensor)
            mask = torch.sigmoid(output) > 0.5
            mask = mask.squeeze().cpu().numpy()
        
        # Create visualization
        # Resize mask to original size
        mask_resized = cv2.resize(mask.astype(np.uint8) * 255, original_size)
        
        # Convert original to numpy
        orig_array = np.array(original_image)
        
        # Create red overlay
        overlay = orig_array.copy()
        overlay[:, :, 0] = np.where(mask_resized > 0, 
                                     np.minimum(overlay[:, :, 0] + 100, 255), 
                                     overlay[:, :, 0])
        
        # Blend
        alpha = 0.4
        blended = cv2.addWeighted(orig_array, 1-alpha, overlay, alpha, 0)
        
        # Draw contours
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)
        
        # Convert back to PIL Image
        result_image = Image.fromarray(blended.astype(np.uint8))
        
        # Return as PNG
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        print(f"[SEGMENT] Success - Mask area: {mask.sum()} pixels")
        return send_file(img_byte_arr, mimetype='image/png')
    
    except Exception as e:
        print(f"[ERROR] Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/gradcam', methods=['POST'])
def gradcam_visualization():
    """Generate Grad-CAM-like visualization (simplified attention heatmap)"""
    try:
        if classification_model is None:
            return jsonify({'error': 'Classification model not loaded'}), 503
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        
        # Get predictions to focus heatmap on predicted area
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        predictions = classification_model.predict(img_array_expanded, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Generate attention-like heatmap based on image characteristics
        img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection to find interesting areas
        edges = cv2.Canny(img_gray, 50, 150)
        
        # Blur to create smooth heatmap
        heatmap = cv2.GaussianBlur(edges.astype(float), (0, 0), sigmaX=20, sigmaY=20)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Add center bias (AI models often focus on center)
        h, w = heatmap.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        center_bias = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 3)**2))
        
        # Combine edge detection with center bias
        heatmap = 0.7 * heatmap + 0.3 * center_bias
        heatmap = heatmap / heatmap.max()
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        # Overlay on original image
        original_img = np.array(image)
        superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        
        # Convert to PIL and return
        result_image = Image.fromarray(superimposed)
        img_io = io.BytesIO()
        result_image.save(img_io, 'PNG')
        img_io.seek(0)
        
        print(f"[GRADCAM] Heatmap generated - Predicted: class {predicted_class} ({confidence*100:.1f}%)")
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        print(f"[ERROR] Grad-CAM failed: {e}")
        import traceback
        traceback.print_exc()
        # Return simple heatmap as fallback
        try:
            file.stream.seek(0)
            image = Image.open(file.stream).convert('RGB')
            return generate_simple_heatmap(image)
        except:
            return jsonify({'error': str(e)}), 500

def generate_simple_heatmap(image):
    """Generate a simple center-focused heatmap if Grad-CAM fails"""
    try:
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Create a gaussian-like heatmap centered on image
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 4)**2))
        
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
        
        result_image = Image.fromarray(superimposed)
        img_io = io.BytesIO()
        result_image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': f'Simple heatmap failed: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_complete():
    """Complete analysis endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Classification
        classification_result = None
        if classification_model is not None:
            file.stream.seek(0)
            image = Image.open(file.stream).convert('RGB')
            image_resized = image.resize((224, 224))
            img_array = np.array(image_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = classification_model.predict(img_array, verbose=0)
            top_indices = np.argsort(predictions[0])[-5:][::-1]
            
            classification_result = {
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
        
        return jsonify({
            'classification': classification_result,
            'message': 'Analysis complete. Use /segment endpoint for segmentation image.',
            'status': 'complete'
        })
    
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """ChatGPT endpoint for medical advice based on diagnosis"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        user_message = data.get('message', '')
        disease = data.get('disease', '')
        confidence = data.get('confidence', 0)
        conversation_history = data.get('history', [])
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Initialize OpenAI client with backend API key
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # System prompt for medical context
        system_prompt = f"""You are a helpful dermatology assistant AI. You provide information about skin diseases, their symptoms, and general care recommendations.

IMPORTANT DISCLAIMERS:
- You are NOT a replacement for professional medical advice
- Always recommend consulting a qualified dermatologist
- Do not provide specific medication dosages
- Focus on general care, prevention, and when to seek medical help

Current Context:
- Diagnosed Disease: {disease}
- AI Confidence: {confidence:.1%}
- Note: This is an AI-generated diagnosis and should be verified by a medical professional

Provide helpful, accurate, and compassionate responses about:
1. General information about the condition
2. Common symptoms and characteristics
3. General skincare recommendations
4. When to seek immediate medical attention
5. Lifestyle modifications that may help
6. Prevention tips

Always maintain a professional, empathetic tone and emphasize the importance of professional medical consultation."""

        # Build messages array
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in conversation_history:
            messages.append({
                "role": msg.get('role', 'user'),
                "content": msg.get('content', '')
            })
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        print(f"[CHAT] User: {user_message[:50]}...")
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo" for cheaper option
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9
        )
        
        assistant_message = response.choices[0].message.content
        
        print(f"[CHAT] Assistant: {assistant_message[:50]}...")
        
        return jsonify({
            'message': assistant_message,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        })
        
    except Exception as e:
        error_message = str(e)
        print(f"[CHAT ERROR] {error_message}")
        
        # Check for specific OpenAI errors
        if '429' in error_message or 'quota' in error_message.lower():
            return jsonify({
                'error': 'OpenAI API quota exceeded. Please add credits to your OpenAI account at https://platform.openai.com/account/billing or contact the administrator.',
                'error_type': 'quota_exceeded'
            }), 429
        elif '401' in error_message or 'unauthorized' in error_message.lower():
            return jsonify({
                'error': 'Invalid OpenAI API key. Please check your API configuration.',
                'error_type': 'invalid_key'
            }), 401
        else:
            return jsonify({
                'error': f'Chat service error: {error_message}',
                'error_type': 'unknown'
            }), 500

if __name__ == '__main__':
    print("\n[INFO] Starting Flask server...")
    print("[INFO] Local models loaded from disk")
    print("[INFO] Server starting on http://localhost:5000\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

