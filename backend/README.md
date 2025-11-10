# Skin Disease Analysis Backend API

## 🚀 Kurulum ve Başlatma

### 1. Python Sanal Ortamı Oluşturma

```bash
# Backend klasörüne git
cd backend

# Sanal ortam oluştur
python -m venv venv

# Sanal ortamı aktifleştir (Windows)
venv\Scripts\activate

# Sanal ortamı aktifleştir (Mac/Linux)
source venv/bin/activate
```

### 2. Gereksinimleri Yükle

```bash
pip install -r requirements.txt
```

### 3. Model Yollarını Kontrol Et

`app.py` dosyasında model yollarının doğru olduğundan emin olun:

```python
CLASSIFICATION_MODEL_PATH = '../model/skin_disease_model.keras'
CLASS_NAMES_PATH = '../model/class_names.txt'
SEGMENTATION_MODEL_PATH = '../models/checkpoints/best_model.pth'
```

### 4. Backend Sunucusunu Başlat

```bash
python app.py
```

Sunucu `http://localhost:5000` adresinde çalışacak.

## 📡 API Endpoints

### Health Check

```
GET http://localhost:5000/api/health
```

### Classification

```
POST http://localhost:5000/api/classify
Content-Type: multipart/form-data
Body: image file
```

### Segmentation

```
POST http://localhost:5000/api/segment
Content-Type: multipart/form-data
Body: image file
Returns: PNG image with overlay
```

### Grad-CAM

```
POST http://localhost:5000/api/gradcam
Content-Type: multipart/form-data
Body: image file
Returns: PNG image with heatmap
```

### Complete Analysis

```
POST http://localhost:5000/api/analyze
Content-Type: multipart/form-data
Body: image file
```

## 🧪 Test Etme

### Postman veya cURL ile Test

```bash
# Health check
curl http://localhost:5000/api/health

# Classification (bir görüntü ile)
curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:5000/api/classify
```

### React Uygulamasından Test

1. Backend sunucusunun çalıştığından emin olun (`python app.py`)
2. React uygulamasını başlatın (`npm run dev`)
3. Demo sayfasına gidin ve bir görüntü yükleyin
4. "Analyze Image" butonuna tıklayın

## 🔧 Sorun Giderme

### CORS Hataları

Backend `flask-cors` kullanıyor, ancak sorun yaşarsanız:

```python
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})
```

### Model Yükleme Hataları

- Model dosyalarının doğru yolda olduğundan emin olun
- TensorFlow ve PyTorch versiyonlarının modellerinizle uyumlu olduğunu kontrol edin

### Port Zaten Kullanımda

Farklı bir port kullanmak için:

```python
app.run(host='0.0.0.0', port=5001, debug=True)
```

React tarafında da API_BASE_URL'i güncelleyin:

```javascript
const API_BASE_URL = "http://localhost:5001/api";
```

## 📦 Üretim Deploy

### Gunicorn ile (Linux/Mac)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker ile

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## 📝 Notlar

- Modeller ilk başlatmada yüklenir (birkaç saniye sürebilir)
- Büyük görüntüler işleme süresini artırabilir
- GPU kullanımı için PyTorch CUDA versiyonunu yükleyin
- Üretimde `debug=False` yapın
