import { useState, useRef, useEffect } from 'react'
import './Demo.css'

function Demo({ onBack }) {
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState(null)
  const [showCamera, setShowCamera] = useState(false)
  const [isScrolled, setIsScrolled] = useState(false)
  const fileInputRef = useRef(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 50) {
        setIsScrolled(true)
      } else {
        setIsScrolled(false)
      }
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  // Handle file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedImage(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result)
      }
      reader.readAsDataURL(file)
      setShowCamera(false)
      setResults(null)
    }
  }

  // Open camera
  const openCamera = async () => {
    setShowCamera(true)
    setResults(null)
    setImagePreview(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    } catch (err) {
      console.error("Error accessing camera:", err)
      alert("Could not access camera. Please check permissions.")
    }
  }

  // Capture photo from camera
  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current
      const video = videoRef.current
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0)
      const imageData = canvas.toDataURL('image/png')
      setImagePreview(imageData)
      
      // Stop camera stream
      const stream = video.srcObject
      const tracks = stream.getTracks()
      tracks.forEach(track => track.stop())
      setShowCamera(false)
    }
  }

  // Mock analysis function (replace with actual API call)
  const analyzeImage = async () => {
    if (!imagePreview) {
      alert("Please upload or capture an image first")
      return
    }

    setIsAnalyzing(true)
    
    // Simulate API call
    setTimeout(() => {
      setResults({
        classification: {
          disease: "Melanocytic Nevi (NV)",
          confidence: 94.7,
          topPredictions: [
            { name: "Melanocytic Nevi (NV)", probability: 94.7 },
            { name: "Melanoma", probability: 3.2 },
            { name: "Benign Keratosis", probability: 1.5 }
          ]
        },
        segmentation: imagePreview, // In real app, this would be the segmented image
        gradcam: imagePreview, // In real app, this would be the Grad-CAM visualization
        recommendations: {
          severity: "Low Risk",
          description: "Based on the AI analysis, this lesion shows characteristics consistent with a benign melanocytic nevus (common mole).",
          recommendations: [
            "Monitor for any changes in size, shape, or color",
            "Schedule routine dermatological check-ups annually",
            "Use sunscreen (SPF 30+) to prevent UV damage",
            "Avoid excessive sun exposure during peak hours",
            "Consider professional photography for baseline documentation"
          ],
          warning: "This AI analysis is for informational purposes only and should not replace professional medical advice. Please consult a dermatologist for proper diagnosis and treatment."
        }
      })
      setIsAnalyzing(false)
    }, 2000)
  }

  return (
    <div className="demo-page">
      {/* Header */}
      <header className={`demo-header ${isScrolled ? 'scrolled' : ''}`}>
        <div className="demo-nav-container">
          <div className="logo">
            <span className="logo-icon">🔬</span>
            <span className="logo-text">SkinAI</span>
          </div>
          <button className="btn-back-nav" onClick={onBack}>
            ← Back to Home
          </button>
        </div>
      </header>

      <div className="demo-container">
        <h1 className="demo-title">Try Our AI Skin Analysis</h1>
        <p className="demo-subtitle">Upload or capture a photo to get instant AI-powered analysis</p>

        {/* Upload/Capture Section */}
        <div className="upload-section">
          <div className="upload-options">
            <button 
              className="btn-upload"
              onClick={() => fileInputRef.current.click()}
            >
              📁 Upload Photo
            </button>
            <button 
              className="btn-camera"
              onClick={openCamera}
            >
              📷 Take Photo
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept="image/*"
              style={{ display: 'none' }}
            />
          </div>

          {/* Camera View */}
          {showCamera && (
            <div className="camera-view">
              <video ref={videoRef} autoPlay playsInline></video>
              <button className="btn-capture" onClick={capturePhoto}>
                📸 Capture
              </button>
            </div>
          )}

          {/* Image Preview */}
          {imagePreview && !showCamera && (
            <div className="image-preview">
              <img src={imagePreview} alt="Preview" />
              <button 
                className="btn-analyze"
                onClick={analyzeImage}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? '🔄 Analyzing...' : '🔬 Analyze Image'}
              </button>
            </div>
          )}
        </div>

        {/* Results Section */}
        {results && (
          <div className="results-section">
            {/* Classification Results */}
            <div className="result-card classification-card">
              <h2>🎯 Classification Results</h2>
              <div className="classification-result">
                <div className="main-prediction">
                  <h3>{results.classification.disease}</h3>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill"
                      style={{ width: `${results.classification.confidence}%` }}
                    ></div>
                  </div>
                  <p className="confidence-text">
                    Confidence: {results.classification.confidence}%
                  </p>
                </div>
                
                <div className="top-predictions">
                  <h4>Top Predictions:</h4>
                  {results.classification.topPredictions.map((pred, index) => (
                    <div key={index} className="prediction-item">
                      <span className="prediction-name">{pred.name}</span>
                      <span className="prediction-prob">{pred.probability}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Segmentation Results */}
            <div className="result-card segmentation-card">
              <h2>✂️ Segmentation Analysis</h2>
              <div className="segmentation-result">
                <img src={results.segmentation} alt="Segmentation" />
                <p className="result-description">
                  Precise boundary detection showing the exact lesion area
                </p>
              </div>
            </div>

            {/* Grad-CAM Visualization */}
            <div className="result-card gradcam-card">
              <h2>🔥 Explainability (Grad-CAM)</h2>
              <div className="gradcam-result">
                <img src={results.gradcam} alt="Grad-CAM" />
                <p className="result-description">
                  Heat map showing which regions influenced the AI decision
                </p>
              </div>
            </div>

            {/* AI Recommendations */}
            <div className="result-card recommendations-card">
              <h2>💡 AI Recommendations</h2>
              <div className="recommendations-content">
                <div className="severity-badge" data-severity={results.recommendations.severity.toLowerCase().replace(' ', '-')}>
                  {results.recommendations.severity}
                </div>
                
                <p className="recommendation-description">
                  {results.recommendations.description}
                </p>

                <div className="recommendation-list">
                  <h4>Recommendations:</h4>
                  <ul>
                    {results.recommendations.recommendations.map((rec, index) => (
                      <li key={index}>{rec}</li>
                    ))}
                  </ul>
                </div>

                <div className="warning-box">
                  <strong>⚠️ Important:</strong> {results.recommendations.warning}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
    </div>
  )
}

export default Demo
