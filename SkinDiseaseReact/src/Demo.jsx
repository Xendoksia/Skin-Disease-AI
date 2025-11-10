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

  // Real API analysis function
  const analyzeImage = async () => {
    if (!imagePreview) {
      alert("Please upload or capture an image first")
      return
    }

    setIsAnalyzing(true)
    
    try {
      // Convert base64 to blob
      const response = await fetch(imagePreview)
      const blob = await response.blob()
      
      const formData = new FormData()
      formData.append('image', blob, 'image.jpg')
      
      // API base URL
      const API_BASE_URL = 'http://localhost:5000/api'
      
      // Classification
      console.log('Requesting classification...')
      const classifyRes = await fetch(`${API_BASE_URL}/classify`, {
        method: 'POST',
        body: formData
      })
      
      if (!classifyRes.ok) {
        throw new Error(`Classification failed: ${classifyRes.statusText}`)
      }
      
      const classifyData = await classifyRes.json()
      console.log('Classification result:', classifyData)
      
      // Segmentation
      console.log('Requesting segmentation...')
      const segmentFormData = new FormData()
      segmentFormData.append('image', blob, 'image.jpg')
      
      const segmentRes = await fetch(`${API_BASE_URL}/segment`, {
        method: 'POST',
        body: segmentFormData
      })
      
      const segmentBlob = await segmentRes.blob()
      const segmentUrl = URL.createObjectURL(segmentBlob)
      console.log('Segmentation complete')
      
      // Grad-CAM
      console.log('Requesting Grad-CAM...')
      const gradcamFormData = new FormData()
      gradcamFormData.append('image', blob, 'image.jpg')
      
      const gradcamRes = await fetch(`${API_BASE_URL}/gradcam`, {
        method: 'POST',
        body: gradcamFormData
      })
      
      const gradcamBlob = await gradcamRes.blob()
      const gradcamUrl = URL.createObjectURL(gradcamBlob)
      console.log('Grad-CAM complete')
      
      // Generate recommendations based on classification
      const recommendations = generateRecommendations(classifyData)
      
      setResults({
        classification: classifyData,
        segmentation: segmentUrl,
        gradcam: gradcamUrl,
        recommendations: recommendations
      })
      
      console.log('Analysis complete!')
      
    } catch (error) {
      console.error('Error during analysis:', error)
      alert(`Analysis failed: ${error.message}. Please make sure the backend server is running on http://localhost:5000`)
    } finally {
      setIsAnalyzing(false)
    }
  }
  
  // Generate recommendations based on classification results
  const generateRecommendations = (classifyData) => {
    const confidence = classifyData.confidence
    const disease = classifyData.disease
    
    let severity = "Low Risk"
    let severityColor = "low-risk"
    
    if (disease.toLowerCase().includes("melanoma") || disease.toLowerCase().includes("carcinoma")) {
      severity = "High Risk"
      severityColor = "high-risk"
    } else if (confidence < 80) {
      severity = "Medium Risk"
      severityColor = "medium-risk"
    }
    
    const baseRecommendations = [
      "Monitor for any changes in size, shape, or color",
      "Use sunscreen (SPF 30+) to prevent UV damage",
      "Avoid excessive sun exposure during peak hours"
    ]
    
    if (severity === "High Risk") {
      baseRecommendations.unshift("⚠️ Consult a dermatologist immediately for professional evaluation")
      baseRecommendations.push("Consider getting a biopsy if recommended by your doctor")
    } else if (severity === "Medium Risk") {
      baseRecommendations.unshift("Schedule a dermatological check-up within the next month")
      baseRecommendations.push("Take photos to track changes over time")
    } else {
      baseRecommendations.unshift("Schedule routine dermatological check-ups annually")
      baseRecommendations.push("Consider professional photography for baseline documentation")
    }
    
    return {
      severity: severity,
      description: `Based on the AI analysis with ${confidence.toFixed(1)}% confidence, this lesion shows characteristics consistent with ${disease}.`,
      recommendations: baseRecommendations,
      warning: "⚠️ This AI analysis is for informational purposes only and should not replace professional medical advice. Please consult a dermatologist for proper diagnosis and treatment."
    }
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
