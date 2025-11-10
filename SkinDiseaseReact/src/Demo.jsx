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
      
      // API base URL
      const API_BASE_URL = 'http://localhost:5000/api'
      
      // Classification
      console.log('Requesting classification...')
      const classifyFormData = new FormData()
      classifyFormData.append('image', blob, 'image.jpg')
      
      const classifyRes = await fetch(`${API_BASE_URL}/classify`, {
        method: 'POST',
        body: classifyFormData
      })
      
      if (!classifyRes.ok) {
        throw new Error(`Classification failed: ${classifyRes.statusText}`)
      }
      
      const classifyData = await classifyRes.json()
      console.log('Classification complete:', classifyData)
      
      // Segmentation
      console.log('Requesting segmentation...')
      const segmentFormData = new FormData()
      segmentFormData.append('image', blob, 'image.jpg')
      
      const segmentRes = await fetch(`${API_BASE_URL}/segment`, {
        method: 'POST',
        body: segmentFormData
      })
      
      if (!segmentRes.ok) {
        throw new Error(`Segmentation failed: ${segmentRes.statusText}`)
      }
      
      const segmentBlob = await segmentRes.blob()
      const segmentUrl = URL.createObjectURL(segmentBlob)
      console.log('Segmentation complete')
      
      // GradCAM
      console.log('Requesting GradCAM...')
      const gradcamFormData = new FormData()
      gradcamFormData.append('image', blob, 'image.jpg')
      
      const gradcamRes = await fetch(`${API_BASE_URL}/gradcam`, {
        method: 'POST',
        body: gradcamFormData
      })
      
      let gradcamUrl = null
      if (gradcamRes.ok) {
        const gradcamBlob = await gradcamRes.blob()
        gradcamUrl = URL.createObjectURL(gradcamBlob)
        console.log('GradCAM complete')
      } else {
        console.warn('GradCAM failed, continuing without it')
      }
      
      setResults({
        classification: classifyData,
        segmentation: segmentUrl,
        gradcam: gradcamUrl,
        recommendations: generateRecommendations(classifyData)
      })
      
      console.log('Results set:', {
        hasClassification: !!classifyData,
        hasSegmentation: !!segmentUrl,
        hasGradcam: !!gradcamUrl,
        classifyData: classifyData
      })
      
      console.log('Analysis complete!')
      
    } catch (error) {
      console.error('Error during analysis:', error)
      alert(`Analysis failed: ${error.message}. Please make sure the backend server is running on http://localhost:5000`)
    } finally {
      setIsAnalyzing(false)
    }
  }
  
  // Generate recommendations based on classification
  const generateRecommendations = (classifyData) => {
    if (!classifyData) {
      return {
        severity: "Analysis Pending",
        description: "Classification data not available",
        recommendations: ["Please ensure the backend is running"],
        warning: "⚠️ This AI analysis is for informational purposes only."
      }
    }

    const disease = classifyData.predicted_class
    const confidence = classifyData.confidence
    
    let severity = "Low Risk"
    let severityColor = "low-risk"
    
    // Determine severity based on disease type and confidence
    const highRiskDiseases = ["Melanoma", "Basal Cell Carcinoma (BCC)"]
    const moderateRiskDiseases = ["Atopic Dermatitis", "Psoriasis pictures Lichen Planus and related diseases"]
    
    if (highRiskDiseases.some(d => disease.includes(d))) {
      severity = confidence > 0.7 ? "High Risk" : "Moderate Risk"
      severityColor = confidence > 0.7 ? "high-risk" : "medium-risk"
    } else if (moderateRiskDiseases.some(d => disease.includes(d))) {
      severity = "Moderate Risk"
      severityColor = "medium-risk"
    } else {
      severity = confidence > 0.8 ? "Low Risk" : "Moderate Risk"
      severityColor = confidence > 0.8 ? "low-risk" : "medium-risk"
    }
    
    const baseRecommendations = [
      `Detection confidence: ${(confidence * 100).toFixed(1)}%`,
      "Schedule a consultation with a dermatologist for professional diagnosis",
      "Monitor for any changes in size, shape, or color",
      "Take clear photos regularly to track progression",
      "Avoid self-diagnosis and self-medication"
    ]
    
    // Disease-specific recommendations
    let specificRecommendations = []
    if (disease.includes("Melanoma")) {
      specificRecommendations = [
        "⚠️ Urgent: Consult a dermatologist immediately",
        "Avoid sun exposure and use high SPF sunscreen",
        "Do not attempt to remove or treat the lesion yourself",
        "Early detection is crucial for successful treatment"
      ]
    } else if (disease.includes("Basal Cell Carcinoma")) {
      specificRecommendations = [
        "Schedule a dermatologist appointment within 1-2 weeks",
        "Protect the area from sun exposure",
        "Document any changes with photos",
        "Treatment is highly effective when caught early"
      ]
    } else if (disease.includes("Eczema") || disease.includes("Atopic Dermatitis")) {
      specificRecommendations = [
        "Keep the area moisturized with fragrance-free products",
        "Avoid known triggers (harsh soaps, allergens, stress)",
        "Consider over-the-counter hydrocortisone cream for mild cases",
        "Consult a dermatologist for persistent or severe symptoms"
      ]
    } else {
      specificRecommendations = [
        "Maintain good skin hygiene",
        "Use sunscreen (SPF 30+) daily",
        "Keep the affected area clean and dry",
        "Avoid scratching or picking at the lesion"
      ]
    }
    
    return {
      severity: severity,
      description: `Detected ${disease} with ${(confidence * 100).toFixed(1)}% confidence. ${
        confidence > 0.7 
          ? "The model is relatively confident in this prediction." 
          : "Confidence level is moderate. Additional professional evaluation recommended."
      }`,
      recommendations: [...baseRecommendations, ...specificRecommendations],
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
            {results.classification && (
              <div className="result-card classification-card">
                <h2>🔍 Classification Results</h2>
                <div className="classification-result">
                  <div className="main-prediction">
                    <h3>{results.classification.predicted_class}</h3>
                    <div className="confidence-bar">
                      <div 
                        className="confidence-fill" 
                        style={{ width: `${results.classification.confidence * 100}%` }}
                      ></div>
                    </div>
                    <p className="confidence-text">
                      Confidence: {(results.classification.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  
                  {results.classification.top_predictions && (
                    <div className="top-predictions">
                      <h4>Other Possibilities:</h4>
                      <ul>
                        {results.classification.top_predictions.slice(1, 4).map((pred, index) => (
                          <li key={index}>
                            <span className="pred-name">{pred.class_name}</span>
                            <span className="pred-conf">{(pred.confidence * 100).toFixed(1)}%</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* GradCAM Visualization */}
            {results.gradcam && (
              <div className="result-card gradcam-card">
                <h2>🎨 AI Attention Map (Grad-CAM)</h2>
                <div className="gradcam-result">
                  <img src={results.gradcam} alt="GradCAM" />
                  <p className="result-description">
                    Heat map showing which areas the AI focused on for classification
                  </p>
                </div>
              </div>
            )}

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
