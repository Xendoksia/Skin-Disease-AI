import { useState, useRef, useEffect } from 'react'
import './Demo.css'

function Demo({ onBack }) {
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState(null)
  const [showCamera, setShowCamera] = useState(false)
  const [isScrolled, setIsScrolled] = useState(false)
  const [showChat, setShowChat] = useState(false)
  const [chatMessages, setChatMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [isChatLoading, setIsChatLoading] = useState(false)
  
  const fileInputRef = useRef(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const chatEndRef = useRef(null)

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
      
      // API base URL - use environment variable or default to localhost
      const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api'
      
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
        gradcam: gradcamUrl
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
      const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api'
      alert(`Analysis failed: ${error.message}. Please make sure the backend server is running at ${apiUrl}`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  // Chatbot functions
  const openChat = () => {
    if (!results?.classification) {
      alert('Please analyze an image first to get diagnosis results before using the chatbot.')
      return
    }
    
    setShowChat(true)
    
    // Initialize chat with welcome message if empty
    if (chatMessages.length === 0) {
      const welcomeMessage = {
        role: 'assistant',
        content: `Hello! I'm your dermatology assistant. I can help answer questions about your diagnosis of ${results.classification.predicted_class}. 

Please remember:
- This is an AI-based diagnosis and should be verified by a medical professional
- I can provide general information and care recommendations
- Always consult a dermatologist for professional medical advice

How can I help you today?`
      }
      setChatMessages([welcomeMessage])
    }
  }

  const sendChatMessage = async () => {
    if (!chatInput.trim() || isChatLoading) return
    
    const userMessage = {
      role: 'user',
      content: chatInput
    }
    
    setChatMessages(prev => [...prev, userMessage])
    setChatInput('')
    setIsChatLoading(true)
    
    try {
      const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api'
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: chatInput,
          disease: results.classification.predicted_class,
          confidence: results.classification.confidence,
          history: chatMessages
        })
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Chat request failed')
      }
      
      const data = await response.json()
      
      const assistantMessage = {
        role: 'assistant',
        content: data.message
      }
      
      setChatMessages(prev => [...prev, assistantMessage])
      
      // Scroll to bottom
      setTimeout(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
      
    } catch (error) {
      console.error('Chat error:', error)
      
      // Show user-friendly error message
      let errorMessage = 'Failed to get response from chatbot.'
      
      if (error.message.includes('quota')) {
        errorMessage = '⚠️ OpenAI API quota exceeded!\n\nThe chatbot service has reached its usage limit. Please contact the administrator to add credits at:\nhttps://platform.openai.com/account/billing'
      } else if (error.message.includes('API key')) {
        errorMessage = '🔑 Invalid API key. Please contact the administrator.'
      } else {
        errorMessage = `❌ Chat error: ${error.message}\n\nPlease try again later.`
      }
      
      alert(errorMessage)
    } finally {
      setIsChatLoading(false)
    }
  }

  // Scroll chat to bottom when new messages arrive
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

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
        {/* Modern Upload Section */}
        <div className="upload-section">
          {!imagePreview && !showCamera && (
            <div className="upload-area-modern">
              <div className="upload-dropzone">
                <div className="upload-icon-wrapper">
                  <span className="upload-icon">☁️</span>
                </div>
                <h3 className="upload-title">Drag & Drop Your Image</h3>
                <p className="upload-subtitle">or choose from your device</p>
                
                <div className="upload-options">
                  <button 
                    className="btn-upload-modern"
                    onClick={() => fileInputRef.current.click()}
                  >
                    <span className="btn-icon">📁</span>
                    <span>Browse Files</span>
                  </button>
                  <button 
                    className="btn-camera-modern"
                    onClick={openCamera}
                  >
                    <span className="btn-icon">📷</span>
                    <span>Open Camera</span>
                  </button>
                </div>
                
                <p className="upload-note">Supports: JPG, PNG, JPEG • Max 10MB</p>
                
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  accept="image/*"
                  style={{ display: 'none' }}
                />
              </div>
            </div>
          )}

          {/* Camera View */}
          {showCamera && (
            <div className="camera-view-modern">
              <div className="camera-container">
                <video ref={videoRef} autoPlay playsInline></video>
                <div className="camera-controls">
                  <button className="btn-capture-modern" onClick={capturePhoto}>
                    <span className="capture-ring"></span>
                    <span className="capture-icon">📸</span>
                    Capture Photo
                  </button>
                  <button className="btn-cancel-camera" onClick={() => setShowCamera(false)}>
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Image Preview */}
          {imagePreview && !showCamera && (
            <div className="image-preview-modern">
              <div className="preview-container">
                <div className="preview-image-wrapper">
                  <img src={imagePreview} alt="Preview" />
                  <button 
                    className="btn-remove-image"
                    onClick={() => {
                      setImagePreview(null)
                      setResults(null)
                    }}
                  >
                    ✕
                  </button>
                </div>
                <button 
                  className="btn-analyze-modern"
                  onClick={analyzeImage}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? (
                    <>
                      <span className="analyze-spinner"></span>
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <span className="btn-icon">🔬</span>
                      <span>Analyze Image</span>
                    </>
                  )}
                </button>
              </div>
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

            {/* AI Assistant Card */}
            <div className="result-card ai-assistant-card">
              <h2>💬 Ask AI Assistant</h2>
              <p className="assistant-description">
                Have questions about your diagnosis? Our AI assistant can help answer questions about treatment, symptoms, and care recommendations.
              </p>
              <button 
                className="btn-open-chat"
                onClick={openChat}
              >
                <span className="chat-icon">🤖</span>
                <span>Start Conversation</span>
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Chatbot Window */}
      {showChat && (
        <div className="chatbot-container">
          <div className="chatbot-header">
            <div className="chatbot-title">
              <span className="chatbot-icon">🤖</span>
              <div>
                <h3>Dermatology Assistant</h3>
                <p className="chatbot-subtitle">
                  Discussing: {results?.classification?.predicted_class}
                </p>
              </div>
            </div>
            <button 
              className="chat-close-btn"
              onClick={() => setShowChat(false)}
              title="Close chat"
            >
              ✕
            </button>
          </div>

          <div className="chatbot-messages">
            {chatMessages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-content">
                  <div className="message-text">{msg.content}</div>
                </div>
              </div>
            ))}
            {isChatLoading && (
              <div className="chat-message assistant">
                <div className="message-avatar">🤖</div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="chatbot-input">
            <input
              type="text"
              placeholder="Ask about treatment, symptoms, care tips..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
              disabled={isChatLoading}
            />
            <button 
              onClick={sendChatMessage}
              disabled={!chatInput.trim() || isChatLoading}
              className="send-btn"
            >
              {isChatLoading ? '⏳' : '📤'}
            </button>
          </div>

          <div className="chatbot-footer">
            <small>💡 This chatbot uses GPT-4 for medical information. Always consult a real doctor.</small>
          </div>
        </div>
      )}

      <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
    </div>
  )
}

export default Demo
