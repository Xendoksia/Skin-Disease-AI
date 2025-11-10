import { useState, useEffect } from 'react'
import './App.css'
import Demo from './Demo'
import aiVideo from './assets/aivid.mp4'

function App() {
  const [currentPage, setCurrentPage] = useState('home')
  const [isScrolled, setIsScrolled] = useState(false)

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

  const scrollToTop = (e) => {
    e.preventDefault()
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const navigateToDemo = (e) => {
    e.preventDefault()
    setCurrentPage('demo')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const navigateToHome = (e) => {
    e.preventDefault()
    setCurrentPage('home')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  if (currentPage === 'demo') {
    return <Demo onBack={navigateToHome} />
  }

  return (
    <div className="App">
      {/* Header */}
      <header className={`header ${isScrolled ? 'scrolled' : ''}`}>
        <div className="container">
          <div className="logo">
            <span className="logo-icon">🔬</span>
            <span className="logo-text">SkinAI</span>
          </div>
          <nav className="nav">
            <a href="#home" onClick={scrollToTop}>Home</a>
            <a href="#features">Features</a>
            <a href="#about">About</a>
            <a href="#demo" onClick={navigateToDemo}>Try Demo</a>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="hero" id="home">
        <div className="container">
          <div className="hero-content">
            <h1 className="hero-title">
              AI-Powered Skin Disease Detection
            </h1>
            <p className="hero-subtitle">
              Advanced deep learning technology for accurate skin lesion classification and segmentation
            </p>
            <div className="hero-buttons">
              <button className="btn btn-primary" onClick={navigateToDemo}>Get Started</button>
              <button className="btn btn-secondary">Learn More</button>
            </div>
          </div>
          <div className="hero-image">
            <video 
              className="hero-video" 
              autoPlay 
              loop 
              muted 
              playsInline
            >
              <source src={aiVideo} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>
        </div>
      </section>

      {/* Features Section with Scroll Snapping */}
      <section className="section" id="classification">
        <div className="content">
          <div className="feature-content-wrapper">
            <div className="feature-icon">🎯</div>
            <div className="feature-content">
              <h2 className="feature-title">Classification</h2>
              <p className="feature-description">
                Deep learning-powered multi-class skin disease diagnosis system trained on 30,000+ dermatological images. 
                Our MobileNetV2-based architecture achieves 95%+ accuracy across 10 disease categories including 
                Melanoma, Eczema, Basal Cell Carcinoma, Psoriasis, and more. The model utilizes transfer learning 
                with ImageNet pre-trained weights, fine-tuned on dermoscopic images with advanced data augmentation 
                techniques for robust real-world performance.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="section" id="segmentation">
        <div className="content">
          <div className="feature-content-wrapper">
            <div className="feature-icon">✂️</div>
            <div className="feature-content">
              <h2 className="feature-title">Segmentation</h2>
              <p className="feature-description">
                State-of-the-art U-Net architecture with EfficientNet-B0 encoder backbone for precise pixel-level 
                lesion boundary detection. Our segmentation pipeline achieves IoU (Intersection over Union) scores 
                above 0.85, enabling accurate delineation of skin lesion borders. The model employs multi-scale 
                feature extraction with attention mechanisms, trained using Dice Loss and Binary Cross-Entropy 
                for optimal boundary prediction in clinical dermoscopy applications.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="section" id="explainability">
        <div className="content">
          <div className="feature-content-wrapper">
            <div className="feature-icon">🔥</div>
            <div className="feature-content">
              <h2 className="feature-title">Explainability</h2>
              <p className="feature-description">
                Grad-CAM visualization showing exactly which regions influenced the AI diagnosis
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats" id="about">
        <div className="container">
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-number">10</div>
              <div className="stat-label">Disease Classes</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">95%</div>
              <div className="stat-label">Accuracy</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">30K+</div>
              <div className="stat-label">Training Images</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">&lt;2s</div>
              <div className="stat-label">Analysis Time</div>
            </div>
          </div>
          
          <div className="footer-content">
            <p>&copy; 2025 SkinAI. Advanced Medical AI Technology.</p>
          </div>
        </div>
      </section>
    </div>
  )
}

export default App
