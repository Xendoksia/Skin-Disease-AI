import { useState } from 'react'
import './App.css'

function App() {
  const scrollToTop = (e) => {
    e.preventDefault()
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="App">
      {/* Header */}
      <header className="header">
        <div className="container">
          <div className="logo">
            <span className="logo-icon">🔬</span>
            <span className="logo-text">SkinAI</span>
          </div>
          <nav className="nav">
            <a href="#home" onClick={scrollToTop}>Home</a>
            <a href="#features">Features</a>
            <a href="#about">About</a>
            <a href="#demo">Try Demo</a>
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
              <button className="btn btn-primary">Get Started</button>
              <button className="btn btn-secondary">Learn More</button>
            </div>
          </div>
          <div className="hero-image">
            <div className="placeholder-box">
              <span>🏥</span>
              <p>AI Model Preview</p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section with Scroll Snapping */}
      <section className="section" id="classification">
        <div className="content">
          <div className="feature-content-wrapper">
            <div className="feature-icon">🎯</div>
            <h2 className="feature-title">Classification</h2>
            <p className="feature-description">
              10 different skin disease categories with 95%+ accuracy using MobileNetV2 architecture
            </p>
          </div>
        </div>
      </section>

      <section className="section" id="segmentation">
        <div className="content">
          <div className="feature-content-wrapper">
            <div className="feature-icon">✂️</div>
            <h2 className="feature-title">Segmentation</h2>
            <p className="feature-description">
              U-Net based lesion segmentation with ResNet34 backbone for precise boundary detection
            </p>
          </div>
        </div>
      </section>

      <section className="section" id="explainability">
        <div className="content">
          <div className="feature-content-wrapper">
            <div className="feature-icon">🔥</div>
            <h2 className="feature-title">Explainability</h2>
            <p className="feature-description">
              Grad-CAM visualization showing exactly which regions influenced the AI diagnosis
            </p>
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
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p>&copy; 2025 SkinAI. Advanced Medical AI Technology.</p>
        </div>
      </footer>
    </div>
  )
}

export default App
