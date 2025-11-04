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

      {/* Features Section */}
      <section className="features" id="features">
        <div className="container">
          <h2 className="section-title">Our Technology</h2>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">🎯</div>
              <h3>Classification</h3>
              <p>10 different skin disease categories with 95%+ accuracy using MobileNetV2</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">✂️</div>
              <h3>Segmentation</h3>
              <p>U-Net based lesion segmentation with ResNet34 backbone for precise boundary detection</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🔥</div>
              <h3>Explainability</h3>
              <p>Grad-CAM visualization showing exactly which regions influenced the diagnosis</p>
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
