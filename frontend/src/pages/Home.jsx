import { useNavigate } from 'react-router-dom'
import { useApp } from '../context/AppContext'

export default function Home() {
  const { setDataMode } = useApp()
  const navigate = useNavigate()

  return (
    <div className="landing-page" id="page-home">
      
      {/* ── Hero Section ── */}
      <section className="hero-section">
        <h1 className="hero-title">
          The AI-Powered <span className="text-gradient">Analytics Engine</span><br />
          for Mutual Funds.
        </h1>
        <p className="hero-subtitle">
          Predict NAVs, analyze risk, and simulate portfolios with enterprise-grade machine learning models. Built for the modern investor.
        </p>
        
        <div className="hero-cta">
          <button
            className="btn btn-glow"
            id="btn-live-analysis"
            onClick={() => { setDataMode('live'); navigate('/new-analysis') }}
          >
            Start Live Analysis ➔
          </button>
          <button
            className="btn btn-secondary"
            id="btn-sample-report"
            onClick={() => { setDataMode('default'); navigate('/overview') }}
            style={{ padding: '1rem 2rem', fontSize: '1.1rem', borderRadius: '30px' }}
          >
            View Sample Report
          </button>
        </div>

        {/* ── Floating Dashboard Showcase ── */}
        <div className="hero-visual">
          <div className="floating-card fc-1">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <div className="ui-skeleton-line" style={{ width: '120px' }}></div>
              <div className="ui-skeleton-line" style={{ width: '60px' }}></div>
            </div>
            <div className="ui-skeleton-chart"></div>
            <div className="ui-skeleton-line" style={{ width: '80%', marginTop: '1rem' }}></div>
          </div>
          
          <div className="floating-card fc-2">
            <div className="ui-skeleton-line" style={{ width: '80px', marginBottom: '0.5rem' }}></div>
            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#10b981', marginBottom: '1rem' }}>+14.2%</div>
            <div className="ui-skeleton-chart" style={{ borderBottomColor: '#10b981', background: 'linear-gradient(180deg, rgba(16,185,129,0.2) 0%, transparent 100%)' }}></div>
          </div>

          <div className="floating-card fc-3">
            <div className="ui-skeleton-line" style={{ width: '100px', marginBottom: '1rem' }}></div>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#ef4444' }}>-2.1% VaR</div>
            <div className="ui-skeleton-line" style={{ width: '100%', marginTop: 'auto' }}></div>
          </div>
        </div>
      </section>

      {/* ── Feature Grid ── */}
      <section className="feature-grid">
        <div className="feature-box">
          <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>🤖</div>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Machine Learning Predictions</h3>
          <p style={{ color: 'var(--muted)', fontSize: '0.9rem', lineHeight: '1.5' }}>
            Forecast future NAVs using advanced time-series models including Prophet, XGBoost, Random Forest, and custom auto-optimised ensembles.
          </p>
        </div>

        <div className="feature-box">
          <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>⚠️</div>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Deep Risk Analysis</h3>
          <p style={{ color: 'var(--muted)', fontSize: '0.9rem', lineHeight: '1.5' }}>
            Calculate Value at Risk (VaR), Conditional VaR (CVaR), and analyze historical drawdowns to understand maximum portfolio exposure.
          </p>
        </div>

        <div className="feature-box">
          <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>🌀</div>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Monte Carlo Simulations</h3>
          <p style={{ color: 'var(--muted)', fontSize: '0.9rem', lineHeight: '1.5' }}>
            Run thousands of Geometric Brownian Motion (GBM) simulations to visualize probability distributions and future performance bands.
          </p>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="site-footer">
        <div style={{ fontWeight: 'bold', fontSize: '1.2rem', color: 'var(--text)' }}>
          📈 FundScope
        </div>
        <p>Professional Mutual Fund Analytics Engine.</p>
        <div className="footer-links">
          <a href="#">Documentation</a>
          <a href="#">API</a>
          <a href="#">Privacy Policy</a>
          <a href="#">Terms of Service</a>
        </div>
        <p style={{ fontSize: '0.8rem', opacity: 0.6, marginTop: '1rem' }}>
          &copy; {new Date().getFullYear()} FundScope. All rights reserved.
        </p>
      </footer>

    </div>
  )
}
