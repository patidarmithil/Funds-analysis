import { Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import { useApp } from './context/AppContext'
import Home        from './pages/Home'
import Overview    from './pages/Overview'
import Analysis    from './pages/Analysis'
import Predictions from './pages/Predictions'
import Risk        from './pages/Risk'
import Backtesting from './pages/Backtesting'
import Simulation  from './pages/Simulation'
import Manual      from './pages/Manual'
import NewAnalysis from './pages/NewAnalysis'

const NAV = [
  { label: '🏠 Home',         path: '/' },
  { label: '📊 Overview',     path: '/overview' },
  { label: '📈 Analysis',     path: '/analysis' },
  { label: '🤖 Predictions',  path: '/predictions' },
  { label: '⚠️ Risk',         path: '/risk' },
  { label: '🔁 Backtesting',  path: '/backtesting' },
  { label: '🌀 Simulation',   path: '/simulation' },
  { label: '📚 Manual',       path: '/manual' },
  { label: '🔍 New Analysis', path: '/new-analysis' },
]

export default function App() {
  const { theme, setTheme, dataMode, liveFunds, backendOk } = useApp()
  const navigate  = useNavigate()
  const location  = useLocation()

  // Apply theme to <html>
  document.documentElement.setAttribute('data-theme', theme)

  const liveCount = Object.keys(liveFunds).length

  return (
    <div className="app-wrapper">
      {/* ── Backend offline banner ── */}
      {backendOk === false && (
        <div className="backend-offline" id="backend-offline-banner">
          ⚠️ Backend offline — analytics features disabled. Check FastAPI is running on port 8000.
        </div>
      )}

      {/* ── Header ── */}
      <header className="header">
        <div className="header-top">
          <div className="brand" id="brand-logo">
            <div className="brand-icon">📈</div>
            <div>
              <div className="brand-name">FundScope</div>
              <div className="brand-sub">Professional Mutual Fund Analytics</div>
            </div>
          </div>

          <div className="header-controls">
            {/* Data mode badge */}
            {dataMode === 'default' ? (
              <span className="data-mode-badge badge-default">📂 Sample Report</span>
            ) : liveCount > 0 ? (
              <span className="data-mode-badge badge-live">🟢 Live ({liveCount} fund{liveCount > 1 ? 's' : ''})</span>
            ) : (
              <span className="data-mode-badge badge-warning">🟡 Live (No data)</span>
            )}

            {/* Theme toggle */}
            <button
              className="theme-toggle"
              id="theme-toggle-btn"
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            >
              {theme === 'dark' ? '☀️ Light' : '🌙 Dark'}
            </button>
          </div>
        </div>

        {/* ── Nav tabs ── */}
        <nav className="nav-tabs" role="navigation" aria-label="Main navigation">
          {NAV.map(({ label, path }) => (
            <button
              key={path}
              id={`nav-${path.replace('/', '') || 'home'}`}
              className={`nav-tab${location.pathname === path ? ' active' : ''}`}
              onClick={() => navigate(path)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      {/* ── Pages ── */}
      <main>
        <Routes>
          <Route path="/"             element={<Home />} />
          <Route path="/overview"     element={<Overview />} />
          <Route path="/analysis"     element={<Analysis />} />
          <Route path="/predictions"  element={<Predictions />} />
          <Route path="/risk"         element={<Risk />} />
          <Route path="/backtesting"  element={<Backtesting />} />
          <Route path="/simulation"   element={<Simulation />} />
          <Route path="/manual"       element={<Manual />} />
          <Route path="/new-analysis" element={<NewAnalysis />} />
        </Routes>
      </main>
    </div>
  )
}
