import { useNavigate } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import { fmt } from '../utils/formatters'

export default function Home() {
  const { fundNames, defaultFunds, loading, setDataMode } = useApp()
  const navigate = useNavigate()

  return (
    <div className="page" id="page-home">
      <h1 className="page-title">📈 FundScope</h1>
      <p className="page-subtitle">Professional Mutual Fund Analytics — AI-powered predictions, risk analysis & backtesting</p>

      {/* ── Quick actions ── */}
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '2rem' }}>
        <button
          className="btn btn-primary"
          id="btn-sample-report"
          onClick={() => { setDataMode('default'); navigate('/overview') }}
        >
          📂 View Sample Report
        </button>
        <button
          className="btn btn-secondary"
          id="btn-live-analysis"
          onClick={() => { setDataMode('live'); navigate('/new-analysis') }}
        >
          🔍 Analyse Live Fund
        </button>
      </div>

      {/* ── Fund list ── */}
      <p className="section-header">Available Sample Funds ({fundNames.length})</p>
      {loading ? (
        <div className="spinner-wrap"><div className="spinner" /></div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '0.75rem' }}>
          {fundNames.map((name, i) => {
            const df = defaultFunds[name]
            const lastNav = df ? df[df.length - 1]?.y : null
            return (
              <div
                key={name}
                className="kpi-card"
                style={{ cursor: 'pointer' }}
                onClick={() => navigate('/analysis')}
              >
                <div className="kpi-label">{name}</div>
                <div className="kpi-value" style={{ fontSize: '1.1rem' }}>{fmt.nav(lastNav)}</div>
              </div>
            )
          })}
        </div>
      )}

      {/* ── Feature highlights ── */}
      <hr className="divider" />
      <p className="section-header">What FundScope does</p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem' }}>
        {[
          ['📊', 'Overview', 'KPI cards, returns table, correlation heatmap'],
          ['📈', 'Analysis', 'Rolling returns, Alpha/Beta, benchmark comparison'],
          ['🤖', 'Predictions', '8 ML models — Prophet, XGBoost, RF, SVR & ensemble'],
          ['⚠️', 'Risk Analysis', 'VaR, CVaR, drawdown chart, stress testing'],
          ['🔁', 'Backtesting', 'Lumpsum vs SIP vs Value Averaging'],
          ['🌀', 'Monte Carlo', 'GBM simulation, percentile bands, probability stats'],
        ].map(([icon, title, desc]) => (
          <div key={title} className="chart-card" style={{ padding: '1rem' }}>
            <div style={{ fontSize: '1.5rem', marginBottom: '0.4rem' }}>{icon}</div>
            <div style={{ fontWeight: 700, marginBottom: '0.25rem' }}>{title}</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--muted)' }}>{desc}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
