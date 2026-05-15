import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { useApp } from '../context/AppContext'
import { analyticsAPI } from '../lib/api'
import { BENCHMARKS } from '../utils/chartConfig'
import { fmt } from '../utils/formatters'

const TABS = ['1M', '3M', '6M', '1Y']

export default function Analysis() {
  const { activeFunds, fundNames } = useApp()
  const [fund,      setFund]      = useState('')
  const [benchmark, setBenchmark] = useState('')
  const [result,    setResult]    = useState(null)
  const [running,   setRunning]   = useState(false)
  const [error,     setError]     = useState(null)
  const [activeTab, setActiveTab] = useState('1Y')

  const run = async () => {
    if (!fund) return
    setRunning(true); setError(null)
    try {
      const { data } = await analyticsAPI.analysis({
        fund_name: fund,
        data: activeFunds[fund] || [],
        benchmark_ticker: benchmark || null,
      })
      setResult(data)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    }
    setRunning(false)
  }

  const navData    = result?.nav_series || []
  const rollingData = result?.rolling   || []

  return (
    <div className="page" id="page-analysis">
      <h1 className="page-title">📈 Analysis</h1>
      <p className="page-subtitle">NAV trends, rolling returns, Alpha/Beta, benchmark comparison</p>

      <div className="controls">
        <div className="control-group">
          <label>Fund</label>
          <select id="select-fund-analysis" value={fund} onChange={e => setFund(e.target.value)}>
            <option value="">— Select fund —</option>
            {fundNames.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div className="control-group">
          <label>Benchmark (optional)</label>
          <select id="select-benchmark" value={benchmark} onChange={e => setBenchmark(e.target.value)}>
            <option value="">— None —</option>
            {Object.keys(BENCHMARKS).map(b => <option key={b} value={BENCHMARKS[b]}>{b}</option>)}
          </select>
        </div>
        <button className="btn btn-primary" id="btn-run-analysis" onClick={run} disabled={!fund || running}>
          {running ? '⏳ Running…' : '▶ Run Analysis'}
        </button>
      </div>

      {error && <div className="alert alert-danger">⚠ {error}</div>}

      {result && (
        <>
          {/* ── Alpha / Beta / R² KPIs ── */}
          {result.alpha != null && (
            <div className="kpi-grid">
              <div className="kpi-card">
                <div className="kpi-label">Alpha (ann.)</div>
                <div className="kpi-value" style={{ color: result.alpha > 0 ? 'var(--success)' : 'var(--danger)' }}>
                  {(result.alpha * 100).toFixed(2)}%
                </div>
              </div>
              <div className="kpi-card"><div className="kpi-label">Beta</div><div className="kpi-value">{fmt.ratio(result.beta)}</div></div>
              <div className="kpi-card"><div className="kpi-label">R²</div><div className="kpi-value">{fmt.ratio(result.r_squared)}</div></div>
              <div className="kpi-card"><div className="kpi-label">Date Range</div><div className="kpi-value" style={{ fontSize: '0.9rem' }}>{result.date_range?.join(' → ')}</div></div>
            </div>
          )}

          <hr className="divider" />

          {/* ── NAV Time Series ── */}
          <p className="section-header">NAV Time Series</p>
          <div className="chart-card">
            <ResponsiveContainer width="100%" height={360}>
              <LineChart data={navData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="ds" tick={{ fill: 'var(--muted)', fontSize: 10 }}
                  tickFormatter={v => v?.slice(0,7)} interval="preserveStartEnd" />
                <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }}
                  formatter={(v, n) => [fmt.nav(v), n]} labelStyle={{ color: 'var(--muted)' }} />
                <Legend />
                <Line type="monotone" dataKey="y" name={fund} stroke="#00d4ff" dot={false} strokeWidth={2} />
                {result.benchmark_series && (
                  <Line type="monotone" data={result.benchmark_series} dataKey="y"
                    name="Benchmark" stroke="#f59e0b" dot={false} strokeWidth={1.5} strokeDasharray="5 3" />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* ── Rolling Returns (tabs) ── */}
          <p className="section-header">Rolling Returns</p>
          <div className="tab-bar">
            {TABS.map(t => (
              <button key={t} className={`tab-btn${activeTab === t ? ' active' : ''}`}
                id={`tab-rolling-${t}`} onClick={() => setActiveTab(t)}>{t}</button>
            ))}
          </div>
          <div className="chart-card">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={rollingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="ds" tick={{ fill: 'var(--muted)', fontSize: 10 }}
                  tickFormatter={v => v?.slice(0,7)} interval="preserveStartEnd" />
                <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} tickFormatter={v => `${v?.toFixed(0)}%`} />
                <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }}
                  formatter={(v) => [v != null ? `${v.toFixed(2)}%` : '—', `${activeTab} Return`]} />
                <ReferenceLine y={0} stroke="var(--muted)" strokeDasharray="4 2" />
                <Line type="monotone" dataKey={activeTab} name={`${activeTab} Return`}
                  stroke="#00d4ff" dot={false} strokeWidth={1.5} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  )
}
