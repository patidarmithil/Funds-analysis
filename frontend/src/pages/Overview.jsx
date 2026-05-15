import { useState } from 'react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip,
  CartesianGrid, Legend, ResponsiveContainer, Cell,
} from 'recharts'
import { useApp } from '../context/AppContext'
import { analyticsAPI } from '../lib/api'
import { CHART_COLORS } from '../utils/chartConfig'
import { fmt, normalise, colorForValue } from '../utils/formatters'

export default function Overview() {
  const { activeFunds, fundNames, loading } = useApp()
  const [summary, setSummary]         = useState([])
  const [correlation, setCorrelation] = useState({})
  const [running, setRunning]         = useState(false)
  const [error, setError]             = useState(null)

  const run = async () => {
    if (!fundNames.length) return
    setRunning(true); setError(null)
    try {
      const fundsData = {}
      for (const [name, records] of Object.entries(activeFunds)) {
        fundsData[name] = records
      }
      const { data } = await analyticsAPI.overview({ funds_data: fundsData })
      setSummary(data.summary || [])
      setCorrelation(data.correlation_matrix || {})
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    }
    setRunning(false)
  }

  // Normalised NAV data for chart
  const normData = (() => {
    if (!fundNames.length) return []
    const maxLen = Math.max(...fundNames.map(n => (activeFunds[n] || []).length))
    const result = []
    for (let i = 0; i < maxLen; i++) {
      const point = {}
      fundNames.forEach(name => {
        const recs = activeFunds[name] || []
        if (i < recs.length) {
          if (i === 0) point.ds = recs[i].ds
          else if (!point.ds) point.ds = recs[i].ds
          const base = Number(recs[0].y)
          point[name] = Math.round((Number(recs[i].y) / base) * 100 * 100) / 100
        }
      })
      result.push(point)
    }
    return result
  })()

  // Top/bottom 5 from summary
  const sorted1y    = [...summary].filter(r => r.ret_1y != null).sort((a,b) => b.ret_1y - a.ret_1y)
  const top5        = sorted1y.slice(0, 5)
  const bottom5     = sorted1y.slice(-5).reverse()

  // KPIs
  const totalFunds  = summary.length || fundNames.length
  const avgSharpe   = summary.length ? (summary.reduce((s,r) => s + (r.sharpe||0), 0) / summary.length).toFixed(2) : '—'
  const avgDD       = summary.length ? (summary.reduce((s,r) => s + (r.max_drawdown||0), 0) / summary.length * 100).toFixed(1) : '—'
  const best        = sorted1y[0]
  const worst       = sorted1y[sorted1y.length - 1]

  // Correlation heatmap (simple table)
  const corrFunds = Object.keys(correlation)

  if (loading) return <div className="spinner-wrap"><div className="spinner" /></div>

  return (
    <div className="page" id="page-overview">
      <h1 className="page-title">📊 Portfolio Overview</h1>
      <p className="page-subtitle">Live snapshot across all tracked mutual funds</p>

      <button className="btn btn-primary" id="btn-run-overview" onClick={run} disabled={running}>
        {running ? '⏳ Computing…' : '▶ Compute Overview'}
      </button>

      {error && <div className="alert alert-danger" style={{ marginTop: '1rem' }}>⚠ {error}</div>}

      {/* ── KPI Cards ── */}
      <div className="kpi-grid" style={{ marginTop: '1.5rem' }}>
        <div className="kpi-card"><div className="kpi-label">Total Funds</div><div className="kpi-value">{totalFunds}</div></div>
        <div className="kpi-card">
          <div className="kpi-label">Best Fund (1Y)</div>
          <div className="kpi-value" style={{ color: 'var(--success)' }}>{best ? fmt.pct(best.ret_1y) : '—'}</div>
          <small className="kpi-sub">{best ? fmt.short(best.fund) : ''}</small>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Worst Fund (1Y)</div>
          <div className="kpi-value" style={{ color: 'var(--danger)' }}>{worst ? fmt.pct(worst.ret_1y) : '—'}</div>
          <small className="kpi-sub">{worst ? fmt.short(worst.fund) : ''}</small>
        </div>
        <div className="kpi-card"><div className="kpi-label">Avg Sharpe</div><div className="kpi-value">{avgSharpe}</div></div>
        <div className="kpi-card"><div className="kpi-label">Avg Max Drawdown</div><div className="kpi-value">{avgDD !== '—' ? `-${avgDD}%` : '—'}</div></div>
      </div>

      <hr className="divider" />

      {/* ── Summary table ── */}
      {summary.length > 0 && (
        <>
          <p className="section-header">Fund Summary Table</p>
          <div className="data-table-wrap">
            <table className="data-table" id="summary-table">
              <thead>
                <tr>
                  <th>Fund</th><th>NAV</th><th>1M%</th><th>3M%</th>
                  <th>6M%</th><th>1Y%</th><th>CAGR</th>
                  <th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Vol</th>
                </tr>
              </thead>
              <tbody>
                {summary.map(r => (
                  <tr key={r.fund}>
                    <td>{r.fund}</td>
                    <td>{fmt.nav(r.current_nav)}</td>
                    <td className={r.ret_1m > 0 ? 'text-success' : 'text-danger'}>{fmt.pct(r.ret_1m)}</td>
                    <td className={r.ret_3m > 0 ? 'text-success' : 'text-danger'}>{fmt.pct(r.ret_3m)}</td>
                    <td className={r.ret_6m > 0 ? 'text-success' : 'text-danger'}>{fmt.pct(r.ret_6m)}</td>
                    <td className={r.ret_1y > 0 ? 'text-success' : 'text-danger'}>{fmt.pct(r.ret_1y)}</td>
                    <td className={r.cagr > 0 ? 'text-success' : 'text-danger'}>{r.cagr != null ? `${(r.cagr*100).toFixed(1)}%` : '—'}</td>
                    <td className={r.sharpe > 0 ? 'text-success' : 'text-danger'}>{fmt.ratio(r.sharpe)}</td>
                    <td className={r.sortino > 0 ? 'text-success' : 'text-danger'}>{fmt.ratio(r.sortino)}</td>
                    <td className="text-danger">{r.max_drawdown != null ? `${(r.max_drawdown*100).toFixed(1)}%` : '—'}</td>
                    <td>{r.volatility != null ? `${(r.volatility*100).toFixed(1)}%` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <hr className="divider" />
        </>
      )}

      {/* ── Top & Worst performers ── */}
      {top5.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
          <div className="chart-card">
            <p className="chart-title">🏆 Top Performers (1Y)</p>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={top5} layout="vertical" margin={{ left: 80 }}>
                <XAxis type="number" tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                <YAxis type="category" dataKey="fund" tick={{ fill: 'var(--text)', fontSize: 11 }}
                  width={80} tickFormatter={v => v.slice(0,10)} />
                <Tooltip formatter={(v) => [`${v.toFixed(1)}%`, '1Y Return']}
                  contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }} />
                <Bar dataKey="ret_1y" radius={[0,4,4,0]}>
                  {top5.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-card">
            <p className="chart-title">📉 Worst Performers (1Y)</p>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={bottom5} layout="vertical" margin={{ left: 80 }}>
                <XAxis type="number" tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                <YAxis type="category" dataKey="fund" tick={{ fill: 'var(--text)', fontSize: 11 }}
                  width={80} tickFormatter={v => v.slice(0,10)} />
                <Tooltip formatter={(v) => [`${v.toFixed(1)}%`, '1Y Return']}
                  contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }} />
                <Bar dataKey="ret_1y" radius={[0,4,4,0]}>
                  {bottom5.map((_, i) => <Cell key={i} fill="#ef4444" />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── Normalised NAV chart ── */}
      <div className="chart-card">
        <p className="chart-title">📈 Normalised NAV Trends (Base = 100)</p>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={normData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="ds" tick={{ fill: 'var(--muted)', fontSize: 10 }}
              tickFormatter={v => v?.slice(0,7)} interval="preserveStartEnd" />
            <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }}
              labelStyle={{ color: 'var(--muted)' }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {fundNames.map((name, i) => (
              <Line key={name} type="monotone" dataKey={name}
                stroke={CHART_COLORS[i % CHART_COLORS.length]} dot={false} strokeWidth={1.5} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── Correlation heatmap (simple table) ── */}
      {corrFunds.length > 0 && (
        <>
          <p className="section-header">🔗 Return Correlation Matrix</p>
          <div className="data-table-wrap">
            <table className="data-table" id="correlation-table">
              <thead>
                <tr>
                  <th>Fund</th>
                  {corrFunds.map(f => <th key={f} title={f}>{f.slice(0,8)}</th>)}
                </tr>
              </thead>
              <tbody>
                {corrFunds.map(row => (
                  <tr key={row}>
                    <td title={row}>{row.slice(0,14)}</td>
                    {corrFunds.map(col => {
                      const v = correlation[col]?.[row]
                      const bg = v != null
                        ? `rgba(${v > 0 ? '0,212,255' : '239,68,68'},${Math.abs(v) * 0.35})`
                        : 'transparent'
                      return (
                        <td key={col} style={{ background: bg, textAlign: 'center', fontWeight: row===col ? 700 : 400 }}>
                          {v != null ? v.toFixed(2) : '—'}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
