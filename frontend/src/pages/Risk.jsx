import { useState } from 'react'
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend,
  ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts'
import { useApp } from '../context/AppContext'
import { analyticsAPI } from '../lib/api'
import { CHART_COLORS } from '../utils/chartConfig'
import { fmt } from '../utils/formatters'

const CONFIDENCE_OPTS = [0.90, 0.95, 0.99]
const TABS = ['📊 VaR & CVaR', '📉 Drawdown', '💥 Stress Test']

export default function Risk() {
  const { activeFunds, fundNames } = useApp()
  const [selected,   setSelected]   = useState([])
  const [confidence, setConfidence] = useState(0.95)
  const [investment, setInvestment] = useState(100000)
  const [results,    setResults]    = useState({})
  const [running,    setRunning]    = useState(false)
  const [tab,        setTab]        = useState(0)
  const [error,      setError]      = useState(null)

  const toggleFund = (f) => setSelected(s => s.includes(f) ? s.filter(x => x !== f) : [...s, f])

  const run = async () => {
    if (!selected.length) return
    setRunning(true); setError(null); setResults({})
    const newR = {}
    for (const fund of selected) {
      try {
        const { data } = await analyticsAPI.risk({
          fund_name: fund, data: activeFunds[fund] || [],
          investment_amount: investment, confidence,
        })
        newR[fund] = data
      } catch (e) { setError(e?.response?.data?.detail || e.message) }
    }
    setResults(newR)
    setRunning(false)
  }

  const resultEntries = Object.entries(results)

  return (
    <div className="page" id="page-risk">
      <h1 className="page-title">⚠️ Risk Analysis</h1>
      <p className="page-subtitle">VaR, CVaR, Drawdown analysis & stress testing</p>

      <div className="controls">
        <div className="control-group" style={{ minWidth: 260 }}>
          <label>Select Funds (multi)</label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {fundNames.map(f => (
              <div key={f} className={`checkbox-item${selected.includes(f) ? ' checked' : ''}`}
                style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }} onClick={() => toggleFund(f)}>
                {f.slice(0,12)}
              </div>
            ))}
          </div>
        </div>
        <div className="control-group">
          <label>Confidence Level</label>
          <select id="select-confidence" value={confidence} onChange={e => setConfidence(Number(e.target.value))}>
            {CONFIDENCE_OPTS.map(c => <option key={c} value={c}>{(c*100).toFixed(0)}%</option>)}
          </select>
        </div>
        <div className="control-group">
          <label>Investment (₹)</label>
          <input id="input-investment" type="number" value={investment} step={10000}
            onChange={e => setInvestment(Number(e.target.value))} style={{ minWidth: 140 }} />
        </div>
        <button className="btn btn-primary" id="btn-run-risk" onClick={run} disabled={!selected.length || running}>
          {running ? '⏳ Running…' : '▶ Run Risk Analysis'}
        </button>
      </div>

      {error && <div className="alert alert-danger">⚠ {error}</div>}

      {resultEntries.length > 0 && (
        <>
          {/* ── Risk metrics summary table ── */}
          <p className="section-header">Risk Metrics Summary</p>
          <div className="data-table-wrap">
            <table className="data-table" id="risk-metrics-table">
              <thead><tr><th>Fund</th><th>VaR 95%</th><th>VaR 99%</th><th>CVaR 95%</th><th>CVaR 99%</th><th>Max DD</th></tr></thead>
              <tbody>
                {resultEntries.map(([f, r]) => (
                  <tr key={f}>
                    <td>{f}</td>
                    <td className="text-danger">{r.var_95 != null ? `${(r.var_95*100).toFixed(2)}%` : '—'}</td>
                    <td className="text-danger">{r.var_99 != null ? `${(r.var_99*100).toFixed(2)}%` : '—'}</td>
                    <td className="text-danger">{r.cvar_95 != null ? `${(r.cvar_95*100).toFixed(2)}%` : '—'}</td>
                    <td className="text-danger">{r.cvar_99 != null ? `${(r.cvar_99*100).toFixed(2)}%` : '—'}</td>
                    <td className="text-danger">{r.max_dd != null ? `${r.max_dd.toFixed(1)}%` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* ── Tabs ── */}
          <div className="tab-bar">
            {TABS.map((t, i) => (
              <button key={t} className={`tab-btn${tab === i ? ' active' : ''}`}
                id={`tab-risk-${i}`} onClick={() => setTab(i)}>{t}</button>
            ))}
          </div>

          {/* VaR tab */}
          {tab === 0 && resultEntries.map(([f, r], fi) => (
            <div key={f}>
              <p className="section-header">{f} — Return Distribution & Risk Thresholds</p>
              <div className="kpi-grid">
                <div className="kpi-card"><div className="kpi-label">Daily VaR ({(confidence*100).toFixed(0)}%)</div>
                  <div className="kpi-value text-danger">{r.var_95 != null ? `${(r.var_95*100).toFixed(2)}%` : '—'}</div></div>
                <div className="kpi-card"><div className="kpi-label">Daily CVaR ({(confidence*100).toFixed(0)}%)</div>
                  <div className="kpi-value text-danger">{r.cvar_95 != null ? `${(r.cvar_95*100).toFixed(2)}%` : '—'}</div></div>
                <div className="kpi-card"><div className="kpi-label">₹ at Risk (daily)</div>
                  <div className="kpi-value">{r.var_95 != null ? fmt.money(Math.abs(r.var_95 * investment)) : '—'}</div></div>
              </div>
              <div className="chart-card">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={r.return_distribution || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="x" tick={{ fill: 'var(--muted)', fontSize: 10 }} tickFormatter={v => `${v.toFixed(1)}%`} />
                    <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                    <Tooltip formatter={(v,n) => [v.toFixed(5), 'Density']}
                      contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }} />
                    <Bar dataKey="density" fill={CHART_COLORS[fi % CHART_COLORS.length]} opacity={0.7} radius={[2,2,0,0]} />
                    {r.var_95 != null && <ReferenceLine x={(r.var_95*100).toFixed(2)} stroke="#ef4444" strokeDasharray="5 3" label={{ value: `VaR`, fill: '#ef4444', fontSize: 10 }} />}
                    {r.cvar_95 != null && <ReferenceLine x={(r.cvar_95*100).toFixed(2)} stroke="#f97316" strokeDasharray="3 3" label={{ value: `CVaR`, fill: '#f97316', fontSize: 10 }} />}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}

          {/* Drawdown tab */}
          {tab === 1 && (
            <div className="chart-card">
              <p className="chart-title">Underwater (Drawdown) Chart</p>
              <ResponsiveContainer width="100%" height={380}>
                <AreaChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="ds" tick={{ fill: 'var(--muted)', fontSize: 10 }}
                    tickFormatter={v => v?.slice(0,7)} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} tickFormatter={v => `${v.toFixed(0)}%`} />
                  <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }} />
                  <Legend />
                  {resultEntries.map(([f, r], i) => (
                    <Area key={f} data={r.drawdown_series || []} type="monotone" dataKey="drawdown"
                      name={f} stroke={CHART_COLORS[i % CHART_COLORS.length]}
                      fill={CHART_COLORS[i % CHART_COLORS.length]} fillOpacity={0.12} strokeWidth={1.5} />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Stress test tab */}
          {tab === 2 && resultEntries.map(([f, r]) => (
            <div key={f}>
              <p className="section-header">{f} — P&L under Stress Scenarios (₹{investment.toLocaleString('en-IN')} invested)</p>
              <div className="chart-card">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={r.stress_test || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="Scenario" tick={{ fill: 'var(--muted)', fontSize: 10 }} />
                    <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} tickFormatter={v => fmt.money(v)} />
                    <Tooltip formatter={(v) => [fmt.money(v), 'P&L']}
                      contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }} />
                    <ReferenceLine y={0} stroke="var(--muted)" strokeDasharray="4 2" />
                    <Bar dataKey="PnL" radius={[4,4,0,0]}>
                      {(r.stress_test || []).map((row, i) => (
                        <Cell key={i} fill={row.PnL >= 0 ? '#10b981' : '#ef4444'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  )
}
