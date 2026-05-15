import { useState } from 'react'
import {
  ComposedChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, Legend, ResponsiveContainer,
} from 'recharts'
import { useApp } from '../context/AppContext'
import { analyticsAPI } from '../lib/api'
import { fmt } from '../utils/formatters'

export default function Backtesting() {
  const { activeFunds, fundNames } = useApp()
  const [fund,       setFund]       = useState('')
  const [lumpsum,    setLumpsum]    = useState(10000)
  const [monthlySip, setMonthlySip] = useState(1000)
  const [startDate,  setStartDate]  = useState('')
  const [result,     setResult]     = useState(null)
  const [running,    setRunning]    = useState(false)
  const [error,      setError]      = useState(null)

  const run = async () => {
    if (!fund) return
    setRunning(true); setError(null)
    try {
      const { data } = await analyticsAPI.backtest({
        fund_name: fund,
        data: activeFunds[fund] || [],
        lumpsum, monthly_sip: monthlySip,
        start_date: startDate || null,
      })
      setResult(data)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    }
    setRunning(false)
  }

  const navGrowth = result?.nav_growth || []

  return (
    <div className="page" id="page-backtesting">
      <h1 className="page-title">🔁 Backtesting</h1>
      <p className="page-subtitle">Compare Lumpsum vs SIP strategies</p>

      <div className="controls">
        <div className="control-group">
          <label>Fund</label>
          <select id="select-fund-backtest" value={fund} onChange={e => setFund(e.target.value)}>
            <option value="">— Select fund —</option>
            {fundNames.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div className="control-group">
          <label>Lumpsum Investment (₹)</label>
          <input id="input-lumpsum" type="number" value={lumpsum} step={1000}
            onChange={e => setLumpsum(Number(e.target.value))} style={{ minWidth: 140 }} />
        </div>
        <div className="control-group">
          <label>Monthly SIP (₹)</label>
          <input id="input-sip" type="number" value={monthlySip} step={500}
            onChange={e => setMonthlySip(Number(e.target.value))} style={{ minWidth: 140 }} />
        </div>
        <div className="control-group">
          <label>Start Date (optional)</label>
          <input id="input-start-date" type="date" value={startDate}
            onChange={e => setStartDate(e.target.value)} style={{ minWidth: 150 }} />
        </div>
        <button className="btn btn-primary" id="btn-run-backtest" onClick={run} disabled={!fund || running}>
          {running ? '⏳ Running…' : '▶ Run Backtest'}
        </button>
      </div>

      {error && <div className="alert alert-danger">⚠ {error}</div>}

      {result && (
        <>
          {/* ── KPI cards ── */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
            <div className="chart-card" style={{ padding: '1.25rem' }}>
              <p className="chart-title">💰 Lumpsum (Buy & Hold)</p>
              <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
                <div className="kpi-card"><div className="kpi-label">Invested</div><div className="kpi-value" style={{fontSize:'1rem'}}>{fmt.money(result.lumpsum_result?.total_invested)}</div></div>
                <div className="kpi-card"><div className="kpi-label">Final Value</div><div className="kpi-value" style={{fontSize:'1rem', color:'var(--success)'}}>{fmt.money(result.lumpsum_result?.final_value)}</div></div>
                <div className="kpi-card"><div className="kpi-label">Total Return</div><div className="kpi-value" style={{fontSize:'1rem', color: result.lumpsum_result?.total_return > 0 ? 'var(--success)' : 'var(--danger)'}}>{fmt.pct(result.lumpsum_result?.total_return)}</div></div>
                <div className="kpi-card"><div className="kpi-label">CAGR</div><div className="kpi-value" style={{fontSize:'1rem'}}>{fmt.pct(result.lumpsum_result?.cagr)}</div></div>
              </div>
            </div>
            <div className="chart-card" style={{ padding: '1.25rem' }}>
              <p className="chart-title">📅 SIP (Monthly ₹{monthlySip.toLocaleString('en-IN')})</p>
              <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
                <div className="kpi-card"><div className="kpi-label">Invested</div><div className="kpi-value" style={{fontSize:'1rem'}}>{fmt.money(result.sip_result?.total_invested)}</div></div>
                <div className="kpi-card"><div className="kpi-label">Final Value</div><div className="kpi-value" style={{fontSize:'1rem', color:'var(--success)'}}>{fmt.money(result.sip_result?.final_value)}</div></div>
                <div className="kpi-card"><div className="kpi-label">Total Return</div><div className="kpi-value" style={{fontSize:'1rem', color: result.sip_result?.total_return > 0 ? 'var(--success)' : 'var(--danger)'}}>{fmt.pct(result.sip_result?.total_return)}</div></div>
                <div className="kpi-card"><div className="kpi-label">CAGR</div><div className="kpi-value" style={{fontSize:'1rem'}}>{fmt.pct(result.sip_result?.cagr)}</div></div>
              </div>
            </div>
          </div>

          {/* ── Portfolio growth chart ── */}
          <div className="chart-card">
            <p className="chart-title">{fund} — Portfolio Growth (NAV + Lumpsum + SIP)</p>
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={navGrowth}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="ds" tick={{ fill: 'var(--muted)', fontSize: 10 }}
                  tickFormatter={v => v?.slice(0,7)} interval="preserveStartEnd" />
                <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} tickFormatter={v => `₹${(v/1000).toFixed(0)}k`} />
                <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }}
                  formatter={(v, n) => [fmt.money(v), n]} />
                <Legend />
                <Line type="monotone" dataKey="lumpsum_value" name="Lumpsum (B&H)"
                  stroke="#00d4ff" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="sip_value" name="SIP"
                  stroke="#10b981" dot={false} strokeWidth={2} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  )
}
