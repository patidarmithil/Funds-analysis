import { useState, useRef, useEffect } from 'react'
import {
  AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend, ResponsiveContainer,
} from 'recharts'
import { useApp } from '../context/AppContext'
import { analyticsAPI } from '../lib/api'
import { PCT_COLORS, PCT_LABELS } from '../utils/chartConfig'
import { drawSimulationPaths } from '../utils/canvasSimulation'
import { fmt } from '../utils/formatters'

const ITER_OPTS = [100, 250, 500, 1000, 2000]

export default function Simulation() {
  const { activeFunds, fundNames } = useApp()
  const [fund,       setFund]       = useState('')
  const [days,       setDays]       = useState(252)
  const [iterations, setIterations] = useState(1000)
  const [investment, setInvestment] = useState(10000)
  const [result,     setResult]     = useState(null)
  const [running,    setRunning]    = useState(false)
  const [error,      setError]      = useState(null)
  const canvasRef = useRef(null)

  // Draw canvas paths whenever result changes
  useEffect(() => {
    if (result?.sample_paths && canvasRef.current) {
      drawSimulationPaths(canvasRef.current, result.sample_paths, {
        surface: 'var(--surface)', muted: '#94a3b8',
      })
    }
  }, [result])

  const run = async () => {
    if (!fund) return
    setRunning(true); setError(null)
    try {
      const { data } = await analyticsAPI.simulate({
        fund_name: fund, data: activeFunds[fund] || [],
        iterations, days, investment,
      })
      setResult(data)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    }
    setRunning(false)
  }

  // Build percentile band chart data
  const bandData = (() => {
    if (!result?.percentile_bands) return []
    const bands = result.percentile_bands
    const len = bands['50']?.length || 0
    return Array.from({ length: len }, (_, i) => {
      const point = { day: i }
      Object.keys(bands).forEach(p => { point[`p${p}`] = Math.round(bands[p][i] * 100) / 100 })
      return point
    })
  })()

  return (
    <div className="page" id="page-simulation">
      <h1 className="page-title">🌀 Monte Carlo Simulation</h1>
      <p className="page-subtitle">Geometric Brownian Motion — scenario path analysis</p>

      <div className="controls">
        <div className="control-group">
          <label>Fund</label>
          <select id="select-fund-sim" value={fund} onChange={e => setFund(e.target.value)}>
            <option value="">— Select fund —</option>
            {fundNames.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div className="control-group">
          <label>Forecast Days: {days}</label>
          <input type="range" id="slider-sim-days" min={30} max={504} value={days}
            onChange={e => setDays(Number(e.target.value))} style={{ minWidth: 150 }} />
        </div>
        <div className="control-group">
          <label>Simulations</label>
          <select id="select-iterations" value={iterations} onChange={e => setIterations(Number(e.target.value))}>
            {ITER_OPTS.map(o => <option key={o} value={o}>{o.toLocaleString()}</option>)}
          </select>
        </div>
        <div className="control-group">
          <label>Investment (₹)</label>
          <input id="input-sim-investment" type="number" value={investment} step={1000}
            onChange={e => setInvestment(Number(e.target.value))} style={{ minWidth: 130 }} />
        </div>
        <button className="btn btn-primary" id="btn-run-simulation" onClick={run} disabled={!fund || running}>
          {running ? '⏳ Simulating…' : '▶ Run Simulation'}
        </button>
      </div>

      {error && <div className="alert alert-danger">⚠ {error}</div>}

      {result && (
        <>
          {/* ── Canvas: 200 grey paths ── */}
          <div className="chart-card">
            <p className="chart-title">{fund} — Monte Carlo ({iterations.toLocaleString()} paths, {days} days) — Grey lines = sample paths</p>
            <canvas
              ref={canvasRef} id="canvas-sim-paths"
              width={900} height={380}
              style={{ width: '100%', height: 380, borderRadius: 8, display: 'block' }}
            />
          </div>

          {/* ── Percentile bands ── */}
          <div className="chart-card">
            <p className="chart-title">Percentile Bands</p>
            <ResponsiveContainer width="100%" height={380}>
              <AreaChart data={bandData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="day" tick={{ fill: 'var(--muted)', fontSize: 10 }} label={{ value: 'Days', fill: 'var(--muted)', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} tickFormatter={v => `₹${v.toFixed(0)}`} />
                <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }}
                  formatter={(v, n) => [fmt.nav(v), n]} />
                <Legend />
                {[95, 75, 50, 25, 5].map(p => (
                  <Area key={p} type="monotone" dataKey={`p${p}`} name={PCT_LABELS[p]}
                    stroke={PCT_COLORS[p]} fill={PCT_COLORS[p]} fillOpacity={p === 75 ? 0.06 : 0}
                    strokeWidth={p === 50 ? 2.5 : 1.5} dot={false} />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* ── Summary KPIs ── */}
          <p className="section-header">Simulation Summary</p>
          <div className="kpi-grid">
            {[
              ['Median Final', fmt.money(result.summary?.median)],
              ['Best Case (95th)', fmt.money(result.summary?.p95)],
              ['Worst Case (5th)', fmt.money(result.summary?.p5)],
              ['Prob. of Profit', `${result.summary?.prob_profit?.toFixed(1)}%`],
              ['Mean Final', fmt.money(result.summary?.mean)],
              ['25th Percentile', fmt.money(result.summary?.p25)],
              ['75th Percentile', fmt.money(result.summary?.p75)],
              ['Prob. >20% Loss', `${result.summary?.prob_loss20?.toFixed(1)}%`],
            ].map(([label, value]) => (
              <div key={label} className="kpi-card">
                <div className="kpi-label">{label}</div>
                <div className="kpi-value" style={{ fontSize: '1rem' }}>{value}</div>
              </div>
            ))}
          </div>

          {/* ── Final value distribution ── */}
          <div className="chart-card">
            <p className="chart-title">Distribution of Final Portfolio Values</p>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={result.final_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="value" tick={{ fill: 'var(--muted)', fontSize: 10 }}
                  tickFormatter={v => `₹${(v/1000).toFixed(0)}k`} />
                <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                <Tooltip formatter={(v, n) => [v.toFixed(7), 'Density']}
                  contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }} />
                <Bar dataKey="density" fill="#00d4ff" opacity={0.75} radius={[2,2,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  )
}
