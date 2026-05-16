import { useState } from 'react'
import {
  ComposedChart, Line, Area, XAxis, YAxis, Tooltip,
  CartesianGrid, Legend, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, Cell,
} from 'recharts'
import { useApp } from '../context/AppContext'
import { analyticsAPI } from '../lib/api'
import { CHART_COLORS, PREDICTION_MODELS } from '../utils/chartConfig'

export default function Predictions() {
  const { activeFunds, fundNames } = useApp()
  const [fund,           setFund]          = useState('')
  const [periods,        setPeriods]        = useState(90)
  const [selected,       setSelected]       = useState(['Prophet', 'XGBoost'])
  const [runEnsemble,    setRunEnsemble]    = useState(false)
  const [results,        setResults]        = useState({})    // { modelName: {forecast, metrics} }
  const [ensemble,       setEnsemble]       = useState(null)
  const [running,        setRunning]        = useState(false)
  const [progress,       setProgress]       = useState(0)
  const [error,          setError]          = useState(null)

  const toggleModel = (m) =>
    setSelected(s => s.includes(m) ? s.filter(x => x !== m) : [...s, m])

  const run = async () => {
    if (!fund || !selected.length) return
    setRunning(true); setError(null); setResults({}); setEnsemble(null); setProgress(0)
    const fundData = activeFunds[fund] || []
    const newResults = {}
    for (let i = 0; i < selected.length; i++) {
      setProgress(Math.round((i / selected.length) * 100))
      try {
        const { data } = await analyticsAPI.predict({
          fund_name: fund, data: fundData, model_name: selected[i], periods,
        })
        newResults[selected[i]] = data
      } catch (_) {}
    }
    setResults(newResults)
    setProgress(100)

    if (runEnsemble && selected.length >= 2) {
      try {
        const { data } = await analyticsAPI.predictEnsemble({
          fund_name: fund, data: fundData, model_names: selected, periods,
        })
        setEnsemble(data)
      } catch (_) {}
    }
    setRunning(false)
  }

  // Build combined chart data
  const historicalData = fund ? (activeFunds[fund] || []).map(r => ({ ds: r.ds, nav: Number(r.y) })) : []
  const todayDs = historicalData.length ? historicalData[historicalData.length - 1].ds : null

  const chartData = [...historicalData]
  const forecastMap = {}
  Object.entries(results).forEach(([modelName, r]) => {
    if (r.forecast) {
      r.forecast.forEach(f => {
        forecastMap[f.ds] = forecastMap[f.ds] || { ds: f.ds }
        forecastMap[f.ds][modelName]         = f.yhat
        forecastMap[f.ds][`${modelName}_lo`] = f.yhat_lower
        forecastMap[f.ds][`${modelName}_hi`] = f.yhat_upper
      })
    }
  })
  const forecastData = Object.values(forecastMap).sort((a,b) => a.ds.localeCompare(b.ds))

  // Metrics table
  const metricsRows = Object.entries(results).map(([m, r]) => ({
    model: m, ...r.metrics,
  })).sort((a, b) => (a.rmse || 999) - (b.rmse || 999))
  const bestRmse = metricsRows[0]?.rmse

  return (
    <div className="page" id="page-predictions">
      <h1 className="page-title">🤖 Predictions & Forecasting</h1>
      <p className="page-subtitle">Select models, tune parameters, compare forecasts</p>

      <div className="controls">
        <div className="control-group">
          <label>Fund</label>
          <select id="select-fund-predict" value={fund} onChange={e => setFund(e.target.value)}>
            <option value="">— Select fund —</option>
            {fundNames.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div className="control-group">
          <label>Forecast Horizon (days): {periods}</label>
          <input type="range" id="slider-periods" min={30} max={365} value={periods}
            onChange={e => setPeriods(Number(e.target.value))} style={{ minWidth: 160 }} />
        </div>
      </div>

      {/* ── Model selection ── */}
      <p className="section-header">Select Models</p>
      <div className="checkbox-grid">
        {PREDICTION_MODELS.map(m => (
          <div key={m} className={`checkbox-item${selected.includes(m) ? ' checked' : ''}`}
            id={`chk-${m.replace(' ', '-')}`} onClick={() => toggleModel(m)}>
            <span></span> {m}
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '1rem' }}>
        <div className={`checkbox-item${runEnsemble ? ' checked' : ''}`} id="chk-ensemble"
          style={{ width: 'auto' }} onClick={() => setRunEnsemble(v => !v)}>
          🔀 Auto-Optimise Ensemble
        </div>
        <button className="btn btn-primary btn-full" id="btn-run-predict"
          onClick={run} disabled={!fund || !selected.length || running} style={{ maxWidth: 200 }}>
          {running ? `⏳ ${progress}%…` : '▶ Run Predictions'}
        </button>
      </div>

      {running && (
        <div className="progress-bar" style={{ marginBottom: '1rem' }}>
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
      )}

      {error && <div className="alert alert-danger">⚠ {error}</div>}

      {/* ── Forecast Chart ── */}
      {Object.keys(results).length > 0 && (
        <>
          <div className="chart-card">
            <p className="chart-title">{fund} — NAV Forecast ({periods} days)</p>
            <ResponsiveContainer width="100%" height={480}>
              <ComposedChart>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="ds" tick={{ fill: 'var(--muted)', fontSize: 10 }}
                  tickFormatter={v => v?.slice(0,7)} interval="preserveStartEnd" />
                <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }}
                  labelStyle={{ color: 'var(--muted)' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {/* Historical NAV */}
                <Line data={historicalData} type="monotone" dataKey="nav" name="Historical NAV"
                  stroke="#94a3b8" dot={false} strokeWidth={1.5} />
                {/* Today vertical line */}
                {todayDs && <ReferenceLine x={todayDs} stroke="#94a3b8" strokeDasharray="4 2" label={{ value: 'Today', fill: '#94a3b8', fontSize: 10 }} />}
                {/* Model forecasts */}
                {Object.keys(results).map((m, i) => {
                  const color = CHART_COLORS[(i + 1) % CHART_COLORS.length]
                  return (
                    <Line key={m} data={forecastData} type="monotone" dataKey={m} name={m}
                      stroke={color} dot={false} strokeWidth={2} />
                  )
                })}
                {/* Ensemble */}
                {ensemble?.ensemble_forecast && (
                  <Line data={ensemble.ensemble_forecast.map(f => ({ ds: f.ds, Ensemble: f.yhat }))}
                    type="monotone" dataKey="Ensemble" name="Ensemble (Optimised)"
                    stroke="#f59e0b" dot={false} strokeWidth={3} strokeDasharray="8 3" />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* ── Metrics table ── */}
          <p className="section-header">📊 Model Accuracy (Hold-out Test Set)</p>
          <div className="data-table-wrap">
            <table className="data-table" id="metrics-table">
              <thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE (%)</th><th>R²</th></tr></thead>
              <tbody>
                {metricsRows.map(r => (
                  <tr key={r.model} className={r.rmse === bestRmse ? 'highlight-row' : ''}>
                    <td>{r.model} {r.rmse === bestRmse ? '🏆' : ''}</td>
                    <td>{r.rmse?.toFixed(4) || '—'}</td>
                    <td>{r.mae?.toFixed(4) || '—'}</td>
                    <td>{r.mape?.toFixed(2) || '—'}</td>
                    <td>{r.r2?.toFixed(4) || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── Ensemble recommendations ── */}
      {ensemble && (
        <>
          <p className="section-header">🎯 Ensemble Recommendations</p>
          <div className="kpi-grid">
            <div className="kpi-card">
              <div className="kpi-label">Best Individual</div>
              <div className="kpi-value" style={{ fontSize: '1rem' }}>{ensemble.recommendations?.best_individual}</div>
              <small className="kpi-sub">RMSE {ensemble.recommendations?.best_individual_rmse?.toFixed(4)}</small>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Ensemble RMSE</div>
              <div className="kpi-value">{ensemble.recommendations?.ensemble_rmse?.toFixed(4)}</div>
              <small className="kpi-sub">{ensemble.recommendations?.improvement_pct != null ? `${ensemble.recommendations.improvement_pct > 0 ? '+' : ''}${ensemble.recommendations.improvement_pct}% vs best` : ''}</small>
            </div>
          </div>

          {/* Ensemble weights bar */}
          {ensemble.weights && (
            <div className="chart-card">
              <p className="chart-title">Optimal Ensemble Weights (SLSQP)</p>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={Object.entries(ensemble.weights).map(([m, w]) => ({ model: m, weight: +(w*100).toFixed(1) }))}>
                  <XAxis dataKey="model" tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                  <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} tickFormatter={v => `${v}%`} />
                  <Tooltip formatter={v => [`${v}%`, 'Weight']}
                    contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }} />
                  <Bar dataKey="weight" radius={[4,4,0,0]}>
                    {Object.keys(ensemble.weights).map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}
    </div>
  )
}
