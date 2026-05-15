export default function Manual() {
  return (
    <div className="page" id="page-manual" style={{ maxWidth: 800 }}>
      <h1 className="page-title">📚 User Manual</h1>
      <p className="page-subtitle">How to use FundScope</p>

      {[
        ['🏠 Home', 'Landing page. View all 14 sample funds and their current NAV. Click "View Sample Report" to explore pre-loaded analysis, or "Analyse Live Fund" to search and fetch any Indian MF from mfapi.in.'],
        ['📊 Overview', 'Click "Compute Overview" to get a full dashboard across all loaded funds — KPI cards (best/worst 1Y, avg Sharpe, avg drawdown), summary table with 11 metrics, top/worst performer bar charts, normalised NAV trend, and correlation heatmap.'],
        ['📈 Analysis', 'Select a fund and optional benchmark. Run Analysis to get NAV time series, rolling returns (1M/3M/6M/1Y tabs), and CAPM regression metrics (Alpha, Beta, R²).'],
        ['🤖 Predictions', 'Select one or more prediction models and a forecast horizon (30–365 days). Each model trains on 80% of NAV history and evaluates on 20% hold-out. Results: forecast chart with confidence bands, accuracy metrics table (RMSE/MAE/MAPE/R²), optional ensemble optimisation with SLSQP weights.'],
        ['⚠️ Risk Analysis', 'Select multiple funds for side-by-side comparison. Choose confidence level (90/95/99%) and investment amount. 3 tabs: VaR & CVaR (return distribution histogram), Drawdown (underwater chart), Stress Test (bull/bear/crash scenarios).'],
        ['🔁 Backtesting', 'Compare Lumpsum vs SIP strategies. Set investment amounts and optional start date. KPI cards show invested amount, final value, total return, CAGR for each strategy. Combined portfolio growth chart overlay.'],
        ['🌀 Monte Carlo', 'Geometric Brownian Motion simulation. Set forecast days (30–504), number of paths (100–2000), and investment amount. Canvas renders 200 grey sample paths. Recharts shows 5 percentile bands (5/25/50/75/95th). Summary stats + final value distribution histogram.'],
        ['🔍 New Analysis', 'Search any Indian mutual fund by name. Select from results → fetch live NAV history → run analysis. Supports all pages after fetching live data. Session data stored temporarily in Supabase.'],
      ].map(([title, desc]) => (
        <div key={title} style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.4rem', color: 'var(--primary)' }}>{title}</h2>
          <p style={{ color: 'var(--muted)', lineHeight: 1.7, fontSize: '0.88rem' }}>{desc}</p>
        </div>
      ))}

      <hr className="divider" />
      <h2 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.75rem' }}>Data Sources</h2>
      <ul style={{ color: 'var(--muted)', fontSize: '0.85rem', lineHeight: 2, paddingLeft: '1.25rem' }}>
        <li><strong>Sample data:</strong> 14 funds × historical NAV from data.xlsx, stored in Supabase</li>
        <li><strong>Live data:</strong> mfapi.in (free Indian MF NAV API) via Azure FastAPI proxy</li>
        <li><strong>Benchmarks:</strong> Yahoo Finance via yfinance (Nifty 50, Sensex, BSE 500, NASDAQ-100, S&P 500)</li>
      </ul>

      <hr className="divider" />
      <h2 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.75rem' }}>Prediction Models</h2>
      <div className="data-table-wrap">
        <table className="data-table">
          <thead><tr><th>Model</th><th>Type</th><th>Best for</th></tr></thead>
          <tbody>
            {[
              ['Prophet', 'Bayesian additive', 'Funds with clear seasonality/trend'],
              ['ARIMA', 'Statistical time series', 'Stationary or weakly non-stationary NAV'],
              ['Holt-Winters', 'Exponential smoothing', 'Trend + seasonal patterns'],
              ['Linear Regression', 'ML — log-returns', 'Baseline, fast, interpretable'],
              ['Ridge Regression', 'ML — regularised', 'Noisy data, many features'],
              ['Random Forest', 'ML — ensemble tree', 'Non-linear patterns'],
              ['SVR', 'ML — kernel', 'Small datasets, complex patterns'],
              ['XGBoost', 'ML — gradient boosting', 'Best overall ML accuracy'],
            ].map(([m, type, best]) => (
              <tr key={m}><td>{m}</td><td>{type}</td><td>{best}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
