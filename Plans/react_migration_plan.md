# FundScope — React Migration Plan (v2)

> **Updated:** May 2026 — incorporates 5 architectural decisions below.

---

## 5 Architectural Decisions (What Changed from v1)

| # | Decision | Impact |
|---|----------|--------|
| 1 | **Universal pre-trained model.pkl** — one big model trained on all funds across many years; no per-request retraining | `model_cache.py` redesigned; `predict` endpoint uses `.pkl` features only |
| 2 | **React charts = Streamlit charts 1:1** — same chart types, colors, data shapes | Recharts component mapping documented per page |
| 3 | **data.xlsx sample always available** — user sees sample report from `data.xlsx`; new fund calls model.pkl feature API then prints on frontend | Two modes preserved: Sample (xlsx) and Live (mfapi) |
| 4 | **Supabase for static/slow data** — fund metadata, cached analytics, xlsx NAV data stored in Supabase; Azure only runs heavy ML | Supabase schema defined below |
| 5 | **Dual port config** — `.env.local` for local dev, Vercel env vars for prod; all services have both addresses documented | Port table below |

---

## Port Reference Table

| Service | Local Dev | Production |
|---------|-----------|------------|
| FastAPI backend | `http://localhost:8000` | `https://fundscopebackend-gbeybdd2gcd3egez.southeastasia-01.azurewebsites.net` |
| React frontend (Vite) | `http://localhost:5173` | `https://<your-vercel-app>.vercel.app` |
| Streamlit (legacy, keep during transition) | `http://localhost:8501` | `https://fundscopefront.streamlit.app` |
| Supabase API | `https://<project>.supabase.co` | same (always cloud) |

**React `.env.local` (gitignored):**
```env
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=https://<project>.supabase.co
VITE_SUPABASE_ANON_KEY=<anon-key>
```

**React Vercel env vars (dashboard):**
```env
VITE_API_URL=https://fundscopebackend-gbeybdd2gcd3egez.southeastasia-01.azurewebsites.net
VITE_SUPABASE_URL=https://<project>.supabase.co
VITE_SUPABASE_ANON_KEY=<anon-key>
```

**FastAPI `.env` (local + Azure App Service):**
```env
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_KEY=<service-role-key>
PORT=8000
```

---

## Current Architecture

```
FundScope/
├── frontend/          <- Streamlit (Python) — UI + ML computation mixed
│   ├── app.py
│   ├── config.py
│   ├── data.xlsx      <- 14 funds sample data
│   ├── modules/       <- ALL ML computation lives here
│   └── app_pages/     <- 9 Streamlit pages
└── backend/           <- FastAPI on Azure — only mfapi proxy currently
    ├── main.py
    └── routers/mf_router.py
```

---

## Target Architecture

```
FundScope/
├── backend/                          <- FastAPI — Azure (ML compute)
│   ├── main.py
│   ├── routers/
│   │   ├── mf_router.py              (existing)
│   │   └── analytics_router.py       (NEW)
│   ├── services/
│   │   ├── mf_service.py             (existing)
│   │   ├── analytics_service.py      (NEW)
│   │   ├── model_cache.py            (NEW — universal pkl loader)
│   │   ├── data_service.py           (NEW — xlsx parser)
│   │   └── supabase_service.py       (NEW — cache reads/writes)
│   ├── modules/                      (MOVED from frontend/)
│   │   ├── analysis.py
│   │   ├── predictions.py
│   │   ├── risk.py
│   │   ├── simulation.py
│   │   ├── backtesting.py
│   │   └── data_loader.py
│   ├── models/
│   │   └── universal_fund_model.pkl  <- THE ONE BIG MODEL (trained offline)
│   ├── config.py                     (MOVED, Plotly blocks stripped)
│   └── data.xlsx                     (MOVED — backend serves it)
│
├── react-frontend/                   <- Vite + React — Vercel
│   ├── src/
│   │   ├── pages/         (9 pages)
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── utils/
│   │   ├── lib/           (api.js, supabaseClient.js)
│   │   └── context/AppContext.jsx
│   ├── .env.local         (gitignored)
│   └── vite.config.js
│
└── supabase/                         <- Upload to Supabase once
    ├── schema.sql
    ├── fund_metadata.json
    └── default_fund_data.json        (parsed from data.xlsx)
```

---

## Phase 1 — Supabase Setup (What Goes There)

### Why Supabase?
Azure App Service has request/compute limits. Supabase handles **read-heavy, static or slow-changing data** so Azure only triggers when ML is actually needed.

### Supabase Tables (supabase/schema.sql)

```sql
-- 1. Fund metadata (static, 14 funds from config.py)
CREATE TABLE fund_metadata (
  id            SERIAL PRIMARY KEY,
  fund_name     TEXT UNIQUE NOT NULL,
  scheme_code   TEXT,
  category      TEXT,
  fund_house    TEXT,
  benchmark     TEXT,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Default NAV data (from data.xlsx, seeded once)
CREATE TABLE default_fund_nav (
  id        SERIAL PRIMARY KEY,
  fund_name TEXT NOT NULL,
  date      DATE NOT NULL,
  nav       NUMERIC(12,4) NOT NULL,
  UNIQUE(fund_name, date)
);

-- 3. Analytics results cache (avoids repeated Azure calls)
CREATE TABLE analytics_cache (
  id            SERIAL PRIMARY KEY,
  fund_name     TEXT NOT NULL,
  analysis_type TEXT NOT NULL,   -- 'summary','risk','backtest','simulate','predict'
  params_hash   TEXT NOT NULL,   -- hash of input params
  result_json   JSONB NOT NULL,
  computed_at   TIMESTAMPTZ DEFAULT NOW(),
  expires_at    TIMESTAMPTZ,     -- NULL = never expire
  UNIQUE(fund_name, analysis_type, params_hash)
);

-- 4. Live fund session data (temporary, per user session)
CREATE TABLE live_fund_sessions (
  id         SERIAL PRIMARY KEY,
  session_id TEXT NOT NULL,
  fund_name  TEXT NOT NULL,
  nav_data   JSONB NOT NULL,   -- [{ds, y}]
  fetched_at TIMESTAMPTZ DEFAULT NOW()
);
```

### What React Calls Directly from Supabase (no Azure load)

| Data | Supabase Table | React hook |
|------|---------------|------------|
| Default 14-fund NAV series | `default_fund_nav` | `useDefaultFunds()` |
| Fund names + metadata | `fund_metadata` | `useFundMeta()` |
| Cached analytics (if fresh) | `analytics_cache` | inside analytics hooks |

### What Still Goes Through Azure Backend

| Endpoint | Why Azure |
|----------|-----------|
| `POST /analytics/predict` | Universal pkl inference |
| `POST /analytics/simulate` | Monte Carlo — CPU intensive |
| `POST /analytics/risk` | scipy computations |
| `POST /analytics/backtest` | Pandas heavy computation |
| `POST /analytics/overview` | Multi-fund metrics computation |
| `GET /mf/fetch-batch` | mfapi.in live calls |
| `POST /analytics/data/upload` | xlsx parsing |

---

## Phase 2 — Backend: Move ML to FastAPI

### Step 2.1 — Move modules/ to backend

Copy `frontend/modules/` → `backend/modules/`

| File | Changes needed |
|------|---------------|
| `data_loader.py` | Remove `import streamlit`, `@st.cache_data`, `st.error()`, `st.warning()` |
| `analysis.py` | No st.* — works as-is |
| `predictions.py` | No st.* — works as-is |
| `risk.py` | Import `STRESS_SCENARIOS` from local `config.py` |
| `simulation.py` | Check for any st.* usage |
| `backtesting.py` | Check for any st.* usage |

Also move `frontend/config.py` → `backend/config.py`. Remove `PLOTLY_DARK`, `PLOTLY_LIGHT`, `PLOTLY_LAYOUT`, `hex_to_rgba` — those are frontend concerns now.

### Step 2.2 — Universal model.pkl Strategy (Decision #1)

> **Key change from v1:** ONE pre-trained model replaces per-fund pkl files. Trained offline on 100+ funds × 10+ years. For any fund (including new ones), extract features from NAV series → call `.predict()`. Zero retraining per request.

```
backend/models/
└── universal_fund_model.pkl    <- trained ONCE offline, uploaded to Azure
```

**Services/model_cache.py:**
```python
import joblib, numpy as np, pandas as pd
from pathlib import Path

_MODEL = None   # loaded once at startup

def load_universal_model():
    global _MODEL
    if _MODEL is None:
        pkl_path = Path("models/universal_fund_model.pkl")
        if pkl_path.exists():
            _MODEL = joblib.load(pkl_path)
        else:
            _MODEL = _build_placeholder_model()
    return _MODEL

def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Same feature set the universal model was trained on."""
    y = df['y'].values
    features = {
        'rolling_5d':   pd.Series(y).rolling(5).mean().iloc[-1],
        'rolling_20d':  pd.Series(y).rolling(20).mean().iloc[-1],
        'rolling_60d':  pd.Series(y).rolling(60).mean().iloc[-1],
        'volatility_20d': pd.Series(y).rolling(20).std().iloc[-1],
        'momentum_14d': y[-1] / y[-14] - 1 if len(y) > 14 else 0,
        'nav_zscore':   (y[-1] - np.mean(y)) / (np.std(y) + 1e-9),
    }
    return np.array(list(features.values())).reshape(1, -1)

def predict_with_universal_model(df, periods):
    model = load_universal_model()
    features = extract_features(df)
    # Model predicts next N days relative returns; apply to last NAV
    rel_returns = model.predict_sequence(features, periods)
    last_nav = df['y'].iloc[-1]
    yhat = last_nav * np.cumprod(1 + rel_returns)
    return build_forecast_df(df, yhat, periods)

def _build_placeholder_model():
    """Fallback until real universal model is ready."""
    from sklearn.linear_model import LinearRegression
    class PlaceholderModel:
        def predict_sequence(self, features, periods):
            drift = 0.0002   # small daily drift
            noise = np.random.normal(drift, 0.005, periods)
            return noise
    return PlaceholderModel()
```

**Later (separate task):**
- Train `universal_fund_model.pkl` on 100+ funds × 10+ years
- Upload to `backend/models/` on Azure via deployment
- No code changes needed — `model_cache.py` interface stays identical

### Step 2.3 — Analytics Endpoints (analytics_router.py)

```
POST /analytics/fund-summary
    Body:    { fund_name, data: [{ds, y}] }
    Returns: { current_nav, cagr, sharpe, sortino, calmar,
               max_drawdown, volatility, ret_1m, ret_3m, ret_6m, ret_1y }

POST /analytics/overview
    Body:    { funds_data: { fund_name: [{ds, y}] } }
    Returns: { summary: [ {fund, metrics...} ],
               correlation_matrix: { fund: { fund: value } } }

POST /analytics/analysis
    Body:    { fund_name, data, benchmark_ticker?, date_range? }
    Returns: { nav_series, rolling_1m, rolling_3m, rolling_6m, rolling_1y,
               alpha, beta, r2, benchmark_series }

POST /analytics/predict
    Body:    { fund_name, data: [{ds, y}], model_name, periods }
    Returns: { forecast: [{ds, yhat, yhat_lower, yhat_upper}],
               metrics: {rmse, mae, mape, r2},
               model_used: "universal" | "fallback" }
    Note:    uses universal_fund_model.pkl — no retraining

POST /analytics/predict-ensemble
    Body:    { fund_name, data, model_names: [], periods }
    Returns: { ensemble_forecast, weights, model_metrics,
               recommendations: {best_individual, ensemble_rmse,
                                 improvement_pct, top_weights} }

POST /analytics/risk
    Body:    { fund_name, data, investment_amount, confidence }
    Returns: { var_95, var_99, cvar_95, cvar_99, max_drawdown,
               drawdown_series: [{ds, drawdown}],
               return_distribution: [{x, density}],
               stress_test: [{scenario, pnl, pct_change}],
               recovery_periods: [{start, end, depth, duration}] }

POST /analytics/backtest
    Body:    { fund_name, data, lumpsum, monthly_sip, start_date }
    Returns: { lumpsum_result: {invested, final_value, cagr, abs_return},
               sip_result: {invested, final_value, xirr, abs_return},
               nav_growth: [{ds, nav, lumpsum_value, sip_value}] }

POST /analytics/simulate
    Body:    { fund_name, data, iterations, days, investment }
    Returns: { percentile_bands: {p5,p25,p50,p75,p95}: [nav,...],
               summary: {median,mean,p5,p25,p75,p95,prob_profit,prob_loss20},
               final_distribution: [{value, density}],
               sample_paths: [[nav,...], ...] }   <- 200 paths only

GET /analytics/data/default
    Returns: { fund_name: [{ds, y}] }  <- fallback if Supabase unavailable

POST /analytics/data/upload
    Body:    multipart/form-data { file: xlsx }
    Returns: { fund_name: [{ds, y}] }

GET /analytics/health
    Returns: { status, model_loaded, supabase_ok, default_funds_count }
```

### Step 2.4 — Supabase Service (backend)

```python
# backend/services/supabase_service.py
import os, hashlib, json
from supabase import create_client
from datetime import datetime, timedelta

_client = None
def get_client():
    global _client
    if not _client:
        _client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    return _client

def get_cached(fund_name, analysis_type, params: dict, max_age_hours=24):
    params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    row = get_client().table("analytics_cache")\
        .select("result_json, computed_at")\
        .eq("fund_name", fund_name).eq("analysis_type", analysis_type)\
        .eq("params_hash", params_hash).single().execute()
    if row.data:
        age = datetime.utcnow() - datetime.fromisoformat(row.data["computed_at"])
        if age < timedelta(hours=max_age_hours):
            return row.data["result_json"]
    return None

def set_cached(fund_name, analysis_type, params: dict, result: dict):
    params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    get_client().table("analytics_cache").upsert({
        "fund_name": fund_name, "analysis_type": analysis_type,
        "params_hash": params_hash, "result_json": result,
    }).execute()
```

### Step 2.5 — CORS update (main.py)

```python
allow_origins=[
    "http://localhost:5173",                          # Vite dev
    "http://localhost:8501",                          # Streamlit legacy
    "https://fundscopefront.streamlit.app",
    "https://<your-vercel-app>.vercel.app",           # update when known
]
```

---

## Phase 3 — React Frontend

### Step 3.1 — Setup

```bash
cd d:/Coding/Project/Finance/FundScope
npx create-vite@latest react-frontend -- --template react
cd react-frontend
npm install axios recharts react-router-dom lucide-react @supabase/supabase-js
```

### Step 3.2 — Global State (AppContext.jsx)

```javascript
{
  dataMode: "default" | "live",
  defaultFunds: { [fundName]: [{ds, y}] },   // from Supabase
  liveFunds:    { [fundName]: [{ds, y}] },   // from /mf/fetch-batch
  selectedFunds: [],
  theme: "dark" | "light",                   // localStorage persist
  dateRange: [start, end]
}
```

### Step 3.3 — Pages

| Streamlit Page | React Route | Supabase direct? | Azure calls |
|---------------|-------------|-----------------|-------------|
| Home | `/` | fund_metadata | none |
| Overview | `/overview` | default_fund_nav | `/analytics/overview` |
| Analysis | `/analysis` | default_fund_nav | `/analytics/analysis` |
| Predictions | `/predictions` | default_fund_nav | `/analytics/predict` |
| Risk Analysis | `/risk` | default_fund_nav | `/analytics/risk` |
| Backtesting | `/backtesting` | default_fund_nav | `/analytics/backtest` |
| Simulation | `/simulation` | default_fund_nav | `/analytics/simulate` |
| User Manual | `/manual` | none (static) | none |
| New Analysis | `/new-analysis` | cache check | `/mf/search`, `/mf/fetch-batch`, `/analytics/predict` |

### Step 3.4 — Chart Strategy: Match Streamlit Exactly (Decision #2)

**Color palette (exact from config.py):**
```javascript
// src/utils/chartConfig.js
export const CHART_COLORS = [
  '#00d4ff','#7c3aed','#10b981','#f59e0b',
  '#ef4444','#f97316','#8b5cf6','#ec4899',
  '#06b6d4','#84cc16','#14b8a6','#6366f1','#fb923c','#a855f7',
];
export const DARK = {
  bg:'#0a0e1a', surface:'#0f1629', surface2:'#162040',
  border:'#1e2d4d', text:'#e2e8f0', muted:'#94a3b8', primary:'#00d4ff',
};
export const LIGHT = {
  bg:'#f0f4f8', surface:'#ffffff', surface2:'#e8eef6',
  border:'#cbd5e1', text:'#0f172a', muted:'#475569', primary:'#0284c7',
};
```

**Streamlit Plotly → Recharts mapping:**

| Page | Streamlit chart | Recharts component |
|------|-----------------|--------------------|
| Overview | Styled DataFrame | `<table>` + conditional color (green/red) |
| Overview | Top performers hbar | `<BarChart layout="vertical">` teal |
| Overview | Worst performers hbar | `<BarChart layout="vertical">` red |
| Overview | Normalised NAV all funds | `<LineChart>` 14 series, CHART_COLORS |
| Overview | Correlation heatmap | Custom SVG grid (react-heat-map) |
| Analysis | NAV time series | `<LineChart>` date x-axis |
| Analysis | Rolling returns 4 tabs | `<LineChart>` × 4 tabs |
| Predictions | Historical + forecast | `<ComposedChart>` `<Line>` + `<Area>` CI |
| Predictions | "Today" vertical line | `<ReferenceLine x={today}>` |
| Predictions | Model metrics table | `<table>` best-RMSE row highlighted |
| Predictions | Ensemble weights bar | `<BarChart>` blues |
| Risk | Return distribution | `<BarChart>` histnorm + normal `<Line>` |
| Risk | VaR/CVaR lines | `<ReferenceLine>` dashed |
| Risk | Drawdown underwater | `<AreaChart>` fill toZero per fund |
| Risk | Stress test bars | `<BarChart>` color by PnL value |
| Backtesting | NAV + SIP + Lumpsum | `<ComposedChart>` 3 series |
| Simulation | 200 grey sample paths | HTML5 `<canvas>` (Recharts can't handle 200 lines) |
| Simulation | Percentile bands | `<AreaChart>` 5 bands |
| Simulation | Final value histogram | `<BarChart>` histnorm |

> **Simulation note:** Use `<canvas>` + `requestAnimationFrame` for the 200 grey paths. Recharts renders 200 `<Line>` elements = major lag. Percentile bands (5 lines) are fine in Recharts.

### Step 3.5 — Data Flow for Sample Report (Decision #3)

```
Overview page load (Sample mode):
  React → Supabase: SELECT * FROM default_fund_nav
  React has all 14 fund NAV series locally
  React → Azure POST /analytics/overview
  Azure computes metrics → returns JSON
  React renders KPI cards + table + charts

Predictions page, "Run" clicked:
  React → Azure POST /analytics/predict
    { fund_name, data (from supabase cache), model_name, periods }
  Azure: extract features → load universal_fund_model.pkl → predict
  React renders forecast chart with CI bands (same as Streamlit)

New fund (New Analysis page):
  User searches → GET /mf/search → picks fund
  POST /mf/fetch-batch → gets live NAV series
  POST /analytics/predict → universal model inference
  Results displayed on frontend
  Stored in live_fund_sessions on Supabase for session
```

---

## Phase 4 — Vercel Deployment

**`react-frontend/vercel.json`:**
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite"
}
```

**`react-frontend/vite.config.js`:**
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Local dev: proxy /api → FastAPI (avoids CORS in dev)
      '/api': {
        target: 'http://localhost:8000',
        rewrite: path => path.replace(/^\/api/, '')
      }
    }
  }
})
```

---

## Phase 5 — Error Handling

| Scenario | Handling |
|----------|----------|
| Azure backend down | Health check on mount → "Backend offline" banner; analytics disabled |
| Supabase unreachable | Fallback: call `GET /analytics/data/default` from Azure |
| universal_fund_model.pkl missing | Placeholder LinearRegression fallback; response includes `model_used: "fallback"` |
| Model timeout | FastAPI returns 202 + job_id → React polls `/analytics/job/{id}` every 3s |
| Analytics cache stale | `expires_at` check; if stale, re-call Azure, update Supabase cache |
| No data for fund | Backend returns `{ error: "Insufficient data" }` → React inline error |
| xlsx wrong format | Backend returns 422 with field errors → React field-level messages |

---

## Execution Order

```
Week 1 — Backend prep
  Day 1: Move modules/ → backend/, strip all streamlit imports
  Day 2: analytics_router.py + analytics_service.py (all endpoints)
  Day 3: model_cache.py (placeholder universal model + feature extractor)
  Day 4: supabase_service.py + seed default_fund_nav from data.xlsx
  Day 5: Test all endpoints via /docs, fix edge cases

Week 2 — React core
  Day 1: Vite scaffold, AppContext, api.js (axios), supabaseClient.js
  Day 2: Home + Overview (KPI cards, tables, 4 charts)
  Day 3: Analysis + Predictions (ComposedChart CI bands)
  Day 4: Risk + Backtesting
  Day 5: Simulation (canvas paths + AreaChart bands)

Week 3 — Polish + deploy
  Day 1: New Analysis page (live fund search + predict flow)
  Day 2: Theme system (dark/light CSS vars, localStorage)
  Day 3: Mobile responsive
  Day 4: Vercel deploy, env vars, CORS lock
  Day 5: End-to-end smoke test (local + prod)

Later (separate task):
  -> Train universal_fund_model.pkl on 100+ funds x 10+ years
  -> Upload to Azure backend/models/
  -> No code changes needed (model_cache.py interface stays same)
```

---

## Files Created Summary

**Backend (new):**
- `backend/routers/analytics_router.py`
- `backend/services/analytics_service.py`
- `backend/services/model_cache.py`
- `backend/services/data_service.py`
- `backend/services/supabase_service.py`
- `backend/modules/` — 6 files moved + cleaned
- `backend/config.py` — moved, Plotly blocks removed
- `backend/data.xlsx` — moved
- `backend/models/universal_fund_model.pkl` — placeholder now, full later

**Supabase (upload once):**
- `supabase/schema.sql`
- `supabase/fund_metadata.json`
- `supabase/default_fund_data.json`

**React frontend (new):**
- `react-frontend/src/pages/` — 9 pages
- `react-frontend/src/components/` — KpiCard, DataTable, charts
- `react-frontend/src/hooks/` — useDefaultFunds, useLiveFunds, useAnalytics
- `react-frontend/src/utils/` — chartConfig.js, formatters.js, canvasSimulation.js
- `react-frontend/src/lib/api.js` — axios wrapper (VITE_API_URL)
- `react-frontend/src/lib/supabaseClient.js`
- `react-frontend/src/context/AppContext.jsx`

**Unchanged:**
- `backend/routers/mf_router.py`
- `backend/services/mf_service.py`
- `frontend/` — keep alive during transition
