# FundScope — System Workflow & Architecture Overview

> How the whole project hangs together, how data moves, and where improvements can be made.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                                  │
│                   React App (Vercel CDN)                             │
│  localhost:5173 (dev)  |  <app>.vercel.app (prod)                   │
└────────────┬──────────────────────────┬────────────────────────────┘
             │                          │
     Supabase JS SDK              Axios HTTP
     (direct from browser)        (via VITE_API_URL)
             │                          │
             ▼                          ▼
┌─────────────────────┐     ┌───────────────────────────────────────┐
│   SUPABASE (cloud)  │     │   FASTAPI BACKEND (Azure App Service) │
│                     │     │   localhost:8000 (dev)                │
│  • fund_metadata    │     │   fundscopebackend-*.azurewebsites.net│
│  • default_fund_nav │     │                                       │
│  • analytics_cache  │     │  routers/                             │
│  • live_fund_sessions│    │    mf_router.py  → mfapi.in           │
│                     │     │    analytics_router.py → modules/     │
└─────────────────────┘     │                                       │
                            │  services/                            │
                            │    model_cache.py                     │
                            │      → universal_fund_model.pkl       │
                            │    supabase_service.py → Supabase     │
                            │    analytics_service.py               │
                            │    data_service.py → data.xlsx        │
                            └───────────────────────────────────────┘
```

---

## Data Flow Scenarios

### Scenario A — User views Sample Report (data.xlsx)

```
1. React mounts
   └─ AppContext: fetch fund_metadata from Supabase
   └─ AppContext: fetch default_fund_nav from Supabase (all 14 funds)
   └─ AppContext: GET /analytics/health from Azure → check backend alive

2. User navigates to Overview
   └─ Check analytics_cache in Supabase (key: "overview", hash of fund list)
   └─ IF cache fresh (< 24h): render directly from Supabase cache
   └─ IF cache stale/missing:
       └─ POST /analytics/overview to Azure
              body: { funds_data: { fundName: [{ds,y}] } }
       └─ Azure: compute cagr, sharpe, sortino, drawdown for all funds
       └─ Azure: write result to Supabase analytics_cache
       └─ React: render KPI cards + table + 4 charts

3. User navigates to Predictions
   └─ Selects fund + model + horizon
   └─ POST /analytics/predict to Azure
          body: { fund_name, data (from Supabase default_fund_nav), model_name, periods }
   └─ Azure: extract_features(data) → load universal_fund_model.pkl → predict
   └─ Azure: return { forecast:[{ds,yhat,yhat_lower,yhat_upper}], metrics }
   └─ React: render ComposedChart (historical line + forecast line + CI area band)
```

### Scenario B — User runs Live Fund Analysis

```
1. User goes to New Analysis page
   └─ Types fund name → GET /mf/search → Azure → mfapi.in → returns scheme list
   └─ User picks fund(s) → POST /mf/fetch-batch → Azure → mfapi.in
   └─ Azure returns NAV series → React stores in liveFunds context state
   └─ React writes to Supabase live_fund_sessions (session_id keyed)

2. User clicks "Predict"
   └─ POST /analytics/predict
          body: { fund_name, data: liveFunds[fund_name], model_name, periods }
   └─ Azure: same universal model inference as Scenario A
   └─ React: same chart rendering

3. User navigates to Risk/Backtest/Simulation with live fund
   └─ Same: POST to Azure with liveFunds data
   └─ Azure computes → React renders (same charts as Sample mode)
```

### Scenario C — New Fund, Universal Model Inference

```
Fund data (any fund, any history length)
    │
    ▼
extract_features(df):
  - rolling_5d_return
  - rolling_20d_return
  - rolling_60d_return
  - volatility_20d
  - momentum_14d
  - nav_zscore
    │
    ▼
universal_fund_model.pkl.predict_sequence(features, periods)
    │
    ▼
Relative daily returns for next N days
    │
    ▼
Apply to last_nav: yhat = last_nav * cumprod(1 + returns)
    │
    ▼
Return { forecast: [{ds, yhat, yhat_lower, yhat_upper}] }
```

---

## Port & URL Configuration

| Service | Local | Production | Set in |
|---------|-------|------------|--------|
| FastAPI | `localhost:8000` | Azure URL | `backend/.env` + Azure App Settings |
| React Vite | `localhost:5173` | Vercel auto | `vite.config.js` |
| Streamlit (legacy) | `localhost:8501` | Streamlit Cloud | `frontend/config.py` |
| Supabase | always cloud | always cloud | `.env.local` + Vercel env |

**React env switching (automatic):**
```javascript
// src/lib/api.js
const BASE = import.meta.env.VITE_API_URL  // localhost:8000 or Azure URL
// Same code, different URL injected at build time
```

---

## Service Responsibilities

| Service | Does | Does NOT |
|---------|------|----------|
| **Supabase** | Store NAV data, fund metadata, analytics cache, session data | Run any computation |
| **Azure FastAPI** | ML computation, mfapi.in proxy, xlsx parsing, pkl inference | Store data long-term |
| **Vercel React** | UI rendering, chart drawing, user interaction | Any ML or data storage |
| **universal_fund_model.pkl** | Feature-based NAV prediction for ANY fund | Per-fund training |

---

## React Page → API Dependency Map

```
/               (Home)
  └─ Supabase: fund_metadata

/overview       (Overview)
  └─ Supabase: default_fund_nav (or liveFunds from context)
  └─ Azure: POST /analytics/overview
  └─ Supabase: analytics_cache (read/write)

/analysis       (Analysis)
  └─ Supabase: default_fund_nav
  └─ Azure: POST /analytics/analysis
  └─ Azure: yfinance benchmark (inside analytics_service.py)

/predictions    (Predictions)
  └─ Supabase: default_fund_nav
  └─ Azure: POST /analytics/predict  (universal pkl)
  └─ Azure: POST /analytics/predict-ensemble  (multiple models)

/risk           (Risk Analysis)
  └─ Supabase: default_fund_nav
  └─ Azure: POST /analytics/risk

/backtesting    (Backtesting)
  └─ Supabase: default_fund_nav
  └─ Azure: POST /analytics/backtest

/simulation     (Monte Carlo)
  └─ Supabase: default_fund_nav
  └─ Azure: POST /analytics/simulate

/manual         (User Manual)
  └─ Static — no API

/new-analysis   (Live Fund Search)
  └─ Azure: GET /mf/search
  └─ Azure: POST /mf/fetch-batch
  └─ Azure: POST /analytics/predict
  └─ Supabase: live_fund_sessions (write)
```

---

## Analytics Cache Strategy (Supabase)

```
On any analytics call:
  1. hash(fund_name + analysis_type + params) → params_hash
  2. SELECT from analytics_cache WHERE fund_name + analysis_type + params_hash
  3. IF found AND computed_at < 24h ago:
       return cached result (no Azure call)
  4. ELSE:
       call Azure → get result
       UPSERT into analytics_cache
       return result

Cache TTL by type:
  overview     → 24h  (changes only when new data comes)
  analysis     → 24h
  predict      → 12h  (model results are deterministic; short TTL for freshness)
  risk         → 24h
  backtest     → 48h  (historical, almost never changes)
  simulate     → 1h   (random, user expects variation)
```

---

## Current State vs Target State

| Component | Current | Target |
|-----------|---------|--------|
| Frontend | Streamlit (Python) | React + Vite (Vercel) |
| ML compute | Streamlit process | FastAPI on Azure |
| Data storage | Local data.xlsx | Supabase `default_fund_nav` |
| Fund metadata | `config.py` hardcoded | Supabase `fund_metadata` |
| Analytics cache | None (recomputes every run) | Supabase `analytics_cache` |
| Predictions | Per-request training (slow) | Universal pkl inference (fast) |
| Live fund data | `session_state` (per tab) | Supabase `live_fund_sessions` |
| CORS | Streamlit only | Streamlit + React dev + Vercel |

---

## Potential Improvements (Future Backlog)

### High Value
| Improvement | Effort | Impact |
|-------------|--------|--------|
| Train universal_fund_model.pkl on 100+ funds × 10+ years | High | Very High — eliminates per-fund training forever |
| Supabase Row-Level Security (RLS) for user sessions | Medium | Security |
| Background job queue (Celery or FastAPI BackgroundTasks) for heavy analytics | Medium | UX — non-blocking UI |
| Supabase Realtime for live NAV updates | Medium | UX — live dashboard |

### Medium Value
| Improvement | Effort | Impact |
|-------------|--------|--------|
| Add more fund years to data.xlsx → re-seed Supabase | Low | Better analytics accuracy |
| Cache benchmark (Nifty/Sensex) data in Supabase | Low | Faster analysis page |
| PDF report export from React | Medium | User feature |
| Fund comparison page (multi-fund overlay, not just overview) | Medium | User feature |
| Portfolio builder — allocate % across funds, see blended metrics | High | User feature |

### Infrastructure
| Improvement | Effort | Impact |
|-------------|--------|--------|
| Move Azure to Azure Container Apps (Docker) — fixes Prophet native deps | Medium | Stability |
| Add `/analytics/job/{id}` polling for long-running simulations | Low | UX |
| Add Supabase analytics_cache automatic purge (pg_cron) | Low | DB hygiene |
| Rate limiting on Azure endpoints | Low | Cost control |

---

## Development Workflow (Daily)

```
Local dev startup:
  Terminal 1: cd backend && uvicorn main:app --reload --port 8000
  Terminal 2: cd react-frontend && npm run dev   (→ localhost:5173)
  Supabase: always cloud (no local setup needed)

Test a new endpoint:
  → http://localhost:8000/docs  (Swagger UI auto-generated)

Deploy:
  Backend: git push → Azure auto-deploys (GitHub Actions or Azure Deploy Center)
  Frontend: git push → Vercel auto-deploys

Check logs:
  Azure: Azure Portal → App Service → Log stream
  Vercel: Vercel dashboard → Deployments → Function logs
  Supabase: Supabase dashboard → Table Editor / Logs
```

---

## File Map Quick Reference

```
FundScope/
├── Plans/
│   ├── react_migration_plan.md    <- detailed implementation steps
│   └── workflow.md                <- this file (architecture overview)
├── backend/
│   ├── main.py                    <- CORS, router registration
│   ├── routers/
│   │   ├── mf_router.py           <- /mf/schemes /mf/search /mf/fetch-batch
│   │   └── analytics_router.py   <- /analytics/* (all ML endpoints)
│   ├── services/
│   │   ├── mf_service.py          <- mfapi.in calls
│   │   ├── analytics_service.py   <- wraps modules/* functions
│   │   ├── model_cache.py         <- universal pkl load + feature extract
│   │   ├── data_service.py        <- xlsx parse + Supabase seed
│   │   └── supabase_service.py    <- cache read/write
│   ├── modules/                   <- pure Python ML (moved from frontend)
│   │   ├── analysis.py
│   │   ├── predictions.py
│   │   ├── risk.py
│   │   ├── simulation.py
│   │   ├── backtesting.py
│   │   └── data_loader.py
│   ├── models/
│   │   └── universal_fund_model.pkl
│   ├── config.py                  <- FUND_NAMES, BENCHMARKS, RISK constants
│   └── data.xlsx
├── react-frontend/
│   ├── src/
│   │   ├── context/AppContext.jsx  <- global state (dataMode, funds, theme)
│   │   ├── lib/
│   │   │   ├── api.js             <- axios(VITE_API_URL)
│   │   │   └── supabaseClient.js  <- createClient(VITE_SUPABASE_URL)
│   │   ├── hooks/
│   │   │   ├── useDefaultFunds.js <- Supabase default_fund_nav
│   │   │   ├── useLiveFunds.js    <- /mf/fetch-batch
│   │   │   └── useAnalytics.js    <- cache check + Azure call
│   │   ├── pages/                 <- 9 page components
│   │   ├── components/            <- KpiCard, DataTable, charts
│   │   └── utils/
│   │       ├── chartConfig.js     <- CHART_COLORS, theme vars
│   │       ├── formatters.js      <- currency, % formatters
│   │       └── canvasSimulation.js <- Monte Carlo canvas renderer
│   ├── .env.local                 <- VITE_API_URL=localhost:8000
│   └── vite.config.js
├── supabase/
│   ├── schema.sql                 <- run once in Supabase SQL editor
│   ├── fund_metadata.json         <- 14 fund records to insert
│   └── default_fund_data.json     <- NAV rows from data.xlsx
└── frontend/                      <- Streamlit (keep until Vercel deploy verified)
```
