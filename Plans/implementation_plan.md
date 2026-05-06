# FundScope — Implementation Plan

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Backend | **NestJS** (TypeScript) | REST API server, mfapi.in proxy |
| HTTP client | **@nestjs/axios** + **axios** | Fetch from mfapi.in |
| Parallel fetch | **Promise.all** | Fetch N funds simultaneously |
| Frontend cache | **st.session_state** | Store scheme list + live fund data |
| Frontend HTTP | **Python `requests`** | Streamlit → NestJS calls |

---

## 📋 Phase 1 — Scaffold NestJS Backend

### 1.1 Create Project

```bash
cd D:\Coding\Project\Finance\FundScope
npx @nestjs/cli new fundscope-backend --package-manager npm --skip-git
cd fundscope-backend
npm install @nestjs/axios axios @nestjs/config
```

### 1.2 Final File Structure

```
fundscope-backend/
├── src/
│   ├── main.ts                  ← CORS, port setup
│   ├── app.module.ts            ← import MfModule
│   └── mf/
│       ├── mf.module.ts         ← wire controller + service
│       ├── mf.controller.ts     ← 3 endpoints
│       └── mf.service.ts        ← fetch logic + normalise
├── .env
└── package.json
```

---

## 📋 Phase 2 — main.ts

```typescript
// src/main.ts
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  app.enableCors({
    origin: ['http://localhost:8501', 'http://127.0.0.1:8501'],
    methods: ['GET', 'POST'],
  });
  await app.listen(process.env.PORT ?? 3000);
  console.log('🚀 FundScope backend: http://localhost:3000');
}
bootstrap();
```

---

## 📋 Phase 3 — mf.service.ts (Core Logic)

```typescript
// src/mf/mf.service.ts
import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';

const MFAPI = 'https://api.mfapi.in';

export interface NavPoint { ds: string; y: number; }
export interface FundResult {
  fund: string;
  code: number;
  data: NavPoint[];
  error?: string;
}

@Injectable()
export class MfService {
  constructor(private readonly http: HttpService) {}

  // ─── 1. Get all scheme codes + names ──────────────────────────────────
  async getAllSchemes(): Promise<any[]> {
    // mfapi returns all schemes at GET /mf (no pagination needed — it returns all)
    const res = await firstValueFrom(this.http.get(`${MFAPI}/mf`));
    return res.data;  // [ { schemeCode, schemeName }, ... ]
  }

  // ─── 2. Search schemes by name ────────────────────────────────────────
  async searchSchemes(query: string): Promise<any[]> {
    const res = await firstValueFrom(
      this.http.get(`${MFAPI}/mf/search`, { params: { q: query } })
    );
    return res.data;  // [ { schemeCode, schemeName }, ... ]
  }

  // ─── 3. Fetch NAV for multiple funds in parallel ───────────────────────
  async fetchBatch(
    funds: { schemeCode: number; schemeName: string }[],
    startDate?: string,
    endDate?: string,
  ): Promise<FundResult[]> {
    const results = await Promise.all(
      funds.map(f => this.fetchOneFund(f.schemeCode, f.schemeName, startDate, endDate))
    );
    return results;
  }

  // ─── Internal: fetch + normalise one fund ─────────────────────────────
  private async fetchOneFund(
    code: number,
    name: string,
    startDate?: string,
    endDate?: string,
  ): Promise<FundResult> {
    try {
      const params: Record<string, string> = {};
      if (startDate) params.startDate = startDate;
      if (endDate)   params.endDate   = endDate;

      const res = await firstValueFrom(
        this.http.get(`${MFAPI}/mf/${code}`, { params })
      );

      const raw: { date: string; nav: string }[] = res.data.data;
      const data: NavPoint[] = raw
        .map(item => ({
          ds: this.toISODate(item.date),   // "26-10-2024" → "2024-10-26"
          y:  parseFloat(item.nav),
        }))
        .filter(item => !isNaN(item.y))
        .sort((a, b) => a.ds.localeCompare(b.ds));  // ascending order

      return { fund: name, code, data };
    } catch (err) {
      return { fund: name, code, data: [], error: err.message };
    }
  }

  // "DD-MM-YYYY" → "YYYY-MM-DD"
  private toISODate(dateStr: string): string {
    const [d, m, y] = dateStr.split('-');
    return `${y}-${m}-${d}`;
  }
}
```

---

## 📋 Phase 4 — mf.controller.ts (Endpoints)

```typescript
// src/mf/mf.controller.ts
import { Controller, Get, Post, Query, Body } from '@nestjs/common';
import { MfService } from './mf.service';

@Controller('mf')
export class MfController {
  constructor(private readonly mfService: MfService) {}

  // GET /mf/schemes — full list of all MF schemes (~17k)
  // Called ONCE per session, cached in Streamlit session_state
  @Get('schemes')
  async getAllSchemes() {
    return this.mfService.getAllSchemes();
  }

  // GET /mf/search?q=hdfc midcap — search by name
  @Get('search')
  async search(@Query('q') q: string) {
    return this.mfService.searchSchemes(q ?? '');
  }

  // POST /mf/fetch-batch — fetch NAV for selected funds
  // Body: { funds: [{schemeCode, schemeName}], startDate?, endDate? }
  @Post('fetch-batch')
  async fetchBatch(@Body() body: {
    funds: { schemeCode: number; schemeName: string }[];
    startDate?: string;
    endDate?: string;
  }) {
    return this.mfService.fetchBatch(body.funds, body.startDate, body.endDate);
  }
}
```

---

## 📋 Phase 5 — Module Wiring

```typescript
// src/mf/mf.module.ts
import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { MfController } from './mf.controller';
import { MfService } from './mf.service';

@Module({
  imports: [HttpModule],
  controllers: [MfController],
  providers: [MfService],
})
export class MfModule {}
```

```typescript
// src/app.module.ts
import { Module } from '@nestjs/common';
import { MfModule } from './mf/mf.module';

@Module({ imports: [MfModule] })
export class AppModule {}
```

---

## 📋 Phase 6 — Update data_loader.py

Add a new function to `modules/data_loader.py` (existing code untouched):

```python
# modules/data_loader.py — ADD BELOW existing code

import requests as _requests
from config import BACKEND_URL

def load_funds_from_api(
    selected_funds: list[dict],   # [{schemeCode, schemeName}, ...]
    start_date: str,              # "YYYY-MM-DD"
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """
    Calls the NestJS backend to fetch NAV data for selected funds.
    Returns the same { fund_name: DataFrame } shape as load_all_funds().
    """
    try:
        resp = _requests.post(
            f"{BACKEND_URL}/mf/fetch-batch",
            json={
                "funds": selected_funds,
                "startDate": start_date,
                "endDate": end_date,
            },
            timeout=60,   # large timeout: fetching many funds takes time
        )
        resp.raise_for_status()
        fund_list = resp.json()   # list of { fund, code, data: [{ds, y}] }
    except Exception as e:
        st.error(f"❌ Could not fetch data from backend: {e}")
        return {}

    result = {}
    for entry in fund_list:
        if not entry.get("data"):
            continue                      # skip failed funds
        name = entry["fund"]
        df = pd.DataFrame(entry["data"])  # columns: ds, y
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"]  = pd.to_numeric(df["y"], errors="coerce")
        df = _preprocess_from_api(df)
        if df is not None:
            result[name] = df
    return result


def _preprocess_from_api(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Adds computed columns to a {ds, y} DataFrame — same as _preprocess()
    but assumes ds/y columns already exist and are correctly typed.
    """
    df = df.dropna(subset=["ds", "y"])
    df = df.sort_values("ds").reset_index(drop=True)
    df["returns"]        = df["y"].pct_change()
    df["log_returns"]    = np.log(df["y"] / df["y"].shift(1))
    df["rolling_30_ret"] = df["returns"].rolling(30).mean()
    return df if len(df) > 5 else None


def search_schemes_api(query: str) -> list[dict]:
    """Search mfapi.in scheme names via NestJS proxy."""
    try:
        resp = _requests.get(
            f"{BACKEND_URL}/mf/search",
            params={"q": query},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()   # [ {schemeCode, schemeName}, ... ]
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)   # cache for 24h
def get_all_schemes_cached() -> list[dict]:
    """Fetch full scheme list once per day from NestJS."""
    try:
        resp = _requests.get(f"{BACKEND_URL}/mf/schemes", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []
```

---

## 📋 Phase 7 — Update config.py

```python
# config.py — add this line
BACKEND_URL = os.getenv("FUNDSCOPE_BACKEND_URL", "http://localhost:3000")
```

---

## 📋 Phase 8 — Update app.py

### 8.1 — Session State Additions

```python
# Add to session state defaults block in app.py
if "data_mode"           not in st.session_state:
    st.session_state.data_mode           = "default"
if "live_funds_data"     not in st.session_state:
    st.session_state.live_funds_data     = {}
if "all_scheme_list"     not in st.session_state:
    st.session_state.all_scheme_list     = []
```

### 8.2 — Sidebar New Analysis Section

Add this after the existing nav buttons in the sidebar:

```python
st.markdown("---")
st.markdown('<p class="section-header">Live Analysis</p>', unsafe_allow_html=True)

col_def, col_new = st.columns(2)
with col_def:
    if st.button("📂 Default", use_container_width=True,
                 type="primary" if st.session_state.data_mode == "default" else "secondary"):
        st.session_state.data_mode = "default"
        st.rerun()
with col_new:
    if st.button("🔍 New Analysis", use_container_width=True,
                 type="primary" if st.session_state.data_mode == "live" else "secondary"):
        st.session_state.data_mode = "live"
        st.session_state.page = "🔍  New Analysis"
        st.rerun()
```

### 8.3 — Add New Analysis Page to PAGES dict

```python
import app_pages.new_analysis as pg_new_analysis

PAGES = {
    "📊  Overview":         pg_overview,
    "📈  Analysis":         pg_analysis,
    "🤖  Predictions":      pg_predictions,
    "⚠️  Risk Analysis":    pg_risk,
    "🔁  Backtesting":      pg_backtest,
    "🌀  Simulation":       pg_simulation,
    "🔍  New Analysis":     pg_new_analysis,   # NEW
    "📚  User Manual":      pg_manual,
}
```

### 8.4 — Route all_funds Based on Mode

```python
# Replace the existing all_funds loading block with:
with st.spinner("Loading fund data…"):
    if st.session_state.data_mode == "live" and st.session_state.live_funds_data:
        all_funds = st.session_state.live_funds_data
    else:
        all_funds = load_all_funds(st.session_state.uploaded_bytes)
        st.session_state.data_mode = "default"

if not all_funds and st.session_state.data_mode == "default":
    st.error("⚠️ Could not load any fund data.")
    st.stop()
```

---

## 📋 Phase 9 — Create app_pages/new_analysis.py

This is the new page shown when user clicks "New Analysis":

```python
# app_pages/new_analysis.py
import streamlit as st
from modules.data_loader import search_schemes_api, load_funds_from_api

def render(all_funds: dict):
    st.markdown('<p class="page-title">🔍 New Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Search live mutual funds and run analysis on custom date ranges</p>',
                unsafe_allow_html=True)

    # ── Search ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Step 1 — Search & Select Funds</p>', unsafe_allow_html=True)

    search_query = st.text_input("Search mutual fund by name", placeholder="e.g. HDFC Midcap, Parag Parikh...")

    results = []
    if search_query and len(search_query) >= 3:
        with st.spinner("Searching…"):
            results = search_schemes_api(search_query)

    if results:
        options = {f"{r['schemeName']} ({r['schemeCode']})": r for r in results}
        chosen_labels = st.multiselect(
            f"Select funds from results ({len(results)} found)",
            list(options.keys()),
        )
        chosen_funds = [options[lbl] for lbl in chosen_labels]
    else:
        chosen_funds = []
        if search_query and len(search_query) >= 3:
            st.info("No results found. Try a different search term.")

    # ── Date Range ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Step 2 — Select Date Range</p>', unsafe_allow_html=True)

    import datetime
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start Date", value=datetime.date(2020, 1, 1),
                                   min_value=datetime.date(2000, 1, 1))
    with col_e:
        end_date   = st.date_input("End Date",   value=datetime.date.today())

    if end_date <= start_date:
        st.warning("End date must be after start date.")
        return

    # ── Load Button ─────────────────────────────────────────────────────
    st.markdown("---")
    if not chosen_funds:
        st.info("☝️ Search and select at least one fund above, then click Load.")
        return

    st.markdown(f"**{len(chosen_funds)} fund(s) selected** — date range: "
                f"`{start_date}` → `{end_date}`")

    if st.button("📥 Load Data & Run Analysis", type="primary", use_container_width=True):
        payload = [
            {"schemeCode": f["schemeCode"], "schemeName": f["schemeName"]}
            for f in chosen_funds
        ]
        with st.spinner(f"Fetching NAV data for {len(chosen_funds)} fund(s)… this may take a moment"):
            live_data = load_funds_from_api(payload, str(start_date), str(end_date))

        if not live_data:
            st.error("❌ No data returned. Check backend is running or try different funds.")
            return

        st.session_state.live_funds_data = live_data
        st.session_state.data_mode = "live"
        st.success(f"✅ Loaded {len(live_data)} fund(s). Navigate to any page to see analysis.")
        st.info("👈 Use the sidebar to switch between Overview, Analysis, Predictions, etc.")

    # ── If data already loaded, show summary ────────────────────────────
    if st.session_state.live_funds_data:
        st.markdown("---")
        st.markdown('<p class="section-header">Currently Loaded Live Funds</p>', unsafe_allow_html=True)
        for name, df in st.session_state.live_funds_data.items():
            date_range = f"{df['ds'].min().date()} → {df['ds'].max().date()}"
            st.markdown(f"- **{name}**: {len(df):,} records &nbsp;&nbsp; `{date_range}`")
```

---

## 🗺️ Implementation Order

| # | Task | File(s) | Effort |
|---|------|---------|--------|
| 1 | Scaffold NestJS project + install deps | `fundscope-backend/` | 10 min |
| 2 | Write `mf.service.ts` | `src/mf/mf.service.ts` | 30 min |
| 3 | Write `mf.controller.ts` + module wiring | `src/mf/` | 15 min |
| 4 | Enable CORS in `main.ts` | `src/main.ts` | 5 min |
| 5 | Test all 3 endpoints with curl | — | 15 min |
| 6 | Add `BACKEND_URL` to `config.py` | `config.py` | 2 min |
| 7 | Add API functions to `data_loader.py` | `modules/data_loader.py` | 25 min |
| 8 | Create `app_pages/new_analysis.py` | `app_pages/new_analysis.py` | 30 min |
| 9 | Update `app.py` (session state + routing) | `app.py` | 20 min |
| 10 | End-to-end test: search → select → load → analysis | — | 30 min |

**Total estimated time: ~3 hours**

---

## 🚨 Known Edge Cases to Handle

| Case | Handling |
|------|---------|
| User selects 10+ funds | Promise.all in NestJS handles in parallel; Streamlit shows spinner |
| A fund has no data for selected date range | mfapi returns empty `data: []`; skip that fund with warning |
| mfapi.in rate limit hit | `Promise.all` may get 429; add per-fund error field in response |
| NestJS backend not running | `requests` call fails; `load_funds_from_api` catches and shows error |
| Very short date range (<30 days) | Warn user: "Not enough data for rolling calculations" |
| International funds (not on mfapi) | Search won't find them; user can only use Indian MFs |
| User navigates away mid-analysis | `live_funds_data` stays in session_state; no data loss |
