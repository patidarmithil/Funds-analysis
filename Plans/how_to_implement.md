# FundScope — How to Implement the Backend (Step-by-Step)

## 🛠️ Tech Stack to Use

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend framework | **NestJS** (TypeScript) | Structured, modular, great for REST APIs |
| HTTP client | **axios** (via `@nestjs/axios`) | Built-in NestJS support, Promise-based |
| Scheduling | **@nestjs/schedule** | Built-in cron job decorator |
| Caching | **In-memory Map** (Phase 1) / **Redis** (Phase 2) | Simple start, scalable later |
| Config management | **@nestjs/config** + `.env` | Clean separation of secrets/settings |
| Frontend data fetch | **Python `requests`** library | Already available in the Streamlit env |

---

## 📋 Phase 1 — Scaffold the NestJS Project

### Step 1.1 — Create the Project

```bash
cd D:\Coding\Project\Finance\FundScope
npx @nestjs/cli new fundscope-backend --package-manager npm
cd fundscope-backend
```

### Step 1.2 — Install Required Packages

```bash
npm install @nestjs/axios axios @nestjs/schedule @nestjs/config
npm install --save-dev @types/node
```

### Step 1.3 — Project File Layout to Create

```
fundscope-backend/src/
├── main.ts                          ← enable CORS, set port
├── app.module.ts                    ← import all modules
├── funds/
│   ├── funds.module.ts              ← wires everything together
│   ├── funds.controller.ts          ← REST route definitions
│   ├── funds.service.ts             ← core fetch + cache logic
│   └── funds.config.ts              ← fund name → scheme code map
└── scheduler/
    └── data-refresh.service.ts      ← cron job for daily refresh
```

---

## 📋 Phase 2 — The Fund Config File

### `funds/funds.config.ts`

This is the **master mapping file** — it links the fund names used in
FundScope to the AMFI scheme codes needed by mfapi.in.

```typescript
// funds/funds.config.ts
export const FUND_SCHEME_MAP: Record<string, number> = {
  'Flexi Cap':                  122639,   // Parag Parikh Flexi Cap
  'India PSU':                  147622,   // look up exact code
  'Infrastructure':             120503,
  'Midcap':                     130503,
  'Focused India':              120684,
  'Large and midcap fund':      118989,
  'Contra':                     119775,
  'Multicap':                   135781,
  'Financial Services':         119598,
  'ESG Integration Strategy':   145552,
  'ELSS Tax Saver':             118834,
  // International funds — may not be on mfapi.in, skip or use fallback
  'Invesco Pan European':       null,
  'Global Consumer Trends':     null,
  'EQQQ NASDAQ-100 ETF':        null,
};
```

> **Note:** Scheme codes for international funds (last 3) may not be
> available on mfapi.in. These can either be skipped in V1 or fetched
> from a separate source (e.g. BSE / fund house website).

---

## 📋 Phase 3 — The Funds Service (Core Logic)

### `funds/funds.service.ts`

```typescript
import { Injectable, OnModuleInit } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';
import { FUND_SCHEME_MAP } from './funds.config';

interface NavPoint { ds: string; y: number; }
interface FundData { fund: string; data: NavPoint[]; lastUpdated: string; }

@Injectable()
export class FundsService implements OnModuleInit {
  // In-memory store: fund name → { data, lastUpdated }
  private cache = new Map<string, FundData>();

  constructor(private readonly http: HttpService) {}

  // Called automatically when the NestJS app starts
  async onModuleInit() {
    await this.refreshAllFunds();
  }

  // Fetch data for ALL funds and populate cache
  async refreshAllFunds(): Promise<void> {
    const entries = Object.entries(FUND_SCHEME_MAP);
    for (const [fundName, schemeCode] of entries) {
      if (!schemeCode) continue;   // skip funds with no code
      try {
        const navData = await this.fetchFundNav(schemeCode);
        this.cache.set(fundName, {
          fund: fundName,
          data: navData,
          lastUpdated: new Date().toISOString(),
        });
        console.log(`✅ Fetched ${fundName} (${navData.length} records)`);
      } catch (err) {
        console.error(`❌ Failed to fetch ${fundName}: ${err.message}`);
      }
    }
  }

  // Fetch NAV for a single scheme from mfapi.in and normalise it
  private async fetchFundNav(schemeCode: number): Promise<NavPoint[]> {
    const url = `https://api.mfapi.in/mf/${schemeCode}`;
    const res = await firstValueFrom(this.http.get(url));
    const rawData: { date: string; nav: string }[] = res.data.data;

    return rawData
      .map(item => ({
        ds: this.parseDate(item.date),  // "04-05-2026" → "2026-05-04"
        y: parseFloat(item.nav),
      }))
      .filter(item => !isNaN(item.y))
      .sort((a, b) => a.ds.localeCompare(b.ds));  // ascending date order
  }

  // Convert "DD-MM-YYYY" from mfapi.in to "YYYY-MM-DD" (pandas-friendly)
  private parseDate(dateStr: string): string {
    const [d, m, y] = dateStr.split('-');
    return `${y}-${m}-${d}`;
  }

  // --- Public methods called by the Controller ---

  getAllFunds(): string[] {
    return Array.from(this.cache.keys());
  }

  getFundData(fundName: string): FundData | null {
    return this.cache.get(fundName) ?? null;
  }

  getAllFundsData(): FundData[] {
    return Array.from(this.cache.values());
  }

  getLatestNav(fundName: string): { fund: string; date: string; nav: number } | null {
    const fund = this.cache.get(fundName);
    if (!fund || fund.data.length === 0) return null;
    const latest = fund.data[fund.data.length - 1];
    return { fund: fundName, date: latest.ds, nav: latest.y };
  }
}
```

---

## 📋 Phase 4 — The Controller (REST Endpoints)

### `funds/funds.controller.ts`

```typescript
import { Controller, Get, Param, Post, NotFoundException } from '@nestjs/common';
import { FundsService } from './funds.service';

@Controller('funds')
export class FundsController {
  constructor(private readonly fundsService: FundsService) {}

  // GET /funds → list of fund names
  @Get()
  listFunds() {
    return { funds: this.fundsService.getAllFunds() };
  }

  // GET /funds/all → full NAV history for every fund
  @Get('all')
  getAllData() {
    return this.fundsService.getAllFundsData();
  }

  // GET /funds/all/latest → latest NAV per fund
  @Get('all/latest')
  getAllLatest() {
    const funds = this.fundsService.getAllFunds();
    return funds.map(f => this.fundsService.getLatestNav(f)).filter(Boolean);
  }

  // POST /funds/refresh → trigger a manual re-fetch
  @Post('refresh')
  async refresh() {
    await this.fundsService.refreshAllFunds();
    return { message: 'Refresh complete', timestamp: new Date().toISOString() };
  }

  // GET /funds/:name/nav → full NAV history for one fund
  @Get(':name/nav')
  getFundNav(@Param('name') name: string) {
    const decoded = decodeURIComponent(name);
    const data = this.fundsService.getFundData(decoded);
    if (!data) throw new NotFoundException(`Fund "${decoded}" not found`);
    return data;
  }

  // GET /funds/:name/latest → latest single NAV
  @Get(':name/latest')
  getLatest(@Param('name') name: string) {
    const decoded = decodeURIComponent(name);
    const latest = this.fundsService.getLatestNav(decoded);
    if (!latest) throw new NotFoundException(`Fund "${decoded}" not found`);
    return latest;
  }
}
```

---

## 📋 Phase 5 — The Scheduler (Daily Auto-Refresh)

### `scheduler/data-refresh.service.ts`

```typescript
import { Injectable } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { FundsService } from '../funds/funds.service';

@Injectable()
export class DataRefreshService {
  constructor(private readonly fundsService: FundsService) {}

  // Runs every day at 8:00 AM IST (2:30 AM UTC)
  @Cron('30 2 * * *')
  async handleDailyRefresh() {
    console.log('🔄 Daily refresh triggered...');
    await this.fundsService.refreshAllFunds();
    console.log('✅ Daily refresh complete');
  }
}
```

---

## 📋 Phase 6 — main.ts (Enable CORS)

```typescript
// main.ts
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  // Allow Streamlit (running on :8501) to call this API
  app.enableCors({
    origin: ['http://localhost:8501', 'http://127.0.0.1:8501'],
    methods: ['GET', 'POST'],
  });

  await app.listen(3000);
  console.log('🚀 FundScope backend running on http://localhost:3000');
}
bootstrap();
```

---

## 📋 Phase 7 — Update Streamlit's data_loader.py

Replace the Excel-reading logic with an HTTP call to NestJS:

```python
# modules/data_loader.py  (new section to add)
import requests
from config import BACKEND_URL  # e.g. "http://localhost:3000"

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_funds_from_api() -> dict[str, pd.DataFrame]:
    """
    Calls the NestJS backend to get all fund NAV data.
    Returns { fund_name: DataFrame } same as load_all_funds().
    """
    try:
        res = requests.get(f"{BACKEND_URL}/funds/all", timeout=15)
        res.raise_for_status()
        funds_json = res.json()   # list of { fund, data: [{ds, y}], ... }
    except Exception as e:
        st.warning(f"⚠️ Backend unavailable: {e}. Falling back to local data.")
        return load_all_funds()   # fall back to Excel

    funds = {}
    for entry in funds_json:
        name = entry['fund']
        df = pd.DataFrame(entry['data'])
        df['ds'] = pd.to_datetime(df['ds'])
        df['y']  = pd.to_numeric(df['y'])
        df = _preprocess_from_api(df, name)
        if df is not None:
            funds[name] = df
    return funds


def _preprocess_from_api(df: pd.DataFrame, name: str) -> pd.DataFrame | None:
    """Same as _preprocess but df already has ds/y columns."""
    df.sort_values('ds', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['returns']        = df['y'].pct_change()
    df['log_returns']    = np.log(df['y'] / df['y'].shift(1))
    df['rolling_30_ret'] = df['returns'].rolling(30).mean()
    return df
```

Add to `config.py`:
```python
BACKEND_URL = os.getenv("FUNDSCOPE_BACKEND_URL", "http://localhost:3000")
```

---

## 📋 Phase 8 — Environment File

### `fundscope-backend/.env`

```env
PORT=3000
CACHE_TTL_HOURS=24
MFAPI_BASE_URL=https://api.mfapi.in/mf
```

---

## 🧪 How to Run Both Services Together

```bash
# Terminal 1 — Start NestJS backend
cd D:\Coding\Project\Finance\FundScope\fundscope-backend
npm run start:dev

# Terminal 2 — Start Streamlit frontend
cd D:\Coding\Project\Finance\FundScope
streamlit run app.py
```

Or use a `Procfile` / `docker-compose.yml` to run both together.

---

## 🗺️ Implementation Order (Recommended)

| # | Task | Effort |
|---|------|--------|
| 1 | Scaffold NestJS project, install deps | 10 min |
| 2 | Create `funds.config.ts` with all scheme codes | 30 min |
| 3 | Implement `funds.service.ts` (fetch + cache) | 45 min |
| 4 | Implement `funds.controller.ts` (endpoints) | 20 min |
| 5 | Add CORS in `main.ts` | 5 min |
| 6 | Test endpoints with curl/Postman | 15 min |
| 7 | Add `data-refresh.service.ts` (scheduler) | 15 min |
| 8 | Update `data_loader.py` in Streamlit | 30 min |
| 9 | Update `config.py` with `BACKEND_URL` | 5 min |
| 10 | End-to-end test: run both, verify data flows | 30 min |

**Total estimated time: ~3.5 hours**

---

## 🚨 Known Challenges & How to Handle Them

| Challenge | Solution |
|-----------|----------|
| International funds not on mfapi.in | Skip in V1; add a `null` check in config |
| mfapi.in returns 429 (rate limit) | Add delay between requests: `await sleep(200)` |
| Scheme codes change | Rarely happens — verify yearly |
| NestJS cold start takes time | Show loading spinner in Streamlit while waiting |
| Data stale during weekend/holiday | mfapi.in still returns last published NAV — OK |
| Backend port conflict | Make port configurable via `.env` |

---

## 🔮 Future Enhancements (V2+)

- **Redis caching** — persist cache across restarts
- **PostgreSQL** — store historical NAV in a proper DB
- **WebSockets** — push real-time NAV updates to frontend
- **Docker Compose** — single command to run everything
- **Authentication** — API key protection for the NestJS endpoints
- **Admin dashboard** — web UI to monitor cache status and force refresh
