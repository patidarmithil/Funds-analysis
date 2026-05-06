# FundScope — Logics & Open Questions

## 🧠 All Internal Logics Explained

---

### Logic 1 — The Scheme List: Load Once, Reuse Forever

**Problem:** mfapi.in has ~17,000+ schemes. Fetching the full list on every user
interaction would be slow and hit rate limits.

**Solution:** The full scheme list is fetched **once per Streamlit session**
via `GET /mf/schemes`, and stored in `st.session_state.all_scheme_list`.
It is also cached on the Streamlit side with `@st.cache_data(ttl=86400)`
meaning it won't be re-fetched for 24 hours even if the session restarts.

**NestJS side:** Just a straight proxy — `GET /mf/schemes` → `GET api.mfapi.in/mf`.
No caching needed on the NestJS side for this since Streamlit handles it.

```
First visit:    st.session_state.all_scheme_list is empty
                → call GET /mf/schemes → stores ~17k items
Second onward:  all_scheme_list is populated
                → use directly, no API call
```

---

### Logic 2 — Search: Real-time via NestJS Proxy

**Problem:** Searching 17k schemes client-side in Python would require loading
all 17k into memory, which is slow. mfapi.in has a built-in search endpoint.

**Solution:** Search is NOT done against the local scheme list. Instead, every
search query is forwarded to `GET api.mfapi.in/mf/search?q=...` via the
NestJS proxy. This is fast and returns ranked results.

**UX trigger:** Search fires when user has typed **at least 3 characters**.
Below 3 chars, no API call is made (avoids meaningless results and rate limits).

---

### Logic 3 — Batch NAV Fetch: Parallel via Promise.all

**Problem:** If user selects 5 funds, fetching them sequentially (one after another)
could take 5–10 seconds. Users will not wait.

**Solution:** NestJS uses `Promise.all()` to fetch all selected funds
**simultaneously** in parallel. All N HTTP requests to mfapi.in fire at once.
Total wait time = time for the slowest fund, not the sum of all.

```
Sequential (bad):   5 funds × 2s each = 10s wait
Parallel (good):    5 funds together   = ~2s wait (slowest one)
```

**Risk:** If user selects 20+ funds, mfapi.in might throttle. To handle this,
we can split into batches of 10 if needed (Phase 2 enhancement).

---

### Logic 4 — Date Normalisation: mfapi Format vs Pandas Format

mfapi.in returns dates in `DD-MM-YYYY` format (Indian style).
Pandas `pd.to_datetime()` and all our analysis code expects `YYYY-MM-DD` (ISO 8601).

**Where it's fixed:** In NestJS `mf.service.ts` — every date is converted
**before** the response is sent to Streamlit. Streamlit receives clean ISO dates.

```typescript
// mfapi returns:  "26-10-2024"
// We convert to: "2024-10-26"
private toISODate(dateStr: string): string {
  const [d, m, y] = dateStr.split('-');
  return `${y}-${m}-${d}`;
}
```

---

### Logic 5 — Data Shape Compatibility (Critical)

All existing analysis pages call `render(all_funds: dict)` where `all_funds`
is a dict of `{ fund_name: DataFrame }`. Every DataFrame must have exactly:

| Column | Type | Source |
|--------|------|--------|
| `ds` | datetime64 | From mfapi `date` field |
| `y` | float64 | From mfapi `nav` field |
| `returns` | float64 | Computed: `y.pct_change()` |
| `log_returns` | float64 | Computed: `log(y / y.shift(1))` |
| `rolling_30_ret` | float64 | Computed: `returns.rolling(30).mean()` |

The function `_preprocess_from_api()` in `data_loader.py` adds these computed
columns to the raw `{ds, y}` data before storing in session_state. This is
the **same transformation** done by `_preprocess()` for the Excel file.

**Result:** All 6 analysis pages work with live data with zero code changes.

---

### Logic 6 — The Two-Mode Switch

The app operates in one of two modes at any time, tracked by
`st.session_state.data_mode`:

```
"default" mode:
  all_funds = load_all_funds(uploaded_bytes)   ← from data.xlsx or uploaded file
  
"live" mode:
  all_funds = st.session_state.live_funds_data ← from mfapi.in via NestJS
```

The switch in `app.py`:
```python
if st.session_state.data_mode == "live" and st.session_state.live_funds_data:
    all_funds = st.session_state.live_funds_data
else:
    all_funds = load_all_funds(st.session_state.uploaded_bytes)
```

When the user is in "New Analysis" page but hasn't loaded any data yet,
`live_funds_data` is empty `{}`, so the app falls back to default mode
for all the other pages — they won't show empty.

---

### Logic 7 — Fund Name as the Dict Key

The `all_funds` dict uses the **human-readable fund name as key**
(e.g. `"HDFC Top 100 Fund - Direct Plan - Growth"`).

For live data, the key comes from mfapi.in's `schemeName` field (the full name).
For default data, it comes from the Excel sheet tab names (abbreviated names).

**Implication:** The "Select Fund" dropdowns in Analysis, Predictions, etc.
will show full mfapi names (long) vs abbreviated Excel names (short).
This is fine — it's just display text, the dict key is what matters.

**If desired later:** We can let the user rename funds after loading (Phase 2).

---

### Logic 8 — What Happens to Failed Funds

If a fund fetch fails (e.g. wrong scheme code, network error), NestJS returns:
```json
{ "fund": "Some Fund", "code": 99999, "data": [], "error": "Request failed" }
```

In `load_funds_from_api()`:
- Funds with empty `data: []` are **skipped silently**
- If all funds fail, the function returns `{}` and shows an error
- If some fail, user sees a warning listing which ones failed

---

### Logic 9 — Minimum Data Check

Some analysis operations need a minimum number of data points:
- Rolling 30-day return: needs at least 30 records
- Predictions (ARIMA etc.): needs at least 60–100 records
- Correlation heatmap: needs enough overlap between fund date ranges

If a loaded fund has fewer than `5` records (e.g. too narrow a date range),
`_preprocess_from_api()` returns `None` and that fund is excluded.

**User-facing:** Show a warning in the "New Analysis" page:
`⚠️ "XYZ Fund" had only 12 trading days in the selected range — skipped.`

---

### Logic 10 — Streamlit Caching Strategy

| Data | Cache Method | TTL | Why |
|------|-------------|-----|-----|
| Default fund data (xlsx) | `@st.cache_data` | Session | Same file = same result |
| All scheme list | `@st.cache_data(ttl=86400)` | 24h | Schemes rarely change |
| Search results | No cache | — | User types dynamically |
| Live fund NAV | `st.session_state` | Session | User explicitly loaded it |
| Benchmark data (yfinance) | `@st.cache_data(ttl=3600)` | 1h | Already implemented |

---

## ❓ Open Questions for You to Decide

These decisions affect implementation and need your input before we build:

---

### Q1 — Fund Name Label in Sidebar Indicator

When a user loads live funds, should the sidebar show:
- **(A)** Just a green dot `🟢 Live Mode Active`
- **(B)** The fund names: `🟢 Live: HDFC Top 100, SBI Bluechip...`
- **(C)** A count: `🟢 Live: 3 funds loaded`

**My recommendation:** Option C — clean and informative.

---

### Q2 — Can User Load New Funds While Keeping Old Ones?

If user loads `HDFC + SBI` and then loads `Parag Parikh` separately, should:
- **(A)** Replace: new load replaces the old live_funds_data entirely
- **(B)** Merge: new funds are added to existing live_funds_data
- **(C)** Ask: "Replace existing data or add to it?"

**My recommendation:** Option A (replace) for simplicity in V1.
Option B is a nice-to-have for V2 — lets you compare many funds across sessions.

---

### Q3 — How Many Funds Can a User Select at Once?

Should there be a cap?
- **No cap**: User could select 50 funds, causing 50 simultaneous mfapi requests
- **Cap at 10**: Safe, fast, still useful
- **Cap at 20**: More flexible, slightly higher risk of rate limiting

**My recommendation:** Cap at **10 funds** per batch load. Show error if user selects more.

---

### Q4 — Default Date Range for New Analysis

When the "New Analysis" page opens, what should the default date range be?
- **(A)** Last 1 year: `today - 365` → `today`
- **(B)** Last 3 years: `today - 1095` → `today`
- **(C)** Last 5 years: `today - 1825` → `today`
- **(D)** No default, user must pick

**My recommendation:** Option B (3 years) — enough data for predictions and rolling metrics.

---

### Q5 — Where to Place "New Analysis" in Navigation?

In the sidebar, should it be:
- **(A)** At the TOP before the other pages (prominent, hard to miss)
- **(B)** At the BOTTOM after User Manual (keep default pages first)
- **(C)** Separate section with a visual divider labelled "Live Data"

**My recommendation:** Option C — clear visual separation, doesn't clutter the default nav.

---

### Q6 — What Happens to International Funds?

The current `data.xlsx` has 3 international funds:
`Invesco Pan European`, `Global Consumer Trends`, `EQQQ NASDAQ-100 ETF`.

mfapi.in is an **Indian MF database** — these international funds are likely
not in it. Users searching for them in "New Analysis" won't find them.

Options:
- **(A)** Accept this limitation, international funds only available in default mode
- **(B)** In default mode, keep them; in live mode, skip if not found on mfapi
- **(C)** Show a note in New Analysis: "Note: International funds may not be available"

**My recommendation:** Option C — honest communication, no workaround needed.

---

### Q7 — Should NestJS be a Permanent Running Service?

For V1 (local development), user must manually start:
```bash
cd fundscope-backend && npm run start:dev
```

**My recommendation:** Option A (Keep it manual) — opening two terminals lets the user easily see and manage both processes, view individual logs separately, and shut down each service independently using standard termination controls.
Option C for any deployment.

---

### Q8 — What to Show on the New Analysis Page Before Data is Loaded?

When user is on "New Analysis" and hasn't loaded anything yet:
- **(A)** Just show the search form (current plan)
- **(B)** Show the search form AND some example popular funds as quick-picks
- **(C)** Show a "suggested funds" list pulled from mfapi.in top-rated funds

**My recommendation:** Option B — pre-fill 5-6 popular funds as one-click options
(e.g. Parag Parikh Flexi Cap, HDFC Midcap, Mirae Asset Large Cap) so the
user can get started fast without typing.
