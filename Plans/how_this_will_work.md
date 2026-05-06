# FundScope — Architecture & Workflow

## 🧭 Overview

FundScope operates with **two modes** running side by side:

| Mode | Data Source | When Used |
|------|-------------|-----------|
| **Default Mode** | `data.xlsx` (existing) | App loads, shows existing pre-loaded funds |
| **Live API Mode**| Live data via FastAPI Backend | User clicks "New Analysis" in sidebar |

Both modes feed the **exact same analysis engine** — the same `render(all_funds: dict)` functions across all 6 pages. Zero changes are needed to the underlying analytics logic.

---

## 🏗️ Target Cloud Architecture

The application is structured into a modern, decoupled cloud-native stack:

1. **Frontend**: **Streamlit Community Cloud** (Hosts the Streamlit UI, making it publicly accessible).
2. **Backend**: **Azure Cloud Server** (Hosts the FastAPI application, serving as the core proxy and data aggregator).
3. **Database**: **Supabase** (PostgreSQL database for persistent storage, user accounts, and caching. Accessed *only* by the FastAPI backend).

```text
┌──────────────────────────────────────────────────────────────┐
│                    Supabase (Database)                       │
│  - Stores user configurations, cached NAV history, etc.      │
└────────────────────────┬─────────────────────────────────────┘
                         │ (SQL / PostgREST via supabase-py)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Azure Cloud (Backend)                       │
│             fundscope-backend (FastAPI)                      │
│                                                              │
│  Role: PROXY, AGGREGATOR & DB MANAGER                        │
│  ① Proxies search queries to mfapi.in                        │
│  ② Fetches NAV for multiple selected funds in parallel       │
│  ③ Connects to Supabase to read/write persistent data        │
│  ④ Returns one clean JSON response to Streamlit              │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP (REST API)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│          Streamlit Community Cloud (Frontend)                │
│                                                              │
│  Role: UI & VISUALIZATION                                    │
│  - Fetches data from Azure Backend                           │
│  - Runs ML predictions and probabilistic simulations         │
│  - Displays dynamic charts and UI components                 │
└──────────────────────────────────────────────────────────────┘
```

## 🔄 Detailed User Flow (Live Analysis Mode)

1. **User searches for funds**:
   Streamlit (Frontend) → `GET <AZURE_BACKEND_URL>/mf/search?q=term` → FastAPI (Backend) queries `mfapi.in`.
2. **User selects funds & date range**:
   User selects the desired funds and clicks "Load Data".
3. **Fetch NAV data**:
   Streamlit (Frontend) → `POST <AZURE_BACKEND_URL>/mf/fetch-batch`.
   FastAPI (Backend) processes this request, gathering data in parallel via `asyncio.gather` from `mfapi.in`, or retrieves cached histories directly from **Supabase**.
4. **Data Delivery**:
   FastAPI normalizes the dates and NAV values and sends a clean JSON payload back to Streamlit.
5. **Visualization**:
   Streamlit converts the JSON to Pandas DataFrames, calculates returns, runs the analysis engine (ML models, Backtesting), and renders the UI.
