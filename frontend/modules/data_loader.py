"""
data_loader.py — Load & preprocess mutual fund data.
Supports: default data.xlsx  OR  user-uploaded file  OR  live mfapi.in via FastAPI backend.
"""
import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests as _requests

from config import DEFAULT_FILE, FUND_NAMES, TRADING_DAYS, BACKEND_URL


# ─── Primary loader ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_all_funds(uploaded_file=None) -> dict[str, pd.DataFrame]:
    """
    Returns a dict  { fund_name: DataFrame(ds, y, returns) }
    Reads from uploaded_file bytes if provided, else DEFAULT_FILE.
    """
    if uploaded_file is not None:
        raw = io.BytesIO(uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file)
    elif os.path.exists(DEFAULT_FILE):
        raw = DEFAULT_FILE
    else:
        return {}

    xl = pd.ExcelFile(raw)
    available = xl.sheet_names
    funds = {}
    for name in FUND_NAMES:
        if name in available:
            try:
                df = xl.parse(name)
                df = _preprocess(df, name)
                if df is not None:
                    funds[name] = df
            except Exception:
                pass
    return funds


def _preprocess(df: pd.DataFrame, name: str) -> pd.DataFrame | None:
    """Clean, rename columns and compute return series."""
    df.columns = df.columns.str.strip()

    # Flexible column detection
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    nav_col  = next((c for c in df.columns if 'nav'  in c.lower()), None)
    if date_col is None or nav_col is None:
        return None

    df = df[[date_col, nav_col]].copy()
    df.rename(columns={date_col: 'ds', nav_col: 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'], dayfirst=True, errors='coerce')
    df['y']  = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(subset=['ds', 'y'], inplace=True)
    df.sort_values('ds', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['returns']        = df['y'].pct_change()          # daily fractional
    df['log_returns']    = np.log(df['y'] / df['y'].shift(1))
    df['rolling_30_ret'] = df['returns'].rolling(30).mean()
    return df


# ─── Benchmark loader ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmark(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Download benchmark data from Yahoo Finance."""
    try:
        bm = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if bm.empty:
            return None
        bm = bm[['Close']].reset_index()
        bm.columns = ['ds', 'y']
        bm['returns'] = bm['y'].pct_change()
        return bm
    except Exception:
        return None


# ─── Helpers ─────────────────────────────────────────────────────────────────

def align_benchmark(fund_df: pd.DataFrame, bm_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return aligned daily return series for fund and benchmark."""
    merged = pd.merge(fund_df[['ds', 'returns']], bm_df[['ds', 'returns']],
                      on='ds', suffixes=('_fund', '_bm')).dropna()
    return merged['returns_fund'], merged['returns_bm']


def get_date_range(df: pd.DataFrame) -> tuple[str, str]:
    return str(df['ds'].min().date()), str(df['ds'].max().date())


def get_monthly_nav(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to end-of-month NAV."""
    tmp = df.set_index('ds')['y'].resample('ME').last().reset_index()
    tmp.columns = ['ds', 'y']
    tmp['returns'] = tmp['y'].pct_change()
    return tmp.dropna().reset_index(drop=True)


# ─── Live API loaders (FastAPI ↔ mfapi.in) ───────────────────────────────────

def search_schemes_api(query: str) -> list[dict]:
    """
    Search mfapi.in scheme names via FastAPI proxy.
    Returns [ {schemeCode: int, schemeName: str}, ... ]
    Returns empty list on any failure.
    """
    if not query or len(query.strip()) < 2:
        return []
    try:
        resp = _requests.get(
            f"{BACKEND_URL}/mf/search",
            params={"q": query.strip()},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def load_funds_from_api(
    selected_funds: list[dict],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """
    Calls the FastAPI backend to fetch NAV history for selected funds in parallel.

    Args:
        selected_funds: list of { schemeCode: int, schemeName: str }
        start_date:     "YYYY-MM-DD"
        end_date:       "YYYY-MM-DD"

    Returns:
        { fund_name: DataFrame } — exact same shape as load_all_funds().
        Funds that return no data or error are silently skipped.
    """
    if not selected_funds:
        return {}

    try:
        resp = _requests.post(
            f"{BACKEND_URL}/mf/fetch-batch",
            json={
                "funds": selected_funds,
                "startDate": start_date,
                "endDate": end_date,
            },
            timeout=90,   # generous timeout: parallel fetch of N funds
        )
        resp.raise_for_status()
        fund_list = resp.json()   # list of { fund, code, data: [{ds, y}], records, error? }
    except _requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to the FundScope backend. "
            "Make sure the FastAPI service is running on `http://localhost:8000`. "
            "Start it with: `cd backend && uvicorn main:app --reload --port 8000`"
        )
        return {}
    except Exception as e:
        st.error(f"❌ Backend request failed: {e}")
        return {}

    result: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []

    for entry in fund_list:
        name = entry.get("fund", "Unknown")
        raw_data = entry.get("data", [])

        if entry.get("error") or not raw_data:
            skipped.append(name)
            continue

        try:
            df = pd.DataFrame(raw_data)           # columns: ds, y
            df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
            df["y"]  = pd.to_numeric(df["y"], errors="coerce")
            df = _preprocess_live(df)
            if df is not None:
                result[name] = df
            else:
                skipped.append(f"{name} (too few records)")
        except Exception:
            skipped.append(name)

    if skipped:
        st.warning(f"⚠️ Skipped funds (no data or error): {', '.join(skipped)}")

    return result


def _preprocess_live(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Adds computed return columns to a raw {ds, y} DataFrame.
    Mirrors _preprocess() but assumes ds/y columns already exist and are typed.
    Returns None if there are fewer than 10 valid records.
    """
    df = df.dropna(subset=["ds", "y"])
    df = df.sort_values("ds").reset_index(drop=True)

    if len(df) < 10:
        return None

    df["returns"]        = df["y"].pct_change()
    df["log_returns"]    = np.log(df["y"] / df["y"].shift(1))
    df["rolling_30_ret"] = df["returns"].rolling(30).mean()
    return df


@st.cache_data(ttl=86400, show_spinner=False)   # cache for 24 hours
def get_all_schemes_cached() -> list[dict]:
    """
    Fetch the full list of all Indian MF schemes (~17k) from FastAPI.
    Cached for 24 hours — called once per day maximum.
    Returns [ {schemeCode: int, schemeName: str}, ... ]
    """
    try:
        resp = _requests.get(f"{BACKEND_URL}/mf/schemes", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def check_backend_health() -> bool:
    """Returns True if the FastAPI backend is reachable, False otherwise."""
    try:
        resp = _requests.get(f"{BACKEND_URL}/mf/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False
