"""
data_loader.py — Load & preprocess mutual fund data.
Supports: default data.xlsx  OR  user-uploaded file.
"""
import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from config import DEFAULT_FILE, FUND_NAMES, TRADING_DAYS


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
