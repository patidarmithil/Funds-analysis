"""
data_loader.py — Load & preprocess mutual fund data (no Streamlit).
"""
import os
import io
import pandas as pd
import numpy as np
import yfinance as yf

from config import DEFAULT_FILE, FUND_NAMES, TRADING_DAYS


# ─── Primary loader ──────────────────────────────────────────────────────────

def load_all_funds(file_path: str | None = None, file_bytes: bytes | None = None) -> dict:
    """
    Returns { fund_name: DataFrame(ds, y, returns) }
    Priority: file_bytes > file_path > DEFAULT_FILE
    """
    if file_bytes is not None:
        raw = io.BytesIO(file_bytes)
    elif file_path and os.path.exists(file_path):
        raw = file_path
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
    df.columns = df.columns.str.strip()
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
    df['returns']        = df['y'].pct_change()
    df['log_returns']    = np.log(df['y'] / df['y'].shift(1))
    df['rolling_30_ret'] = df['returns'].rolling(30).mean()
    return df


def preprocess_live(df: pd.DataFrame) -> pd.DataFrame | None:
    """Add computed columns to a raw {ds, y} DataFrame from live API."""
    df = df.dropna(subset=['ds', 'y'])
    df = df.sort_values('ds').reset_index(drop=True)
    if len(df) < 10:
        return None
    df['returns']        = df['y'].pct_change()
    df['log_returns']    = np.log(df['y'] / df['y'].shift(1))
    df['rolling_30_ret'] = df['returns'].rolling(30).mean()
    return df


# ─── Benchmark loader ────────────────────────────────────────────────────────

def load_benchmark(ticker: str, start: str, end: str) -> pd.DataFrame | None:
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

def align_benchmark(fund_df: pd.DataFrame, bm_df: pd.DataFrame):
    merged = pd.merge(fund_df[['ds', 'returns']], bm_df[['ds', 'returns']],
                      on='ds', suffixes=('_fund', '_bm')).dropna()
    return merged['returns_fund'], merged['returns_bm']


def get_date_range(df: pd.DataFrame):
    return str(df['ds'].min().date()), str(df['ds'].max().date())


def get_monthly_nav(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.set_index('ds')['y'].resample('ME').last().reset_index()
    tmp.columns = ['ds', 'y']
    tmp['returns'] = tmp['y'].pct_change()
    return tmp.dropna().reset_index(drop=True)


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-serialisable list of {ds, y} dicts."""
    out = df[['ds', 'y']].copy()
    out['ds'] = out['ds'].astype(str)
    return out.to_dict(orient='records')


def records_to_df(records: list[dict]) -> pd.DataFrame | None:
    """Convert [{ds, y}] list back to preprocessed DataFrame."""
    if not records:
        return None
    df = pd.DataFrame(records)
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y']  = pd.to_numeric(df['y'], errors='coerce')
    return preprocess_live(df)
