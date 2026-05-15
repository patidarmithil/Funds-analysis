"""
analysis.py — Comprehensive financial metrics for mutual fund analytics.
"""
import numpy as np
import pandas as pd
from config import RISK_FREE_RATE, TRADING_DAYS


# ─── Returns ──────────────────────────────────────────────────────────────────

def calculate_cagr(df: pd.DataFrame, years: float | None = None) -> float:
    """Compound Annual Growth Rate."""
    if len(df) < 2:
        return np.nan
    start, end = df['y'].iloc[0], df['y'].iloc[-1]
    if years is None:
        delta = (df['ds'].iloc[-1] - df['ds'].iloc[0]).days
        years = delta / 365.25
    if years <= 0 or start <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1


def calculate_annualized_return(df: pd.DataFrame) -> float:
    r = df['returns'].dropna()
    return (1 + r.mean()) ** TRADING_DAYS - 1


def calculate_rolling_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling period returns: 30, 90, 180, 365, 756 days."""
    out = df[['ds']].copy()
    for w, label in [(30, '1M'), (90, '3M'), (180, '6M'), (365, '1Y'), (756, '3Y')]:
        out[label] = df['y'].pct_change(w) * 100
    return out


def calculate_period_return(df: pd.DataFrame, days: int) -> float | None:
    if len(df) <= days:
        return None
    v0 = df['y'].iloc[-days - 1]
    v1 = df['y'].iloc[-1]
    return (v1 / v0 - 1) * 100 if v0 > 0 else None


# ─── Risk-adjusted metrics ────────────────────────────────────────────────────

def calculate_sharpe(df: pd.DataFrame, rf: float = RISK_FREE_RATE) -> float:
    r = df['returns'].dropna()
    if r.std() == 0:
        return np.nan
    daily_rf = (1 + rf) ** (1 / TRADING_DAYS) - 1
    excess   = r - daily_rf
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)


def calculate_sortino(df: pd.DataFrame, rf: float = RISK_FREE_RATE) -> float:
    r = df['returns'].dropna()
    daily_rf  = (1 + rf) ** (1 / TRADING_DAYS) - 1
    excess    = r - daily_rf
    downside  = excess[excess < 0]
    if len(downside) == 0:
        return np.nan
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)
    return excess.mean() * TRADING_DAYS / downside_std


def calculate_max_drawdown(df: pd.DataFrame) -> float:
    nav  = df['y']
    peak = nav.cummax()
    dd   = (nav - peak) / peak
    return float(dd.min())


def calculate_calmar(df: pd.DataFrame) -> float:
    ann_ret = calculate_annualized_return(df)
    mdd     = abs(calculate_max_drawdown(df))
    return ann_ret / mdd if mdd != 0 else np.nan


def calculate_volatility(df: pd.DataFrame) -> float:
    return df['returns'].dropna().std() * np.sqrt(TRADING_DAYS)


def calculate_rolling_volatility(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    out = df[['ds']].copy()
    out['volatility'] = df['returns'].rolling(window).std() * np.sqrt(TRADING_DAYS) * 100
    return out


# ─── Alpha & Beta ─────────────────────────────────────────────────────────────

def calculate_alpha_beta(fund_ret: pd.Series, bm_ret: pd.Series,
                         rf: float = RISK_FREE_RATE) -> dict:
    daily_rf = (1 + rf) ** (1 / TRADING_DAYS) - 1
    x = (bm_ret   - daily_rf).dropna()
    y = (fund_ret  - daily_rf).reindex(x.index).dropna()
    x = x.reindex(y.index)
    if len(x) < 30:
        return dict(alpha=np.nan, beta=np.nan, r_squared=np.nan)
    beta, alpha_d = np.polyfit(x, y, 1)
    residuals     = y - (alpha_d + beta * x)
    ss_res        = (residuals ** 2).sum()
    ss_tot        = ((y - y.mean()) ** 2).sum()
    r2            = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    alpha_ann     = alpha_d * TRADING_DAYS
    return dict(alpha=alpha_ann, beta=beta, r_squared=r2)


# ─── Distribution stats ───────────────────────────────────────────────────────

def calculate_skewness_kurtosis(df: pd.DataFrame) -> dict:
    r = df['returns'].dropna()
    return dict(skewness=float(r.skew()), kurtosis=float(r.kurt()))


# ─── Correlation matrix ───────────────────────────────────────────────────────

def calculate_correlation_matrix(all_funds: dict) -> pd.DataFrame:
    series = {}
    for name, df in all_funds.items():
        s = df.set_index('ds')['returns'].rename(name)
        series[name] = s
    combined = pd.DataFrame(series).dropna(how='all')
    return combined.corr()


# ─── Full summary row ─────────────────────────────────────────────────────────

def get_fund_summary(df: pd.DataFrame) -> dict:
    return dict(
        current_nav  = round(df['y'].iloc[-1], 2),
        ret_1m       = calculate_period_return(df, 30),
        ret_3m       = calculate_period_return(df, 90),
        ret_6m       = calculate_period_return(df, 180),
        ret_1y       = calculate_period_return(df, 365),
        cagr         = calculate_cagr(df),
        sharpe       = calculate_sharpe(df),
        sortino      = calculate_sortino(df),
        calmar       = calculate_calmar(df),
        max_drawdown = calculate_max_drawdown(df),
        volatility   = calculate_volatility(df),
        **calculate_skewness_kurtosis(df),
    )
