"""
risk.py — VaR, CVaR, Drawdown analysis, Stress testing.
"""
import numpy as np
import pandas as pd
from config import STRESS_SCENARIOS


# ─── Value at Risk ────────────────────────────────────────────────────────────

def calculate_var(df: pd.DataFrame, confidence: float = 0.95) -> float:
    """Parametric VaR  at given confidence level (daily return basis)."""
    r = df['returns'].dropna()
    return float(np.percentile(r, (1 - confidence) * 100))


def calculate_cvar(df: pd.DataFrame, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) — mean of returns below VaR."""
    r   = df['returns'].dropna()
    var = calculate_var(df, confidence)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) > 0 else var


# ─── Drawdown series ─────────────────────────────────────────────────────────

def calculate_drawdown_series(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a df with columns: ds, drawdown (fraction 0 to -1)."""
    nav  = df['y'].values
    peak = np.maximum.accumulate(nav)
    dd   = (nav - peak) / peak
    return pd.DataFrame({'ds': df['ds'], 'drawdown': dd * 100})


def calculate_underwater(df: pd.DataFrame) -> pd.DataFrame:
    """Alias of drawdown_series for clarity."""
    return calculate_drawdown_series(df)


def calculate_recovery_periods(df: pd.DataFrame) -> list[dict]:
    """
    Identify drawdown episodes: start, trough, recovery date, duration.
    Returns a list of dicts sorted by severity.
    """
    nav   = df['y'].values
    dates = df['ds'].values
    peak  = np.maximum.accumulate(nav)
    dd    = (nav - peak) / peak

    episodes = []
    in_dd    = False
    start_i  = 0

    for i in range(1, len(dd)):
        if not in_dd and dd[i] < -0.01:
            in_dd   = True
            start_i = i - 1
        if in_dd:
            if nav[i] >= peak[start_i]:   # fully recovered
                trough_i = np.argmin(dd[start_i:i]) + start_i
                episodes.append(dict(
                    start    = pd.Timestamp(dates[start_i]).date(),
                    trough   = pd.Timestamp(dates[trough_i]).date(),
                    recovery = pd.Timestamp(dates[i]).date(),
                    drawdown = round(dd[trough_i] * 100, 2),
                    duration = i - start_i,
                ))
                in_dd = False

    # If still in drawdown at end
    if in_dd:
        trough_i = np.argmin(dd[start_i:]) + start_i
        episodes.append(dict(
            start    = pd.Timestamp(dates[start_i]).date(),
            trough   = pd.Timestamp(dates[trough_i]).date(),
            recovery = 'Ongoing',
            drawdown = round(dd[trough_i] * 100, 2),
            duration = len(dd) - start_i,
        ))

    return sorted(episodes, key=lambda e: e['drawdown'])


# ─── Stress test ─────────────────────────────────────────────────────────────

def stress_test(df: pd.DataFrame, investment: float = 100_000) -> pd.DataFrame:
    """
    For each scenario in STRESS_SCENARIOS, compute:
    - Final portfolio value
    - P&L
    """
    current_nav = df['y'].iloc[-1]
    rows = []
    for label, shock in STRESS_SCENARIOS.items():
        shocked_nav = current_nav * (1 + shock)
        units       = investment / current_nav
        final_value = units * shocked_nav
        rows.append(dict(
            Scenario    = label,
            Shock       = f"{shock*100:+.0f}%",
            Nav_After   = round(shocked_nav, 2),
            Portfolio   = round(final_value, 2),
            PnL         = round(final_value - investment, 2),
            PnL_Pct     = round(shock * 100, 1),
        ))
    return pd.DataFrame(rows)


# ─── Risk summary ─────────────────────────────────────────────────────────────

def get_risk_summary(df: pd.DataFrame, confidence: float = 0.95) -> dict:
    return dict(
        var_95  = calculate_var(df, 0.95),
        var_99  = calculate_var(df, 0.99),
        cvar_95 = calculate_cvar(df, 0.95),
        cvar_99 = calculate_cvar(df, 0.99),
        max_dd  = float(calculate_drawdown_series(df)['drawdown'].min()),
    )
