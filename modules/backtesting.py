"""
backtesting.py — Strategy simulation and comparison.
Strategies: Buy & Hold, SIP, Value Averaging.
"""
import numpy as np
import pandas as pd
from config import BACKTEST


# ─── Strategies ───────────────────────────────────────────────────────────────

def buy_and_hold(df: pd.DataFrame, investment: float = BACKTEST['lumpsum']) -> pd.DataFrame:
    """
    Invest `investment` at the first available NAV.
    Track portfolio value daily.
    """
    start_nav = df['y'].iloc[0]
    units     = investment / start_nav
    out       = df[['ds', 'y']].copy()
    out['portfolio_value'] = units * out['y']
    out['strategy']        = 'Buy & Hold'
    return out


def sip_strategy(df: pd.DataFrame, monthly: float = BACKTEST['monthly_sip']) -> pd.DataFrame:
    """
    Invest `monthly` at the start of each calendar month.
    Track portfolio value daily.
    """
    total_units = 0.0
    purchase_dates = df['ds'].dt.to_period('M').drop_duplicates().dt.to_timestamp()
    purchase_set   = set(purchase_dates)

    records = []
    for _, row in df.iterrows():
        # Buy on first trading day of each month
        if row['ds'].to_period('M').to_timestamp() in purchase_set and (
            not records or row['ds'].to_period('M') != pd.Timestamp(records[-1][0]).to_period('M')
        ):
            total_units += monthly / row['y']
        records.append((row['ds'], row['y'], total_units * row['y']))

    out = pd.DataFrame(records, columns=['ds', 'y', 'portfolio_value'])
    out['strategy'] = 'SIP'
    return out


def value_averaging(df: pd.DataFrame,
                    monthly_target_growth: float = 0.01,
                    initial: float = BACKTEST['lumpsum']) -> pd.DataFrame:
    """
    Value Averaging: adjust contribution so portfolio hits a rising target.
    Monthly target = initial * (1 + monthly_target_growth)^t
    """
    total_units  = 0.0
    total_invest = 0.0
    month_count  = 0
    last_period  = None

    records = []
    for _, row in df.iterrows():
        cur_period = row['ds'].to_period('M')
        if cur_period != last_period:
            month_count += 1
            last_period  = cur_period
            target       = initial * (1 + monthly_target_growth) ** month_count
            current_val  = total_units * row['y']
            contribution = target - current_val
            if contribution > 0:
                total_units  += contribution / row['y']
                total_invest += contribution
        records.append((row['ds'], row['y'], total_units * row['y']))

    out = pd.DataFrame(records, columns=['ds', 'y', 'portfolio_value'])
    out['strategy'] = 'Value Averaging'
    return out


# ─── Metrics ─────────────────────────────────────────────────────────────────

def portfolio_metrics(strategy_df: pd.DataFrame, monthly_invest: float | None = None) -> dict:
    """Compute key performance metrics for a strategy output dataframe."""
    pv    = strategy_df['portfolio_value']
    ds    = strategy_df['ds']
    years = (ds.iloc[-1] - ds.iloc[0]).days / 365.25

    total_invested = monthly_invest * int(years * 12) if monthly_invest else pv.iloc[0]
    final_value    = pv.iloc[-1]
    total_return   = (final_value - total_invested) / total_invested * 100
    cagr           = (final_value / total_invested) ** (1 / years) - 1 if years > 0 else np.nan

    dd   = (pv - pv.cummax()) / pv.cummax()
    mdd  = float(dd.min()) * 100

    return dict(
        total_invested = round(total_invested, 2),
        final_value    = round(final_value, 2),
        total_return   = round(total_return, 2),
        cagr           = round(cagr * 100, 2),
        max_drawdown   = round(mdd, 2),
    )


# ─── Compare all ─────────────────────────────────────────────────────────────

def compare_strategies(df: pd.DataFrame,
                       lumpsum: float = BACKTEST['lumpsum'],
                       monthly: float = BACKTEST['monthly_sip']) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    - combined df for plotting (all strategies, columns: ds, portfolio_value, strategy)
    - summary metrics df
    """
    bnh = buy_and_hold(df, lumpsum)
    sip = sip_strategy(df, monthly)
    va  = value_averaging(df, initial=lumpsum)

    combined = pd.concat([
        bnh[['ds', 'portfolio_value', 'strategy']],
        sip[['ds', 'portfolio_value', 'strategy']],
        va [['ds', 'portfolio_value', 'strategy']],
    ], ignore_index=True)

    summary = pd.DataFrame([
        {'Strategy': 'Buy & Hold',       **portfolio_metrics(bnh, None)},
        {'Strategy': 'SIP',              **portfolio_metrics(sip, monthly)},
        {'Strategy': 'Value Averaging',  **portfolio_metrics(va, None)},
    ])
    return combined, summary
