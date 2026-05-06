"""
simulation.py — Monte Carlo simulation & scenario analysis.
"""
import numpy as np
import pandas as pd
from config import MONTE_CARLO


# ─── Monte Carlo ─────────────────────────────────────────────────────────────

def monte_carlo(df: pd.DataFrame,
                days: int = MONTE_CARLO['days'],
                iterations: int = MONTE_CARLO['iterations'],
                seed: int = 42) -> np.ndarray:
    """
    Geometric Brownian Motion simulation.
    Returns shape (iterations, days+1) array of price paths.
    """
    rng      = np.random.default_rng(seed)
    returns  = df['returns'].dropna()
    mu       = returns.mean()
    sigma    = returns.std()
    last_nav = df['y'].iloc[-1]

    # GBM: S_{t+1} = S_t * exp((mu - 0.5*sigma²)*dt + sigma*sqrt(dt)*Z)
    dt    = 1
    drift = (mu - 0.5 * sigma ** 2) * dt
    shocks = rng.standard_normal((iterations, days))
    log_returns = drift + sigma * np.sqrt(dt) * shocks
    paths = last_nav * np.exp(np.cumsum(log_returns, axis=1))
    # Prepend current price
    start  = np.full((iterations, 1), last_nav)
    return np.hstack([start, paths])


def get_percentile_bands(paths: np.ndarray,
                         percentiles: list[int] = [5, 25, 50, 75, 95]) -> dict[int, np.ndarray]:
    """Return dict of {percentile: 1-D array of length n_days}."""
    return {p: np.percentile(paths, p, axis=0) for p in percentiles}


def final_value_distribution(paths: np.ndarray) -> np.ndarray:
    """Array of final values (last column) across all paths."""
    return paths[:, -1]


def simulation_summary(paths: np.ndarray, initial_investment: float = 10_000,
                        last_nav: float = 1.0) -> dict:
    """
    Key statistics from the simulation.
    Converts price paths to portfolio values.
    units = investment / last_nav
    """
    units    = initial_investment / last_nav
    finals   = final_value_distribution(paths) * units / last_nav * last_nav
    # Simpler: portfolio_finals = (finals / last_nav) * initial_investment
    portfolio_finals = (paths[:, -1] / paths[:, 0]) * initial_investment

    return dict(
        mean        = float(np.mean(portfolio_finals)),
        median      = float(np.median(portfolio_finals)),
        p5          = float(np.percentile(portfolio_finals, 5)),
        p25         = float(np.percentile(portfolio_finals, 25)),
        p75         = float(np.percentile(portfolio_finals, 75)),
        p95         = float(np.percentile(portfolio_finals, 95)),
        prob_profit = float(np.mean(portfolio_finals > initial_investment) * 100),
        prob_loss20 = float(np.mean(portfolio_finals < initial_investment * 0.8) * 100),
    )
