"""
simulation.py — Monte Carlo simulation (no Streamlit).
"""
import numpy as np
import pandas as pd
from config import MONTE_CARLO


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
    dt     = 1
    drift  = (mu - 0.5 * sigma ** 2) * dt
    shocks = rng.standard_normal((iterations, days))
    log_returns = drift + sigma * np.sqrt(dt) * shocks
    paths  = last_nav * np.exp(np.cumsum(log_returns, axis=1))
    start  = np.full((iterations, 1), last_nav)
    return np.hstack([start, paths])


def get_percentile_bands(paths: np.ndarray,
                         percentiles: list = [5, 25, 50, 75, 95]) -> dict:
    return {p: np.percentile(paths, p, axis=0).tolist() for p in percentiles}


def final_value_distribution(paths: np.ndarray) -> np.ndarray:
    return paths[:, -1]


def simulation_summary(paths: np.ndarray,
                        initial_investment: float = 10_000,
                        last_nav: float = 1.0) -> dict:
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
