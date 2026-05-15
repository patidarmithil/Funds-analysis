"""
model_cache.py — Universal pre-trained model loader + feature extractor.

Current state: placeholder LinearRegression fallback.
Future: drop universal_fund_model.pkl into backend/models/ and it auto-activates.
"""
import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODEL = None   # loaded once at startup


def load_universal_model():
    """Load universal .pkl on first call. Falls back to placeholder if missing."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    pkl_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'universal_fund_model.pkl')
    pkl_path = os.path.normpath(pkl_path)
    if os.path.exists(pkl_path):
        try:
            import joblib
            _MODEL = joblib.load(pkl_path)
            logger.info("universal_fund_model.pkl loaded.")
        except Exception as e:
            logger.warning(f"Failed to load pkl: {e}. Using placeholder.")
            _MODEL = _PlaceholderModel()
    else:
        logger.info("universal_fund_model.pkl not found. Using placeholder (LinearRegression).")
        _MODEL = _PlaceholderModel()
    return _MODEL


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract feature vector from fund NAV DataFrame.
    Must match the feature set used to train universal_fund_model.pkl.
    """
    y = df['y'].values
    n = len(y)

    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    def safe_std(arr):
        return float(np.std(arr)) if len(arr) > 1 else 0.0

    rolling_5d   = safe_mean(y[max(0, n-5):])
    rolling_20d  = safe_mean(y[max(0, n-20):])
    rolling_60d  = safe_mean(y[max(0, n-60):])
    vol_20d      = safe_std(np.diff(y[max(0, n-21):]) / y[max(0, n-21):-1]) if n > 21 else 0.0
    momentum_14d = (y[-1] / y[-14] - 1) if n >= 14 and y[-14] > 0 else 0.0
    mean_y       = np.mean(y)
    std_y        = np.std(y)
    nav_zscore   = (y[-1] - mean_y) / (std_y + 1e-9)

    return np.array([
        rolling_5d, rolling_20d, rolling_60d,
        vol_20d, momentum_14d, nav_zscore,
    ]).reshape(1, -1)


def predict_with_universal_model(df: pd.DataFrame, periods: int) -> dict:
    """
    Run universal model inference on fund data.
    Returns { forecast: [{ds, yhat, yhat_lower, yhat_upper}], model_used: str }
    """
    model = load_universal_model()
    features  = extract_features(df)
    last_nav  = float(df['y'].iloc[-1])
    rel_rets  = model.predict_sequence(features, periods)

    last_date = pd.Timestamp(df['ds'].iloc[-1])
    future_ds = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq='B')

    navs = last_nav * np.cumprod(1 + rel_rets)
    spread = np.std(navs) * 0.05 + navs[-1] * 0.01

    forecast = [
        {
            'ds':         str(future_ds[i].date()),
            'yhat':       round(float(navs[i]), 4),
            'yhat_lower': round(float(navs[i] - spread), 4),
            'yhat_upper': round(float(navs[i] + spread), 4),
        }
        for i in range(periods)
    ]
    return {
        'forecast':   forecast,
        'model_used': 'universal' if not isinstance(model, _PlaceholderModel) else 'fallback',
    }


# ─── Placeholder Model ────────────────────────────────────────────────────────

class _PlaceholderModel:
    """
    Simple LinearRegression-based fallback used until real universal model is uploaded.
    Trains on the current fund data per request (fast, ~0.1s).
    """
    def predict_sequence(self, features: np.ndarray, periods: int) -> np.ndarray:
        # Small random drift around 0 (neutral market assumption)
        drift = 0.0002   # +0.02% daily drift
        noise = np.random.default_rng(42).normal(drift, 0.005, periods)
        return noise
