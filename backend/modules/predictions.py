"""
predictions.py — Robust NAV forecasting models (no Streamlit).
"""
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model   import LinearRegression, Ridge
from sklearn.ensemble       import RandomForestRegressor
from sklearn.svm            import SVR
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import mean_squared_error, mean_absolute_error
from xgboost                import XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
from prophet                import Prophet

warnings.filterwarnings('ignore')


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _future_dates(df: pd.DataFrame, periods: int) -> pd.DatetimeIndex:
    last = pd.Timestamp(df['ds'].iloc[-1])
    return pd.date_range(last + pd.Timedelta(days=1), periods=periods, freq='B')


def _build_result(future_ds, yhat: np.ndarray, ci_pct: float = 0.05) -> pd.DataFrame:
    spread = np.std(yhat) * ci_pct + yhat[-1] * 0.01
    return pd.DataFrame({
        'ds':         pd.Series(future_ds),
        'yhat':       yhat,
        'yhat_lower': yhat - spread,
        'yhat_upper': yhat + spread,
    })


def _log_return_features(series: pd.Series, n_lags: int = 10) -> tuple:
    log_ret = np.log(series / series.shift(1)).dropna()
    df = pd.DataFrame({'r': log_ret.values})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['r'].shift(lag)
    df['rm5']  = df['r'].rolling(5).mean()
    df['rm20'] = df['r'].rolling(20).mean()
    df['rs5']  = df['r'].rolling(5).std()
    df['rs20'] = df['r'].rolling(20).std()
    df.dropna(inplace=True)
    X = df.drop(columns='r')
    y = df['r']
    return X, y


def _forecast_from_returns(model, last_nav: float, last_returns: np.ndarray,
                            periods: int, scaler=None, n_lags: int = 10) -> np.ndarray:
    hist = list(last_returns[-60:])
    nav  = last_nav
    navs = []
    for _ in range(periods):
        lag_feats = hist[-n_lags:][::-1]
        rm5  = float(np.mean(hist[-5:]))
        rm20 = float(np.mean(hist[-20:]))
        rs5  = float(np.std(hist[-5:]))
        rs20 = float(np.std(hist[-20:]))
        row = np.array(lag_feats + [rm5, rm20, rs5, rs20], dtype=float).reshape(1, -1)
        if scaler is not None:
            row = scaler.transform(row)
        pred_return = float(model.predict(row)[0])
        pred_return = np.clip(pred_return, -0.05, 0.05)
        nav = nav * np.exp(pred_return)
        hist.append(pred_return)
        navs.append(nav)
    return np.array(navs)


# ─── Individual Models ────────────────────────────────────────────────────────

def _run_prophet(train: pd.DataFrame, periods: int) -> pd.DataFrame:
    df_p = train[['ds', 'y']].copy()
    m = Prophet(
        growth='linear', seasonality_mode='multiplicative',
        yearly_seasonality=True, weekly_seasonality=False,
        daily_seasonality=False, interval_width=0.80,
        changepoint_prior_scale=0.05,
    )
    m.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
    m.fit(df_p)
    future   = m.make_future_dataframe(periods=periods, freq='B')
    forecast = m.predict(future)
    result   = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
    result['ds'] = pd.to_datetime(result['ds'])
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        result[col] = result[col].clip(lower=train['y'].min() * 0.5)
    return result.reset_index(drop=True)


def _run_arima(train: pd.DataFrame, periods: int) -> pd.DataFrame:
    series = train['y'].values
    try:
        m = StatsARIMA(series, order=(1, 1, 1)).fit(disp=False)
    except Exception:
        m = StatsARIMA(series, order=(0, 1, 0)).fit(disp=False)
    fc_obj    = m.get_forecast(steps=periods)
    fc_mean   = fc_obj.predicted_mean
    fc_ci     = fc_obj.conf_int(alpha=0.20)
    future_ds = _future_dates(train, periods)
    return pd.DataFrame({
        'ds':         future_ds,
        'yhat':       np.maximum(fc_mean, 0.01),
        'yhat_lower': np.maximum(fc_ci.iloc[:, 0].values, 0.01),
        'yhat_upper': fc_ci.iloc[:, 1].values,
    })


def _run_holt_winters(train: pd.DataFrame, periods: int) -> pd.DataFrame:
    series = train['y'].values
    n      = len(series)
    season = min(252, n // 3)
    try:
        m = ExponentialSmoothing(
            series, trend='add',
            seasonal='add' if season >= 4 else None,
            seasonal_periods=season if season >= 4 else None,
        ).fit(optimized=True)
    except Exception:
        m = ExponentialSmoothing(series, trend='add').fit(optimized=True)
    fc        = m.forecast(periods)
    future_ds = _future_dates(train, periods)
    fc        = np.maximum(fc, 0.01)
    spread    = np.abs(fc) * 0.04
    return pd.DataFrame({
        'ds': future_ds, 'yhat': fc,
        'yhat_lower': fc - spread, 'yhat_upper': fc + spread,
    })


def _run_sklearn_model(train: pd.DataFrame, periods: int, model_cls, **kwargs) -> pd.DataFrame:
    n_lags   = 10
    series   = train['y']
    log_rets = np.log(series / series.shift(1)).dropna()
    X, y = _log_return_features(series, n_lags)
    need_scale = model_cls in (SVR, Ridge, LinearRegression)
    scaler     = StandardScaler() if need_scale else None
    X_tr       = scaler.fit_transform(X) if need_scale else X.values
    model = model_cls(**kwargs)
    model.fit(X_tr, y.values)
    future_navs = _forecast_from_returns(
        model, last_nav=float(series.iloc[-1]),
        last_returns=log_rets.values, periods=periods, scaler=scaler, n_lags=n_lags,
    )
    return _build_result(_future_dates(train, periods), future_navs)


def _run_xgboost(train: pd.DataFrame, periods: int) -> pd.DataFrame:
    n_lags   = 10
    series   = train['y']
    log_rets = np.log(series / series.shift(1)).dropna()
    X, y = _log_return_features(series, n_lags)
    model = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    )
    model.fit(X, y.values)
    future_navs = _forecast_from_returns(
        model, last_nav=float(series.iloc[-1]),
        last_returns=log_rets.values, periods=periods, n_lags=n_lags,
    )
    return _build_result(_future_dates(train, periods), future_navs)


# ─── Dispatch Map ─────────────────────────────────────────────────────────────

_MODEL_FNS = {
    'Prophet':           _run_prophet,
    'ARIMA':             _run_arima,
    'Holt-Winters':      _run_holt_winters,
    'Linear Regression': lambda tr, p: _run_sklearn_model(tr, p, LinearRegression),
    'Ridge Regression':  lambda tr, p: _run_sklearn_model(tr, p, Ridge, alpha=0.5),
    'Random Forest':     lambda tr, p: _run_sklearn_model(
        tr, p, RandomForestRegressor, n_estimators=200, random_state=42, n_jobs=-1
    ),
    'SVR':               lambda tr, p: _run_sklearn_model(
        tr, p, SVR, kernel='rbf', C=10.0, epsilon=0.005
    ),
    'XGBoost':           _run_xgboost,
}


# ─── Evaluation ──────────────────────────────────────────────────────────────

def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true[:len(y_pred)]
    y_pred = y_pred[:len(y_true)]
    rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae    = float(mean_absolute_error(y_true, y_pred))
    mape   = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2     = float(1 - ss_res / (ss_tot + 1e-9))
    return dict(rmse=round(rmse, 4), mae=round(mae, 4), mape=round(mape, 2), r2=round(r2, 4))


# ─── Public API ──────────────────────────────────────────────────────────────

def run_model(df: pd.DataFrame, model_name: str, future_periods: int = 90):
    n_test  = max(30, int(len(df) * 0.20))
    train   = df.iloc[:-n_test].copy().reset_index(drop=True)
    test    = df.iloc[-n_test:].copy().reset_index(drop=True)
    fn      = _MODEL_FNS[model_name]
    val_fc  = fn(train, n_test)
    y_pred  = val_fc['yhat'].values[:n_test]
    y_true  = test['y'].values[:len(y_pred)]
    metrics = _evaluate(y_true, y_pred)
    full_fc = fn(df.reset_index(drop=True), future_periods)
    return full_fc, metrics


def run_ensemble(df: pd.DataFrame, model_names: list, future_periods: int = 90,
                 progress_cb=None) -> dict:
    n_test  = max(30, int(len(df) * 0.20))
    train   = df.iloc[:-n_test].copy().reset_index(drop=True)
    test    = df.iloc[-n_test:].copy().reset_index(drop=True)
    y_true  = test['y'].values
    val_preds      = {}
    full_forecasts = {}
    model_metrics  = {}
    for i, name in enumerate(model_names):
        if progress_cb:
            progress_cb(i / len(model_names), f"Training {name}…")
        try:
            fn  = _MODEL_FNS[name]
            vfc = fn(train, n_test)
            yp  = vfc['yhat'].values[:len(y_true)]
            val_preds[name]     = yp
            model_metrics[name] = _evaluate(y_true[:len(yp)], yp)
            ffc = fn(df.reset_index(drop=True), future_periods)
            full_forecasts[name] = ffc
        except Exception:
            pass
    if len(val_preds) < 2:
        weights = {n: 1 / len(val_preds) for n in val_preds}
    else:
        models_list = list(val_preds.keys())
        min_len     = min(len(val_preds[m]) for m in models_list)
        min_len     = min(min_len, len(y_true))
        pred_matrix = np.column_stack([val_preds[m][:min_len] for m in models_list])
        actual      = y_true[:min_len]
        def objective(w):
            return float(np.sqrt(np.mean((pred_matrix @ w - actual) ** 2)))
        n   = len(models_list)
        res = minimize(
            objective, x0=np.ones(n) / n, method='SLSQP',
            bounds=[(0, 1)] * n,
            constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
        )
        w_opt = np.clip(res.x, 0, 1)
        w_opt /= w_opt.sum()
        weights = {models_list[i]: round(float(w_opt[i]), 4) for i in range(n)}
    if val_preds:
        min_len_val = min(len(v) for v in val_preds.values())
        min_len_val = min(min_len_val, len(y_true))
        ens_val     = sum(val_preds[m][:min_len_val] * weights.get(m, 0) for m in val_preds)
        ens_metrics = _evaluate(y_true[:min_len_val], ens_val)
    else:
        ens_metrics = {}
    if full_forecasts:
        ref_fc    = next(iter(full_forecasts.values()))
        future_ds = ref_fc['ds'].values
        n_out     = len(future_ds)
        ens_yhat  = np.zeros(n_out)
        ens_lower = np.zeros(n_out)
        ens_upper = np.zeros(n_out)
        for name, w in weights.items():
            if name in full_forecasts:
                ffc = full_forecasts[name]
                ens_yhat  += w * ffc['yhat'].values[:n_out]
                ens_lower += w * ffc['yhat_lower'].values[:n_out]
                ens_upper += w * ffc['yhat_upper'].values[:n_out]
        ens_forecast = pd.DataFrame({
            'ds': future_ds, 'yhat': ens_yhat,
            'yhat_lower': ens_lower, 'yhat_upper': ens_upper,
        })
    else:
        ens_forecast = pd.DataFrame()
    best_solo = min(model_metrics, key=lambda m: model_metrics[m].get('rmse', 1e9)) \
                if model_metrics else None
    recommendations = {
        'best_individual':      best_solo,
        'best_individual_rmse': model_metrics[best_solo]['rmse'] if best_solo else None,
        'ensemble_rmse':        ens_metrics.get('rmse'),
        'top_weights':          sorted(weights.items(), key=lambda x: -x[1])[:3],
        'improvement_pct':      round(
            (model_metrics[best_solo]['rmse'] - ens_metrics.get('rmse', 0))
            / model_metrics[best_solo]['rmse'] * 100, 1
        ) if best_solo and ens_metrics.get('rmse') else None,
    }
    if progress_cb:
        progress_cb(1.0, "Done!")
    return dict(
        weights=weights, model_metrics=model_metrics,
        ensemble_metrics=ens_metrics, ensemble_forecast=ens_forecast,
        recommendations=recommendations,
    )
