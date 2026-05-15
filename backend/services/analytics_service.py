"""
analytics_service.py — Orchestrates ML computation for all analytics endpoints.
Wraps modules/* functions and serialises results to JSON-safe dicts.
"""
import numpy as np
import pandas as pd
from modules.data_loader    import records_to_df, load_benchmark, align_benchmark, get_date_range
from modules.analysis       import (
    get_fund_summary, calculate_correlation_matrix, calculate_rolling_returns,
    calculate_alpha_beta, calculate_rolling_volatility,
)
from modules.predictions    import run_model, run_ensemble
from modules.risk           import (
    get_risk_summary, calculate_drawdown_series, calculate_recovery_periods, stress_test,
    calculate_var, calculate_cvar,
)
from modules.simulation     import monte_carlo, get_percentile_bands, simulation_summary
from modules.backtesting    import compare_strategies, buy_and_hold, sip_strategy, portfolio_metrics
from services.model_cache   import predict_with_universal_model


def _safe(v):
    """Convert numpy/nan to JSON-safe Python types."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _summary_safe(d: dict) -> dict:
    return {k: _safe(v) for k, v in d.items()}


# ─── Overview ─────────────────────────────────────────────────────────────────

def compute_overview(funds_data: dict) -> dict:
    """
    funds_data: { fund_name: [{ds, y}] }
    Returns: { summary: [...], correlation_matrix: {...} }
    """
    dfs = {}
    summary_rows = []
    for name, records in funds_data.items():
        df = records_to_df(records)
        if df is None:
            continue
        dfs[name] = df
        row = get_fund_summary(df)
        row['fund'] = name
        summary_rows.append(_summary_safe(row))

    # Correlation matrix
    corr = {}
    if len(dfs) >= 2:
        corr_df = calculate_correlation_matrix(dfs)
        corr = {col: {row: _safe(v) for row, v in corr_df[col].items()} for col in corr_df.columns}

    return {'summary': summary_rows, 'correlation_matrix': corr}


# ─── Fund Summary ─────────────────────────────────────────────────────────────

def compute_fund_summary(fund_name: str, records: list) -> dict:
    df = records_to_df(records)
    if df is None:
        return {'error': 'Insufficient data'}
    s = get_fund_summary(df)
    s['fund'] = fund_name
    return _summary_safe(s)


# ─── Analysis ─────────────────────────────────────────────────────────────────

def compute_analysis(fund_name: str, records: list,
                     benchmark_ticker: str | None = None,
                     date_range: list | None = None) -> dict:
    df = records_to_df(records)
    if df is None:
        return {'error': 'Insufficient data'}

    # Date filter
    if date_range and len(date_range) == 2:
        df = df[(df['ds'] >= pd.Timestamp(date_range[0])) &
                (df['ds'] <= pd.Timestamp(date_range[1]))].reset_index(drop=True)

    nav_series = [{'ds': str(r['ds'].date()), 'y': round(float(r['y']), 4)} for _, r in df.iterrows()]
    rolling    = calculate_rolling_returns(df)
    rolling_out = []
    for _, r in rolling.iterrows():
        rolling_out.append({
            'ds': str(r['ds'].date()),
            '1M': _safe(r.get('1M')), '3M': _safe(r.get('3M')),
            '6M': _safe(r.get('6M')), '1Y': _safe(r.get('1Y')),
        })

    result = {
        'fund':        fund_name,
        'nav_series':  nav_series,
        'rolling':     rolling_out,
        'date_range':  list(get_date_range(df)),
    }

    # Benchmark alpha/beta
    if benchmark_ticker:
        start, end = get_date_range(df)
        bm_df = load_benchmark(benchmark_ticker, start, end)
        if bm_df is not None:
            fund_ret, bm_ret = align_benchmark(df, bm_df)
            ab = calculate_alpha_beta(fund_ret, bm_ret)
            result['alpha']     = _safe(ab['alpha'])
            result['beta']      = _safe(ab['beta'])
            result['r_squared'] = _safe(ab['r_squared'])
            result['benchmark_series'] = [
                {'ds': str(r['ds'].date()), 'y': round(float(r['y']), 4)}
                for _, r in bm_df.iterrows()
            ]

    return result


# ─── Predictions ─────────────────────────────────────────────────────────────

def compute_predict(fund_name: str, records: list,
                    model_name: str, periods: int) -> dict:
    df = records_to_df(records)
    if df is None:
        return {'error': 'Insufficient data'}
    if model_name == 'Universal':
        result = predict_with_universal_model(df, periods)
        result['fund'] = fund_name
        result['model'] = model_name
        return result
    try:
        fc, metrics = run_model(df, model_name, future_periods=periods)
        forecast = [
            {
                'ds':         str(r['ds'].date()) if hasattr(r['ds'], 'date') else str(r['ds'])[:10],
                'yhat':       round(float(r['yhat']), 4),
                'yhat_lower': round(float(r['yhat_lower']), 4),
                'yhat_upper': round(float(r['yhat_upper']), 4),
            }
            for _, r in fc.iterrows()
        ]
        return {
            'fund':       fund_name,
            'model':      model_name,
            'model_used': 'trained',
            'forecast':   forecast,
            'metrics':    _summary_safe(metrics),
        }
    except Exception as e:
        return {'error': str(e)}


def compute_predict_ensemble(fund_name: str, records: list,
                              model_names: list, periods: int) -> dict:
    df = records_to_df(records)
    if df is None:
        return {'error': 'Insufficient data'}
    try:
        result = run_ensemble(df, model_names, future_periods=periods)
        ef     = result['ensemble_forecast']
        forecast = []
        if not ef.empty:
            for _, r in ef.iterrows():
                forecast.append({
                    'ds':         str(r['ds'])[:10],
                    'yhat':       round(float(r['yhat']), 4),
                    'yhat_lower': round(float(r['yhat_lower']), 4),
                    'yhat_upper': round(float(r['yhat_upper']), 4),
                })
        return {
            'fund':             fund_name,
            'ensemble_forecast': forecast,
            'weights':          result['weights'],
            'model_metrics':    {k: _summary_safe(v) for k, v in result['model_metrics'].items()},
            'recommendations':  result['recommendations'],
        }
    except Exception as e:
        return {'error': str(e)}


# ─── Risk ─────────────────────────────────────────────────────────────────────

def compute_risk(fund_name: str, records: list,
                 investment_amount: float = 100_000,
                 confidence: float = 0.95) -> dict:
    df = records_to_df(records)
    if df is None:
        return {'error': 'Insufficient data'}

    summary = get_risk_summary(df, confidence)
    dd_df   = calculate_drawdown_series(df)
    recovery = calculate_recovery_periods(df)
    stress   = stress_test(df, investment_amount)

    # Return distribution histogram data
    returns = df['returns'].dropna() * 100
    hist, edges = np.histogram(returns, bins=60, density=True)
    dist = [{'x': round(float((edges[i]+edges[i+1])/2), 4), 'density': round(float(hist[i]), 6)}
            for i in range(len(hist))]

    return {
        'fund':             fund_name,
        'var_95':           _safe(summary['var_95']),
        'var_99':           _safe(summary['var_99']),
        'cvar_95':          _safe(summary['cvar_95']),
        'cvar_99':          _safe(summary['cvar_99']),
        'max_dd':           _safe(summary['max_dd']),
        'drawdown_series':  [{'ds': str(r['ds'].date()), 'drawdown': round(float(r['drawdown']), 4)}
                             for _, r in dd_df.iterrows()],
        'return_distribution': dist,
        'stress_test':      stress.to_dict(orient='records'),
        'recovery_periods': recovery,
    }


# ─── Backtest ─────────────────────────────────────────────────────────────────

def compute_backtest(fund_name: str, records: list,
                     lumpsum: float = 10_000,
                     monthly_sip: float = 1_000,
                     start_date: str | None = None) -> dict:
    df = records_to_df(records)
    if df is None:
        return {'error': 'Insufficient data'}

    if start_date:
        df = df[df['ds'] >= pd.Timestamp(start_date)].reset_index(drop=True)

    bnh = buy_and_hold(df, lumpsum)
    sip = sip_strategy(df, monthly_sip)

    nav_growth = []
    for i, row in bnh.iterrows():
        nav_growth.append({
            'ds':             str(row['ds'].date()),
            'nav':            round(float(row['y']), 4),
            'lumpsum_value':  round(float(row['portfolio_value']), 2),
            'sip_value':      round(float(sip.iloc[i]['portfolio_value']), 2) if i < len(sip) else None,
        })

    return {
        'fund':           fund_name,
        'lumpsum_result': _summary_safe(portfolio_metrics(bnh, None)),
        'sip_result':     _summary_safe(portfolio_metrics(sip, monthly_sip)),
        'nav_growth':     nav_growth,
    }


# ─── Simulation ───────────────────────────────────────────────────────────────

def compute_simulate(fund_name: str, records: list,
                     iterations: int = 1000, days: int = 252,
                     investment: float = 10_000) -> dict:
    df = records_to_df(records)
    if df is None:
        return {'error': 'Insufficient data'}

    paths = monte_carlo(df, days=days, iterations=iterations)
    bands = get_percentile_bands(paths, [5, 25, 50, 75, 95])
    summary = simulation_summary(paths, investment, df['y'].iloc[-1])

    # Sample 200 paths for frontend canvas rendering
    sample_idx   = np.random.default_rng(0).choice(iterations, min(200, iterations), replace=False)
    sample_paths = paths[sample_idx].tolist()

    # Final value distribution
    finals = (paths[:, -1] / paths[:, 0]) * investment
    hist, edges = np.histogram(finals, bins=60, density=True)
    fin_dist = [{'value': round(float((edges[i]+edges[i+1])/2), 2), 'density': round(float(hist[i]), 8)}
                for i in range(len(hist))]

    return {
        'fund':              fund_name,
        'percentile_bands':  {str(k): v for k, v in bands.items()},
        'summary':           _summary_safe(summary),
        'final_distribution': fin_dist,
        'sample_paths':      sample_paths,
        'days':              days,
        'iterations':        iterations,
    }
