"""
FundScope — Backend Configuration (Plotly-free version)
"""
import os

# ─── Fund Names ──────────────────────────────────────────────────────────────
FUND_NAMES = [
    'Flexi Cap', 'India PSU', 'Infrastructure', 'Midcap',
    'Focused India', 'Large and midcap fund', 'Contra',
    'Multicap', 'Financial Services', 'ESG Integration Strategy',
    'ELSS Tax Saver', 'Invesco Pan European',
    'Global Consumer Trends', 'EQQQ NASDAQ-100 ETF',
]

# ─── Data ────────────────────────────────────────────────────────────────────
DEFAULT_FILE = os.path.join(os.path.dirname(__file__), 'data.xlsx')

# ─── Benchmarks ──────────────────────────────────────────────────────────────
BENCHMARKS = {
    'Nifty 50':         '^NSEI',
    'Sensex (BSE 30)':  '^BSESN',
    'BSE 500':          'BSE-500.BO',
    'Nifty Midcap 150': '^NSMIDCP',
    'S&P 500 (US)':     '^GSPC',
    'NASDAQ-100':       '^NDX',
}

# ─── Financial Constants ─────────────────────────────────────────────────────
RISK_FREE_RATE = 0.065   # 6.5% annualised — Indian 10Y G-Sec
TRADING_DAYS   = 252

# ─── Prediction Models ───────────────────────────────────────────────────────
PREDICTION_MODELS = [
    'Prophet', 'ARIMA', 'Holt-Winters',
    'Linear Regression', 'Ridge Regression',
    'Random Forest', 'SVR', 'XGBoost',
]

# ─── Monte Carlo Defaults ────────────────────────────────────────────────────
MONTE_CARLO = dict(days=252, iterations=1000)

# ─── Backtesting Defaults ────────────────────────────────────────────────────
BACKTEST = dict(lumpsum=10000, monthly_sip=1000)

# ─── Stress-Test Scenarios ───────────────────────────────────────────────────
STRESS_SCENARIOS = {
    'Bull Market (+30%)':  +0.30,
    'Mild Bull (+15%)':    +0.15,
    'Flat (0%)':            0.00,
    'Mild Bear (-15%)':    -0.15,
    'Bear Market (-30%)':  -0.30,
    'Market Crash (-50%)': -0.50,
}

# ─── Chart Colors (passed to frontend via API responses if needed) ───────────
CHART_COLORS = [
    '#00d4ff', '#7c3aed', '#10b981', '#f59e0b',
    '#ef4444', '#f97316', '#8b5cf6', '#ec4899',
    '#06b6d4', '#84cc16', '#14b8a6', '#6366f1',
    '#fb923c', '#a855f7',
]
