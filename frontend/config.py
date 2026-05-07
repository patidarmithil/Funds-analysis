"""
FundScope — Central Configuration
"""

import os

# ─── Fund Names ─────────────────────────────────────────────────────────────
FUND_NAMES = [
    'Flexi Cap', 'India PSU', 'Infrastructure', 'Midcap',
    'Focused India', 'Large and midcap fund', 'Contra',
    'Multicap', 'Financial Services', 'ESG Integration Strategy',
    'ELSS Tax Saver', 'Invesco Pan European',
    'Global Consumer Trends', 'EQQQ NASDAQ-100 ETF',
]

# ─── Data ───────────────────────────────────────────────────────────────────
DEFAULT_FILE = os.path.join(os.path.dirname(__file__), 'data.xlsx')

# ─── Backend URL (FastAPI service) ──────────────────────────────────────────
# Override with env var FUNDSCOPE_BACKEND_URL in production
BACKEND_URL = os.getenv('FUNDSCOPE_BACKEND_URL', 'https://fundscopebackend-gbeybdd2gcd3egez.southeastasia-01.azurewebsites.net')

# ─── Benchmarks ─────────────────────────────────────────────────────────────
BENCHMARKS = {
    'Nifty 50':           '^NSEI',
    'Sensex (BSE 30)':    '^BSESN',
    'BSE 500':            'BSE-500.BO',
    'Nifty Midcap 150':   '^NSMIDCP',
    'S&P 500 (US)':       '^GSPC',
    'NASDAQ-100':         '^NDX',
}

# ─── Financial Constants ─────────────────────────────────────────────────────
RISK_FREE_RATE = 0.065       # 6.5% annualised — Indian 10Y G-Sec
TRADING_DAYS   = 252

# ─── Prediction Models ───────────────────────────────────────────────────────
PREDICTION_MODELS = [
    'Prophet',
    'ARIMA',
    'Holt-Winters',
    'Linear Regression',
    'Ridge Regression',
    'Random Forest',
    'SVR',
    'XGBoost',
]

# ─── Colour Palette ──────────────────────────────────────────────────────────
COLORS = {
    'primary':   '#00d4ff',
    'secondary': '#7c3aed',
    'success':   '#10b981',
    'warning':   '#f59e0b',
    'danger':    '#ef4444',
    'bg':        '#0a0e1a',
    'surface':   '#0f1629',
    'surface2':  '#162040',
    'border':    '#1e2d4d',
    'text':      '#e2e8f0',
    'muted':     '#94a3b8',
}

# Chart series colours (for multi-fund plots)
CHART_COLORS = [
    '#00d4ff', '#7c3aed', '#10b981', '#f59e0b',
    '#ef4444', '#f97316', '#8b5cf6', '#ec4899',
    '#06b6d4', '#84cc16', '#14b8a6', '#6366f1',
    '#fb923c', '#a855f7',
]

# ─── Plotly Layouts (Dark & Light) ───────────────────────────────────────────
_COMMON_LAYOUT = dict(
    font          = dict(family='Inter, sans-serif', size=13),
    margin        = dict(l=40, r=40, t=50, b=40),
    xaxis         = dict(showgrid=True, zeroline=False),
    yaxis         = dict(showgrid=True, zeroline=False),
)

PLOTLY_DARK = {
    **_COMMON_LAYOUT,
    'plot_bgcolor':  '#0f1629',
    'paper_bgcolor': '#0a0e1a',
    'font':          {**_COMMON_LAYOUT['font'], 'color': '#e2e8f0'},
    'xaxis':         {**_COMMON_LAYOUT['xaxis'], 'gridcolor': '#1e2d4d'},
    'yaxis':         {**_COMMON_LAYOUT['yaxis'], 'gridcolor': '#1e2d4d'},
    'legend':        dict(bgcolor='#162040', bordercolor='#1e2d4d', borderwidth=1),
    'hoverlabel':    dict(bgcolor='#162040', bordercolor='#1e2d4d', font_size=13),
}

PLOTLY_LIGHT = {
    **_COMMON_LAYOUT,
    'plot_bgcolor':  '#ffffff',
    'paper_bgcolor': '#f8fafc',
    'font':          {**_COMMON_LAYOUT['font'], 'color': '#1e293b'},
    'xaxis':         {**_COMMON_LAYOUT['xaxis'], 'gridcolor': '#e2e8f0'},
    'yaxis':         {**_COMMON_LAYOUT['yaxis'], 'gridcolor': '#e2e8f0'},
    'legend':        dict(bgcolor='#ffffff', bordercolor='#e2e8f0', borderwidth=1),
    'hoverlabel':    dict(bgcolor='#ffffff', bordercolor='#e2e8f0', font_size=13),
}

# Default for legacy code or quick access
PLOTLY_LAYOUT = PLOTLY_DARK.copy()

# ─── Simulation Defaults ──────────────────────────────────────────────────────
MONTE_CARLO = dict(days=252, iterations=1000)

# ─── Backtesting Defaults ─────────────────────────────────────────────────────
BACKTEST = dict(lumpsum=10000, monthly_sip=1000)

# ─── Stress-Test Scenarios ───────────────────────────────────────────────────
STRESS_SCENARIOS = {
    'Bull Market (+30%)':  +0.30,
    'Mild Bull (+15%)':    +0.15,
    'Flat (0%)':            0.00,
    'Mild Bear (−15%)':    -0.15,
    'Bear Market (−30%)':  -0.30,
    'Market Crash (−50%)': -0.50,
}

# ─── Helpers ─────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Convert hex color string to rgba string."""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    if lv == 3:
        r, g, b = [int(hex_color[i:i+1]*2, 16) for i in range(0, 3)]
    else:
        r, g, b = [int(hex_color[i:i+lv//3], 16) for i in range(0, lv, lv//3)]
    return f"rgba({r}, {g}, {b}, {alpha})"
