// Chart colors — exact match to backend config.py CHART_COLORS
export const CHART_COLORS = [
  '#00d4ff', '#7c3aed', '#10b981', '#f59e0b',
  '#ef4444', '#f97316', '#8b5cf6', '#ec4899',
  '#06b6d4', '#84cc16', '#14b8a6', '#6366f1',
  '#fb923c', '#a855f7',
]

export const THEMES = {
  dark: {
    bg:       '#0a0e1a',
    surface:  '#0f1629',
    surface2: '#162040',
    border:   '#1e2d4d',
    text:     '#e2e8f0',
    muted:    '#94a3b8',
    primary:  '#00d4ff',
    success:  '#10b981',
    warning:  '#f59e0b',
    danger:   '#ef4444',
    gridColor: '#1e2d4d',
    chartBg:  '#0f1629',
    paperBg:  '#0a0e1a',
  },
  light: {
    bg:       '#f0f4f8',
    surface:  '#ffffff',
    surface2: '#e8eef6',
    border:   '#cbd5e1',
    text:     '#0f172a',
    muted:    '#475569',
    primary:  '#0284c7',
    success:  '#059669',
    warning:  '#d97706',
    danger:   '#dc2626',
    gridColor: '#e2e8f0',
    chartBg:  '#ffffff',
    paperBg:  '#f8fafc',
  },
}

// Percentile band colors for Monte Carlo (matching Streamlit)
export const PCT_COLORS = {
  95: '#10b981',
  75: '#00d4ff',
  50: '#7c3aed',
  25: '#f59e0b',
  5:  '#ef4444',
}

export const PCT_LABELS = {
  95: '95th Pct',
  75: '75th Pct',
  50: 'Median',
  25: '25th Pct',
  5:  '5th Pct',
}

export const PREDICTION_MODELS = [
  'Prophet', 'ARIMA', 'Holt-Winters',
  'Linear Regression', 'Ridge Regression',
  'Random Forest', 'SVR', 'XGBoost',
]

export const BENCHMARKS = {
  'Nifty 50':         '^NSEI',
  'Sensex (BSE 30)':  '^BSESN',
  'BSE 500':          'BSE-500.BO',
  'Nifty Midcap 150': '^NSMIDCP',
  'S&P 500 (US)':     '^GSPC',
  'NASDAQ-100':       '^NDX',
}
