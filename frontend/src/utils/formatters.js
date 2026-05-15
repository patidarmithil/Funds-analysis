// formatters.js — number and date formatting utilities

export const fmt = {
  nav:     (v) => v != null ? `₹${Number(v).toFixed(2)}` : '—',
  pct:     (v) => v != null ? `${Number(v).toFixed(1)}%` : '—',
  pct2:    (v) => v != null ? `${Number(v).toFixed(2)}%` : '—',
  ratio:   (v) => v != null ? Number(v).toFixed(2) : '—',
  int:     (v) => v != null ? Math.round(v).toLocaleString('en-IN') : '—',
  money:   (v) => v != null ? `₹${Math.round(v).toLocaleString('en-IN')}` : '—',
  date:    (v) => v ? new Date(v).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }) : '—',
  short:   (s, n=12) => s ? (s.length > n ? s.slice(0, n) + '…' : s) : '—',
}

export const colorForValue = (v, inverse = false) => {
  if (v == null) return ''
  const positive = inverse ? v < 0 : v > 0
  return positive ? '#10b981' : v === 0 ? '' : '#ef4444'
}

// Convert raw records [{ds, y}] to Recharts-friendly format
export const navToChartData = (records) =>
  records.map(r => ({ ds: r.ds, y: Number(r.y) }))

// Normalise NAV series to base 100
export const normalise = (records) => {
  if (!records || records.length === 0) return []
  const base = Number(records[0].y)
  return records.map(r => ({ ds: r.ds, y: (Number(r.y) / base) * 100 }))
}
