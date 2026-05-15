import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE,
  timeout: 120_000,   // 2 min — ML endpoints can be slow
  headers: { 'Content-Type': 'application/json' },
})

// ─── MF endpoints ─────────────────────────────────────────────────────────────

export const mfAPI = {
  search:     (q)     => api.get('/mf/search', { params: { q } }),
  schemes:    ()      => api.get('/mf/schemes'),
  fetchBatch: (body)  => api.post('/mf/fetch-batch', body),
  health:     ()      => api.get('/mf/health'),
}

// ─── Analytics endpoints ──────────────────────────────────────────────────────

export const analyticsAPI = {
  health:         ()      => api.get('/analytics/health'),
  defaultData:    ()      => api.get('/analytics/data/default'),
  uploadXlsx:     (form)  => api.post('/analytics/data/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }),
  fundSummary:    (body)  => api.post('/analytics/fund-summary', body),
  overview:       (body)  => api.post('/analytics/overview', body),
  analysis:       (body)  => api.post('/analytics/analysis', body),
  predict:        (body)  => api.post('/analytics/predict', body),
  predictEnsemble:(body)  => api.post('/analytics/predict-ensemble', body),
  risk:           (body)  => api.post('/analytics/risk', body),
  backtest:       (body)  => api.post('/analytics/backtest', body),
  simulate:       (body)  => api.post('/analytics/simulate', body),
}

export default api
