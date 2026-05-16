import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import { mfAPI } from '../lib/api'

export default function NewAnalysis() {
  const { setLiveFunds, setDataMode } = useApp()
  const navigate = useNavigate()

  const [query,    setQuery]    = useState('')
  const [schemes,  setSchemes]  = useState([])
  const [selected, setSelected] = useState([])
  const [startDate, setStartDate] = useState('2020-01-01')
  const [endDate,  setEndDate]  = useState(new Date().toISOString().slice(0,10))
  const [searching, setSearching] = useState(false)
  const [fetching,  setFetching]  = useState(false)
  const [error,    setError]    = useState(null)
  const [fetchDone, setFetchDone] = useState(false)

  const search = async () => {
    if (query.trim().length < 2) return
    setSearching(true); setError(null); setSchemes([])
    try {
      const { data } = await mfAPI.search(query.trim())
      setSchemes(data.slice(0, 40))
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    }
    setSearching(false)
  }

  const toggleScheme = (s) =>
    setSelected(prev =>
      prev.find(x => x.schemeCode === s.schemeCode)
        ? prev.filter(x => x.schemeCode !== s.schemeCode)
        : [...prev, s]
    )

  const fetchAndLoad = async () => {
    if (!selected.length) return
    setFetching(true); setError(null)
    try {
      const { data } = await mfAPI.fetchBatch({
        funds: selected.map(s => ({ schemeCode: s.schemeCode, schemeName: s.schemeName })),
        startDate, endDate,
      })
      const funds = {}
      for (const entry of data) {
        if (!entry.error && entry.data?.length > 0) {
          funds[entry.fund] = entry.data
        }
      }
      if (!Object.keys(funds).length) throw new Error('No valid NAV data returned.')
      setLiveFunds(funds)
      setDataMode('live')
      setFetchDone(true)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    }
    setFetching(false)
  }

  return (
    <div className="page" id="page-new-analysis">
      <h1 className="page-title">🔍 New Analysis</h1>
      <p className="page-subtitle">Search any Indian mutual fund by name and fetch live NAV history</p>

      {/* ── Search ── */}
      <div className="controls" style={{ marginBottom: '1rem' }}>
        <div className="control-group" style={{ flex: 1 }}>
          <label>Fund Name</label>
          <input id="input-fund-search" type="text" value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && search()}
            placeholder="e.g. Mirae Asset Midcap"
            style={{ minWidth: 300 }} />
        </div>
        <button className="btn btn-secondary" id="btn-search-fund" onClick={search} disabled={searching}>
          {searching ? '⏳ Searching…' : '🔍 Search'}
        </button>
      </div>

      {/* ── Results ── */}
      {schemes.length > 0 && (
        <>
          <p className="section-header">Results ({schemes.length}) — click to select</p>
          <div className="checkbox-grid" style={{ maxHeight: 320, overflowY: 'auto', padding: '1rem', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: 16, marginBottom: '1rem', background: 'var(--surface)' }}>
            {schemes.map(s => {
              const isSel = selected.find(x => x.schemeCode === s.schemeCode)
              return (
                <div key={s.schemeCode}
                  onClick={() => toggleScheme(s)}
                  className={`checkbox-item${isSel ? ' checked' : ''}`}
                  style={{ textAlign: 'left', justifyContent: 'flex-start', padding: '0.8rem 1.2rem', minHeight: '80px', alignItems: 'flex-start' }}
                >
                  {s.schemeName}
                </div>
              )
            })}
          </div>
        </>
      )}

      {/* ── Selected + date range ── */}
      {selected.length > 0 && (
        <>
          <div className="alert alert-info">
            ✅ {selected.length} fund{selected.length > 1 ? 's' : ''} selected: {selected.map(s => s.schemeName.slice(0,25)).join(', ')}
          </div>
          <div className="controls">
            <div className="control-group">
              <label>Start Date</label>
              <input id="input-live-start" type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
            </div>
            <div className="control-group">
              <label>End Date</label>
              <input id="input-live-end" type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
            </div>
            <button className="btn btn-primary" id="btn-fetch-nav" onClick={fetchAndLoad} disabled={fetching}>
              {fetching ? '⏳ Fetching NAV…' : '📥 Load Live Data'}
            </button>
          </div>
        </>
      )}

      {error && <div className="alert alert-danger">⚠ {error}</div>}

      {fetchDone && (
        <div className="alert alert-success">
          ✅ Live data loaded! Now navigate to any analysis page.
          <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.75rem' }}>
            <button className="btn btn-primary" onClick={() => navigate('/overview')}>📊 Overview</button>
            <button className="btn btn-secondary" onClick={() => navigate('/predictions')}>🤖 Predictions</button>
            <button className="btn btn-secondary" onClick={() => navigate('/risk')}>⚠️ Risk</button>
          </div>
        </div>
      )}
    </div>
  )
}
