import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { supabase } from '../lib/supabaseClient'
import { analyticsAPI } from '../lib/api'

const AppContext = createContext(null)

export function AppProvider({ children }) {
  const [theme, setThemeState]       = useState(() => localStorage.getItem('fundscope-theme') || 'dark')
  const [dataMode, setDataMode]      = useState('default')   // 'default' | 'live'
  const [defaultFunds, setDefaultFunds] = useState({})       // { fundName: [{ds,y}] }
  const [liveFunds, setLiveFunds]    = useState({})
  const [selectedFunds, setSelectedFunds] = useState([])
  const [backendOk, setBackendOk]    = useState(null)        // null=unknown, true, false
  const [loading, setLoading]        = useState(true)

  // ── Theme persistence ──────────────────────────────────────────────────────
  const setTheme = (t) => {
    setThemeState(t)
    localStorage.setItem('fundscope-theme', t)
  }

  // ── Backend health check ───────────────────────────────────────────────────
  useEffect(() => {
    analyticsAPI.health()
      .then(() => setBackendOk(true))
      .catch(() => setBackendOk(false))
  }, [])

  // ── Load default fund NAV data ─────────────────────────────────────────────
  useEffect(() => {
    loadDefaultFunds()
  }, [])

  const loadDefaultFunds = useCallback(async () => {
    setLoading(true)
    try {
      // Try Supabase first (faster, no Azure compute)
      if (supabase) {
        const { data, error } = await supabase
          .from('default_fund_nav')
          .select('fund_name, date, nav')
          .order('fund_name')
          .order('date')
        if (!error && data && data.length > 0) {
          const funds = {}
          for (const row of data) {
            if (!funds[row.fund_name]) funds[row.fund_name] = []
            funds[row.fund_name].push({ ds: row.date, y: Number(row.nav) })
          }
          setDefaultFunds(funds)
          setLoading(false)
          return
        }
      }
    } catch (_) {}

    // Fallback: Azure backend
    try {
      const { data } = await analyticsAPI.defaultData()
      setDefaultFunds(data)
    } catch (e) {
      console.error('Failed to load default fund data:', e)
    }
    setLoading(false)
  }, [])

  // ── Active funds helper ────────────────────────────────────────────────────
  const activeFunds = dataMode === 'live' && Object.keys(liveFunds).length > 0
    ? liveFunds
    : defaultFunds

  const fundNames = Object.keys(activeFunds)

  return (
    <AppContext.Provider value={{
      theme, setTheme,
      dataMode, setDataMode,
      defaultFunds, setDefaultFunds,
      liveFunds, setLiveFunds,
      selectedFunds, setSelectedFunds,
      backendOk,
      loading,
      loadDefaultFunds,
      activeFunds,
      fundNames,
    }}>
      {children}
    </AppContext.Provider>
  )
}

export const useApp = () => {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used inside AppProvider')
  return ctx
}
