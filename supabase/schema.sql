-- FundScope Supabase Schema
-- Run this in Supabase SQL Editor (one-time setup)

-- 1. Fund metadata
CREATE TABLE IF NOT EXISTS fund_metadata (
  id          SERIAL PRIMARY KEY,
  fund_name   TEXT UNIQUE NOT NULL,
  scheme_code TEXT,
  category    TEXT,
  fund_house  TEXT,
  benchmark   TEXT,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Default NAV data (seeded from data.xlsx on backend startup)
CREATE TABLE IF NOT EXISTS default_fund_nav (
  id        SERIAL PRIMARY KEY,
  fund_name TEXT        NOT NULL,
  date      DATE        NOT NULL,
  nav       NUMERIC(12,4) NOT NULL,
  UNIQUE(fund_name, date)
);
CREATE INDEX IF NOT EXISTS idx_dfn_fund ON default_fund_nav(fund_name);
CREATE INDEX IF NOT EXISTS idx_dfn_date ON default_fund_nav(date);

-- 3. Analytics results cache
CREATE TABLE IF NOT EXISTS analytics_cache (
  id            SERIAL PRIMARY KEY,
  fund_name     TEXT        NOT NULL,
  analysis_type TEXT        NOT NULL,
  params_hash   TEXT        NOT NULL,
  result_json   JSONB       NOT NULL,
  computed_at   TIMESTAMPTZ DEFAULT NOW(),
  expires_at    TIMESTAMPTZ,
  UNIQUE(fund_name, analysis_type, params_hash)
);

-- 4. Live fund sessions (temporary per-user session data)
CREATE TABLE IF NOT EXISTS live_fund_sessions (
  id         SERIAL PRIMARY KEY,
  session_id TEXT        NOT NULL,
  fund_name  TEXT        NOT NULL,
  nav_data   JSONB       NOT NULL,
  fetched_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_lfs_session ON live_fund_sessions(session_id);

-- Seed fund_metadata (run after creating table)
INSERT INTO fund_metadata (fund_name) VALUES
  ('Flexi Cap'),
  ('India PSU'),
  ('Infrastructure'),
  ('Midcap'),
  ('Focused India'),
  ('Large and midcap fund'),
  ('Contra'),
  ('Multicap'),
  ('Financial Services'),
  ('ESG Integration Strategy'),
  ('ELSS Tax Saver'),
  ('Invesco Pan European'),
  ('Global Consumer Trends'),
  ('EQQQ NASDAQ-100 ETF')
ON CONFLICT (fund_name) DO NOTHING;
