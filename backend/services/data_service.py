"""
data_service.py — Parse data.xlsx and seed into Supabase on startup.
"""
import os
import logging
import pandas as pd
from modules.data_loader import load_all_funds, df_to_records
from services import supabase_service

logger = logging.getLogger(__name__)

# In-memory cache of default fund data (loaded once at startup)
_DEFAULT_FUNDS: dict = {}


def load_default_funds() -> dict:
    """
    Load data.xlsx into memory. Called once at startup.
    Returns { fund_name: DataFrame }
    """
    global _DEFAULT_FUNDS
    if _DEFAULT_FUNDS:
        return _DEFAULT_FUNDS
    _DEFAULT_FUNDS = load_all_funds()
    logger.info(f"Loaded {len(_DEFAULT_FUNDS)} default funds from data.xlsx.")
    return _DEFAULT_FUNDS


def get_default_funds() -> dict:
    """Return in-memory default funds (call load_default_funds() first)."""
    return _DEFAULT_FUNDS


def seed_supabase():
    """
    Seed default_fund_nav table in Supabase from data.xlsx.
    Safe to call multiple times — uses upsert.
    """
    funds = load_default_funds()
    if not funds:
        logger.warning("No funds loaded — skipping Supabase seed.")
        return
    records = []
    for fund_name, df in funds.items():
        for _, row in df.iterrows():
            records.append({
                "fund_name": fund_name,
                "date":      str(row['ds'].date()),
                "nav":       round(float(row['y']), 4),
            })
    ok = supabase_service.seed_default_fund_nav(records)
    if ok:
        logger.info(f"Seeded {len(records)} rows to Supabase default_fund_nav.")
    else:
        logger.warning("Supabase seed skipped (client unavailable).")


def default_funds_as_json() -> dict:
    """Return { fund_name: [{ds, y}] } serialisable dict."""
    funds = get_default_funds()
    return {name: df_to_records(df) for name, df in funds.items()}


def parse_uploaded_xlsx(file_bytes: bytes) -> dict:
    """Parse user-uploaded xlsx bytes → { fund_name: [{ds, y}] }"""
    from modules.data_loader import load_all_funds, df_to_records
    funds = load_all_funds(file_bytes=file_bytes)
    return {name: df_to_records(df) for name, df in funds.items()}
