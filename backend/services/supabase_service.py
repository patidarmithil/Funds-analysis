"""
supabase_service.py — Supabase read/write/cache operations for FundScope backend.
"""
import os
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        logger.warning("SUPABASE_URL or SUPABASE_SERVICE_KEY not set. Supabase disabled.")
        return None
    try:
        from supabase import create_client
        _client = create_client(url, key)
        logger.info("Supabase client initialised.")
    except Exception as e:
        logger.error(f"Supabase init failed: {e}")
        _client = None
    return _client


def _params_hash(params: dict) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()


# ─── Analytics Cache ─────────────────────────────────────────────────────────

def get_cached(fund_name: str, analysis_type: str, params: dict,
               max_age_hours: int = 24) -> dict | None:
    """Return cached result if fresh, else None."""
    client = _get_client()
    if client is None:
        return None
    try:
        ph = _params_hash(params)
        resp = (
            client.table("analytics_cache")
            .select("result_json, computed_at")
            .eq("fund_name", fund_name)
            .eq("analysis_type", analysis_type)
            .eq("params_hash", ph)
            .maybe_single()
            .execute()
        )
        if resp.data:
            computed_at = datetime.fromisoformat(resp.data["computed_at"].replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - computed_at
            if age < timedelta(hours=max_age_hours):
                return resp.data["result_json"]
    except Exception as e:
        logger.warning(f"Cache read failed: {e}")
    return None


def set_cached(fund_name: str, analysis_type: str, params: dict, result: dict):
    """Upsert analytics result into cache."""
    client = _get_client()
    if client is None:
        return
    try:
        ph = _params_hash(params)
        client.table("analytics_cache").upsert({
            "fund_name":     fund_name,
            "analysis_type": analysis_type,
            "params_hash":   ph,
            "result_json":   result,
            "computed_at":   datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


# ─── Default Fund NAV ─────────────────────────────────────────────────────────

def get_default_fund_nav(fund_name: str | None = None) -> list[dict]:
    """Fetch all default fund NAV rows (or for a specific fund)."""
    client = _get_client()
    if client is None:
        return []
    try:
        q = client.table("default_fund_nav").select("fund_name, date, nav")
        if fund_name:
            q = q.eq("fund_name", fund_name)
        resp = q.order("fund_name").order("date").execute()
        return resp.data or []
    except Exception as e:
        logger.warning(f"default_fund_nav read failed: {e}")
        return []


def seed_default_fund_nav(records: list[dict]):
    """
    Bulk upsert NAV records into Supabase.
    records: [ { fund_name, date, nav } ]
    """
    client = _get_client()
    if client is None:
        return False
    try:
        # Upsert in chunks of 500
        for i in range(0, len(records), 500):
            client.table("default_fund_nav").upsert(records[i:i+500]).execute()
        logger.info(f"Seeded {len(records)} NAV rows into Supabase.")
        return True
    except Exception as e:
        logger.error(f"Seed failed: {e}")
        return False


# ─── Live Fund Sessions ───────────────────────────────────────────────────────

def save_live_session(session_id: str, fund_name: str, nav_data: list[dict]):
    client = _get_client()
    if client is None:
        return
    try:
        client.table("live_fund_sessions").upsert({
            "session_id": session_id,
            "fund_name":  fund_name,
            "nav_data":   nav_data,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.warning(f"Session save failed: {e}")


def get_live_session(session_id: str) -> list[dict]:
    client = _get_client()
    if client is None:
        return []
    try:
        resp = (
            client.table("live_fund_sessions")
            .select("fund_name, nav_data")
            .eq("session_id", session_id)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        logger.warning(f"Session read failed: {e}")
        return []
