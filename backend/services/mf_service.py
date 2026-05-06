import asyncio
import os
import httpx
from typing import Optional
from schemas.mf_schema import SchemeItem, NavPoint, FundResult, BatchRequest

MFAPI_BASE = os.getenv("MFAPI_BASE_URL", "https://api.mfapi.in")
TIMEOUT = float(os.getenv("MFAPI_TIMEOUT_SEC", "20"))

def _ddmmyyyy_to_iso(date_str: str) -> Optional[str]:
    """Convert mfapi.in date format 'DD-MM-YYYY' → 'YYYY-MM-DD'."""
    if not date_str:
        return None
    parts = date_str.split("-")
    if len(parts) != 3:
        return None
    d, m, y = parts
    return f"{y}-{m.zfill(2)}-{d.zfill(2)}"

async def _fetch_one_fund(
    client: httpx.AsyncClient,
    code: int,
    name: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> FundResult:
    """Fetch + normalise NAV history for a single fund from mfapi.in."""
    params = {}
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    try:
        resp = await client.get(f"{MFAPI_BASE}/mf/{code}", params=params)
        resp.raise_for_status()
        raw = resp.json().get("data", [])

        nav_points: list[NavPoint] = []
        for item in raw:
            iso_date = _ddmmyyyy_to_iso(item.get("date", ""))
            try:
                nav_val = float(item.get("nav", "nan"))
            except (ValueError, TypeError):
                continue
            if iso_date and not (nav_val != nav_val):  # filter NaN
                nav_points.append(NavPoint(ds=iso_date, y=nav_val))

        # Sort ascending by date
        nav_points.sort(key=lambda p: p.ds)

        print(f"  ✓ {name} ({code}): {len(nav_points)} records")
        return FundResult(fund=name, code=code, data=nav_points, records=len(nav_points))

    except Exception as exc:
        print(f"  ✗ {name} ({code}): {exc}")
        return FundResult(fund=name, code=code, data=[], records=0, error=str(exc))

async def get_all_schemes_service() -> list[dict]:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{MFAPI_BASE}/mf")
        resp.raise_for_status()
        data = resp.json()
        print(f"Scheme list fetched: {len(data)} schemes")
        return data

async def search_schemes_service(q: str) -> list[dict]:
    q = q.strip()
    if len(q) < 2:
        return []
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{MFAPI_BASE}/mf/search", params={"q": q})
        resp.raise_for_status()
        return resp.json()

async def fetch_batch_service(body: BatchRequest) -> list[FundResult]:
    print(
        f"Batch fetch: {len(body.funds)} fund(s)"
        + (f", from {body.startDate}" if body.startDate else "")
        + (f" to {body.endDate}" if body.endDate else "")
    )

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = [
            _fetch_one_fund(client, f.schemeCode, f.schemeName, body.startDate, body.endDate)
            for f in body.funds
        ]
        results = await asyncio.gather(*tasks)

    ok = sum(1 for r in results if not r.error)
    failed = sum(1 for r in results if r.error)
    print(f"Batch complete: {ok} succeeded, {failed} failed")

    return results
