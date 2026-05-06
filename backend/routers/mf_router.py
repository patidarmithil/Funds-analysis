import os
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from schemas.mf_schema import SchemeItem, FundResult, BatchRequest
from services import mf_service

router = APIRouter(prefix="/mf")

MAX_BATCH = int(os.getenv("MAX_FUNDS_PER_BATCH", "10"))

@router.get("/health", tags=["Meta"])
def health():
    """Quick health-check to confirm the backend is running."""
    return {
        "status":    "ok",
        "service":   "fundscope-backend (FastAPI)",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

@router.get("/schemes", response_model=list[SchemeItem], tags=["Mutual Funds"])
async def get_all_schemes():
    """Returns the full list of all Indian MF schemes (~17k) from mfapi.in."""
    return await mf_service.get_all_schemes_service()

@router.get("/search", response_model=list[SchemeItem], tags=["Mutual Funds"])
async def search_schemes(q: str = Query("", min_length=0)):
    """Search Indian MF schemes by name via mfapi.in search proxy."""
    return await mf_service.search_schemes_service(q)

@router.post("/fetch-batch", response_model=list[FundResult], tags=["Mutual Funds"])
async def fetch_batch(body: BatchRequest):
    """Fetch full NAV history for N selected funds in parallel."""
    if not body.funds:
        raise HTTPException(status_code=400, detail="No funds provided in request body.")
    if len(body.funds) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_BATCH} funds per batch. You requested {len(body.funds)}.",
        )
    return await mf_service.fetch_batch_service(body)
