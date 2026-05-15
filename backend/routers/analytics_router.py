"""
analytics_router.py — All ML analytics endpoints for FundScope.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from services import analytics_service, supabase_service
from services.data_service import default_funds_as_json, parse_uploaded_xlsx, get_default_funds
from services.model_cache import load_universal_model

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class FundDataPayload(BaseModel):
    fund_name: str
    data: list   # [{ds, y}]

class FundsOverviewPayload(BaseModel):
    funds_data: dict   # { fund_name: [{ds, y}] }

class AnalysisPayload(BaseModel):
    fund_name: str
    data: list
    benchmark_ticker: Optional[str] = None
    date_range: Optional[list] = None

class PredictPayload(BaseModel):
    fund_name: str
    data: list
    model_name: str
    periods: int = 90

class EnsemblePayload(BaseModel):
    fund_name: str
    data: list
    model_names: list
    periods: int = 90

class RiskPayload(BaseModel):
    fund_name: str
    data: list
    investment_amount: float = 100_000
    confidence: float = 0.95

class BacktestPayload(BaseModel):
    fund_name: str
    data: list
    lumpsum: float = 10_000
    monthly_sip: float = 1_000
    start_date: Optional[str] = None

class SimulatePayload(BaseModel):
    fund_name: str
    data: list
    iterations: int = 1000
    days: int = 252
    investment: float = 10_000


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health")
def analytics_health():
    model = load_universal_model()
    funds = get_default_funds()
    return {
        "status":             "ok",
        "model_loaded":       model is not None,
        "model_type":         type(model).__name__,
        "default_funds_count": len(funds),
    }


# ─── Default data ─────────────────────────────────────────────────────────────

@router.get("/data/default")
def get_default_data():
    """Return all default fund NAV data as JSON (fallback if Supabase unavailable)."""
    data = default_funds_as_json()
    if not data:
        raise HTTPException(status_code=503, detail="Default fund data not loaded.")
    return data


@router.post("/data/upload")
async def upload_xlsx(file: UploadFile = File(...)):
    """Parse uploaded xlsx file → { fund_name: [{ds, y}] }"""
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Only .xlsx files accepted.")
    content = await file.read()
    result = parse_uploaded_xlsx(content)
    if not result:
        raise HTTPException(status_code=422, detail="No valid fund sheets found in xlsx.")
    return result


# ─── Analytics endpoints ──────────────────────────────────────────────────────

@router.post("/fund-summary")
def fund_summary(payload: FundDataPayload):
    result = analytics_service.compute_fund_summary(payload.fund_name, payload.data)
    if 'error' in result:
        raise HTTPException(status_code=422, detail=result['error'])
    return result


@router.post("/overview")
def overview(payload: FundsOverviewPayload):
    # Check cache
    cache_params = {'funds': sorted(payload.funds_data.keys())}
    cached = supabase_service.get_cached("_all_", "overview", cache_params, max_age_hours=24)
    if cached:
        return cached
    result = analytics_service.compute_overview(payload.funds_data)
    supabase_service.set_cached("_all_", "overview", cache_params, result)
    return result


@router.post("/analysis")
def analysis(payload: AnalysisPayload):
    cache_params = {
        'fund': payload.fund_name,
        'benchmark': payload.benchmark_ticker,
        'date_range': payload.date_range,
    }
    cached = supabase_service.get_cached(payload.fund_name, "analysis", cache_params)
    if cached:
        return cached
    result = analytics_service.compute_analysis(
        payload.fund_name, payload.data,
        payload.benchmark_ticker, payload.date_range,
    )
    if 'error' in result:
        raise HTTPException(status_code=422, detail=result['error'])
    supabase_service.set_cached(payload.fund_name, "analysis", cache_params, result)
    return result


@router.post("/predict")
def predict(payload: PredictPayload):
    result = analytics_service.compute_predict(
        payload.fund_name, payload.data, payload.model_name, payload.periods
    )
    if 'error' in result:
        raise HTTPException(status_code=422, detail=result['error'])
    return result


@router.post("/predict-ensemble")
def predict_ensemble(payload: EnsemblePayload):
    result = analytics_service.compute_predict_ensemble(
        payload.fund_name, payload.data, payload.model_names, payload.periods
    )
    if 'error' in result:
        raise HTTPException(status_code=422, detail=result['error'])
    return result


@router.post("/risk")
def risk(payload: RiskPayload):
    cache_params = {
        'fund': payload.fund_name,
        'investment': payload.investment_amount,
        'confidence': payload.confidence,
    }
    cached = supabase_service.get_cached(payload.fund_name, "risk", cache_params)
    if cached:
        return cached
    result = analytics_service.compute_risk(
        payload.fund_name, payload.data,
        payload.investment_amount, payload.confidence,
    )
    if 'error' in result:
        raise HTTPException(status_code=422, detail=result['error'])
    supabase_service.set_cached(payload.fund_name, "risk", cache_params, result)
    return result


@router.post("/backtest")
def backtest(payload: BacktestPayload):
    cache_params = {
        'fund': payload.fund_name,
        'lumpsum': payload.lumpsum,
        'sip': payload.monthly_sip,
        'start': payload.start_date,
    }
    cached = supabase_service.get_cached(payload.fund_name, "backtest", cache_params, max_age_hours=48)
    if cached:
        return cached
    result = analytics_service.compute_backtest(
        payload.fund_name, payload.data,
        payload.lumpsum, payload.monthly_sip, payload.start_date,
    )
    if 'error' in result:
        raise HTTPException(status_code=422, detail=result['error'])
    supabase_service.set_cached(payload.fund_name, "backtest", cache_params, result)
    return result


@router.post("/simulate")
def simulate(payload: SimulatePayload):
    result = analytics_service.compute_simulate(
        payload.fund_name, payload.data,
        payload.iterations, payload.days, payload.investment,
    )
    if 'error' in result:
        raise HTTPException(status_code=422, detail=result['error'])
    return result
