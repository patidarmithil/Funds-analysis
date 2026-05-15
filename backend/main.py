import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routers.mf_router        import router as mf_router
from routers.analytics_router import router as analytics_router
from services.data_service    import load_default_funds, seed_supabase
from services.model_cache     import load_universal_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load data.xlsx + universal model + seed Supabase
    load_default_funds()
    load_universal_model()
    seed_supabase()          # safe no-op if Supabase not configured
    yield
    # Shutdown: nothing needed


app = FastAPI(
    title="FundScope Backend",
    description="FastAPI backend — ML analytics + mfapi.in proxy for FundScope React app",
    version="3.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Add your Vercel domain below once known.
ALLOWED_ORIGINS = [
    "http://localhost:5173",          # Vite dev
    "http://127.0.0.1:5173",
    "http://localhost:8501",          # Streamlit legacy (keep during transition)
    "http://127.0.0.1:8501",
    "https://fundscopefront.streamlit.app",
    "https://fundscope-xi.vercel.app"
]

frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    ALLOWED_ORIGINS.append(frontend_url)

# Allow all origins in development if env var set
if os.getenv("CORS_ALLOW_ALL", "").lower() == "true":
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(mf_router)
app.include_router(analytics_router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
