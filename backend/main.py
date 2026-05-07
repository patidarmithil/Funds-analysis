import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.mf_router import router as mf_router

app = FastAPI(
    title="FundScope Backend",
    description="FastAPI proxy for mfapi.in — powers the FundScope Streamlit dashboard",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "https://fundscopefront.streamlit.app"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(mf_router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
