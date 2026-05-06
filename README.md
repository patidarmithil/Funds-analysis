# FundScope — Mutual Fund Analytics & Prediction

FundScope is a professional, modern web application built with **Streamlit** and **FastAPI** for comprehensive analysis, prediction, risk assessment, and simulation of mutual fund Net Asset Value (NAV). 

Designed with a sleek, responsive interface, it empowers investors to make data-driven decisions using advanced machine learning models and probabilistic simulations.

## 🌟 Key Features

* **📊 Portfolio Overview**: Live snapshots across all tracked mutual funds.
* **🤖 Advanced Predictions & Forecasting**: Forecast future NAVs using lightweight models (Prophet, ARIMA, XGBoost) with auto-optimized ensembles.
* **⚠️ Risk Assessment**: Calculates Value at Risk (VaR), Conditional Value at Risk (CVaR), and historical drawdowns.
* **🔁 Investment Backtesting**: Compares Buy and Hold strategies vs. Systematic Investment Plans (SIP).
* **🌀 Monte Carlo Simulation**: Runs thousands of random-walk simulations to project a range of future outcomes.

---

## 🚀 Cloud Architecture & Technology Stack

FundScope is built on a decoupled, scalable cloud architecture:

* **Frontend (UI & Analytics)**: [Streamlit](https://streamlit.io/) — Deployed to **Streamlit Community Cloud**.
* **Backend (API & Proxy)**: **FastAPI** (Python) — Deployed to an **Azure Cloud Server**.
* **Database (Persistence)**: **Supabase** (PostgreSQL) — Accessed securely via the backend.
* **Machine Learning Engine**: `xgboost`, `prophet`, `scikit-learn`, `statsmodels`.

### 🔄 Data Flow
1. **Frontend (Streamlit)** requests live mutual fund data or saved user preferences from the **Azure Backend**.
2. **Backend (FastAPI)** fetches data from **Supabase** (for DB records) or `mfapi.in` (for live market data).
3. **Backend** processes and normalizes the payload, sending a unified REST response back to the **Frontend**.
4. **Frontend** runs local Pandas/ML computations and visualizes the results.

---

## ⚙️ Local Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Backend Setup (FastAPI)
Open a terminal and run:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup (Streamlit)
Open a **new** terminal and run:
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
📁 FundScope
│
├── 📁 frontend/            # Streamlit UI (Deployed to Streamlit Cloud)
│   ├── app.py              # Main Application Entry Point
│   ├── config.py           # Theme & Layout Variables
│   ├── app_pages/          # UI Views (Overview, Risk, Backtesting, etc.)
│   └── modules/            # Local ML calculations & Frontend API fetching
│
├── 📁 backend/             # FastAPI Service (Deployed to Azure Cloud)
│   ├── main.py             # Server Initialization & CORS
│   ├── routers/            # API Endpoints
│   ├── services/           # Business logic & external API integration
│   ├── schemas/            # Pydantic validation models
│   └── dependencies/       # Dependency injection (e.g., Supabase clients)
│
├── 📁 docs/                # API Documentation (Swagger JSON)
├── 📁 Plans/               # Architecture & Deployment strategy files
├── .gitignore              # Secures sensitive keys & caches
└── README.md
```
