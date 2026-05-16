# FundScope — Mutual Fund Analytics & Prediction

FundScope is a professional, modern web application built with **React** and **FastAPI** for comprehensive analysis, prediction, risk assessment, and simulation of mutual fund Net Asset Value (NAV). 

Designed with a sleek, responsive interface featuring **glassmorphism** aesthetics, it empowers investors to make data-driven decisions using advanced machine learning models and probabilistic simulations.

## 🌟 Key Features

* **📊 Portfolio Overview**: Live snapshots across all tracked mutual funds with a modern UI.
* **🤖 Advanced Predictions & Forecasting**: Forecast future NAVs using lightweight models (Prophet, ARIMA, XGBoost) with auto-optimized ensembles running on the FastAPI backend.
* **⚠️ Risk Assessment**: Calculates Value at Risk (VaR), Conditional Value at Risk (CVaR), and historical drawdowns.
* **🔁 Investment Backtesting**: Compares Buy and Hold strategies vs. Systematic Investment Plans (SIP).
* **🌀 Monte Carlo Simulation**: Runs thousands of random-walk simulations to project a range of future outcomes.

---

## 🚀 Cloud Architecture & Technology Stack

FundScope is built on a decoupled, scalable cloud architecture:

* **Frontend (UI & Analytics)**: **React + Vite** (Deployed on **Vercel**), custom CSS with premium Glassmorphism design.
* **Backend (API & Proxy)**: **FastAPI** (Python) — Deployed to an **Azure Cloud Server**.
* **Database (Persistence)**: **Supabase** (PostgreSQL) — Accessed securely via the backend.
* **Machine Learning Engine**: `xgboost`, `prophet`, `scikit-learn`, `statsmodels`.

### 🔄 Data Flow
1. **Frontend (React)** requests live mutual fund data or saved user preferences from the **Azure Backend**.
2. **Backend (FastAPI)** fetches data from **Supabase** (for DB records) or `mfapi.in` (for live market data).
3. **Backend** processes and normalizes the payload, sending a unified REST response back to the **Frontend**.
4. **Frontend** renders dynamic charts using **Recharts** and displays professional data views.

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

### 3. Frontend Setup (React/Vite)
Open a **new** terminal and run:
```bash
cd frontend
npm install
npm run dev
```
The application will be available at `http://localhost:5173`.

---

## 📂 Project Structure

```text
📁 FundScope
│
├── 📁 frontend/            # React/Vite UI (Deployed to Vercel)
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Full page views
│   │   ├── context/        # React Context API for state management
│   │   ├── index.css       # Premium Glassmorphism CSS styles
│   │   └── App.jsx         # Main Application Entry Point
│   ├── package.json
│   └── vite.config.js
│
├── 📁 backend/             # FastAPI Service (Deployed to Azure Cloud)
│   ├── main.py             # Server Initialization & CORS
│   ├── routers/            # API Endpoints
│   ├── services/           # Business logic & ML calculations
│   ├── schemas/            # Pydantic validation models
│   └── requirements.txt
│
├── 📁 Plans/               # Architecture & Deployment strategy files
└── README.md
```
