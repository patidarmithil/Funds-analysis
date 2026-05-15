# FundScope — How to Run Locally & Deploy

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Backend |
| Node.js | 18+ | React frontend |
| npm | 9+ | React package manager |

---

## 1. Environment Setup

### 1.1 Backend `.env` (create at `backend/.env`)

```env
# Supabase (get from Supabase dashboard → Project Settings → API)
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_SERVICE_KEY=<your-service-role-key>   # NOT the anon key

# Optional
PORT=8000
CORS_ALLOW_ALL=false   # set true only in dev if needed
```

> **Supabase not set up yet?** Backend still works — just skips cache/seed. Set it up later.

### 1.2 React `.env.local` (create at `frontend/.env.local`)

```env
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=https://<your-project>.supabase.co
VITE_SUPABASE_ANON_KEY=<your-anon-key>   # use anon key here (public)
```

Copy from the example file:
```powershell
Copy-Item frontend\.env.local.example frontend\.env.local
# then edit with your values
```

---

## 2. Supabase Setup (one-time)

1. Go to [supabase.com](https://supabase.com) → create new project
2. Open **SQL Editor** → paste content of `supabase/schema.sql` → **Run**
3. Get your keys from **Project Settings → API**:
   - `URL` → use as `SUPABASE_URL` and `VITE_SUPABASE_URL`
   - `anon public` key → use as `VITE_SUPABASE_ANON_KEY`
   - `service_role` key → use as `SUPABASE_SERVICE_KEY` (backend only, never expose to frontend)
4. Backend auto-seeds `default_fund_nav` from `data.xlsx` on first startup

---

## 3. Running Locally

Open **two terminals**:

### Terminal 1 — FastAPI Backend

```powershell
cd d:\Coding\Project\Finance\FundScope\backend

# First time only — install dependencies
pip install -r requirements.txt

# Start backend
uvicorn main:app --reload --port 8000
```

Backend available at:
- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/analytics/health`

### Terminal 2 — React Frontend

```powershell
cd d:\Coding\Project\Finance\FundScope\frontend

# First time only — install packages
npm install


# Start dev server
npm run dev
```

React app available at: `http://localhost:5173`

### Verify everything works

1. Open `http://localhost:8000/docs` — Swagger should load
2. Open `http://localhost:5173` — FundScope home page should load
3. Click "📊 Overview" → click "▶ Compute Overview" → should return fund metrics
4. Check `http://localhost:8000/analytics/health` — should show `model_loaded: true`

### Also run Streamlit (optional, during transition)

```powershell
# Terminal 3 (optional)
cd d:\Coding\Project\Finance\FundScope\frontend
streamlit run app.py
# Available at http://localhost:8501
```

---

## 4. Prophet Installation Note

Prophet has native C++ dependencies. If install fails:

```powershell
# Windows — install Microsoft C++ Build Tools first
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Then retry:
pip install prophet

# Alternative: use conda
conda install -c conda-forge prophet
```

On Azure: if deploy fails due to Prophet, switch to **Azure Container Apps** with a Dockerfile that uses `python:3.11-slim` + build tools.

---

## 5. Deploying to Production

### 5.1 FastAPI → Azure App Service

**Deploy via GitHub Actions (recommended):**

Azure Portal → App Service → Deployment Center → GitHub → connect repo → set root path to `backend/`

**Or deploy via ZIP:**
```powershell
cd backend
# Zip everything
Compress-Archive -Path . -DestinationPath ../backend_deploy.zip
# Upload via Azure Portal → App Service → Advanced Tools → Kudu
```

**Azure App Service — Application Settings (env vars to set):**

| Setting | Value |
|---------|-------|
| `SUPABASE_URL` | `https://<project>.supabase.co` |
| `SUPABASE_SERVICE_KEY` | `<service-role-key>` |
| `CORS_ALLOW_ALL` | `false` |
| `SCM_DO_BUILD_DURING_DEPLOYMENT` | `true` |
| `WEBSITE_RUN_FROM_PACKAGE` | `1` (if using zip deploy) |

**After first deploy — add your Vercel domain to CORS in `backend/main.py`:**
```python
# Uncomment this line in backend/main.py:
# "https://<your-app>.vercel.app",
```

### 5.2 React → Vercel

1. Push `frontend/` to GitHub (or the full repo)
2. Go to [vercel.com](https://vercel.com) → New Project → Import from GitHub
3. Set **Root Directory** to `frontend`
4. Vercel auto-detects Vite framework
5. Add **Environment Variables** in Vercel dashboard:

| Variable | Value |
|----------|-------|
| `VITE_API_URL` | `https://fundscopebackend-gbeybdd2gcd3egez.southeastasia-01.azurewebsites.net` |
| `VITE_SUPABASE_URL` | `https://<project>.supabase.co` |
| `VITE_SUPABASE_ANON_KEY` | `<your-anon-key>` |

6. Click Deploy → Vercel builds and deploys automatically on every push

---

## 6. Port Quick Reference

| Service | Local | Production |
|---------|-------|------------|
| FastAPI | `http://localhost:8000` | `https://fundscopebackend-*.azurewebsites.net` |
| React Vite | `http://localhost:5173` | `https://<app>.vercel.app` |
| Streamlit (legacy) | `http://localhost:8501` | `https://fundscopefront.streamlit.app` |
| Supabase | always cloud | always cloud |

---

## 7. Things to Keep in Mind

### Backend

- **data.xlsx must be in `backend/`** — the backend serves it on startup and seeds Supabase. Do not delete it.
- **`backend/models/`** — place `universal_fund_model.pkl` here once trained. Backend auto-loads on startup. Until then, a LinearRegression placeholder runs.
- **`SUPABASE_SERVICE_KEY` is secret** — never expose to frontend or commit to Git.
- **Prophet is slow to install** — ~5–10 min on first `pip install`. Be patient.
- **Azure cold start** — Azure App Service free/basic tier has cold starts. First request after idle may take 30–60s.
- **`modules/` import paths** — all modules use `from config import ...` which resolves from the `backend/` working directory. Always run uvicorn from `backend/` directory, not from root.

### React

- **`.env.local` is gitignored** — each developer creates their own. Never commit it.
- **All API calls use `VITE_API_URL`** — change this one env var to switch between local and prod.
- **Supabase anon key is safe to expose** — it's public by design. Row-level security controls access.
- **`npm run dev` vs `npm run build`** — use `dev` locally; `build` only for production validation.
- **Vite proxy** — local dev uses `/api` proxy to forward to FastAPI, avoiding CORS. In production, React calls Azure directly via `VITE_API_URL`.

### Common Issues

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'config'` | Run uvicorn from `backend/` directory, not root |
| React shows "Backend offline" | FastAPI not running on port 8000. Start it first. |
| Supabase seed not running | Check `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in `backend/.env` |
| CORS error in browser | Add your Vercel domain to `ALLOWED_ORIGINS` in `backend/main.py` |
| Prophet install fails on Windows | Install Microsoft C++ Build Tools first |
| `data.xlsx` not found | Confirm `data.xlsx` exists in `backend/` directory |
| Azure deploy fails (Prophet) | Switch to Azure Container Apps with Dockerfile |
| Recharts canvas not rendering | Confirm `react-frontend/src/utils/canvasSimulation.js` exists |

---

## 8. Development Workflow (Daily)

```
1. Start backend:  cd backend && uvicorn main:app --reload --port 8000
2. Start React:    cd frontend && npm run dev
3. Open:           http://localhost:5173
4. Test API:       http://localhost:8000/docs
5. Push to GitHub → Vercel auto-deploys React, Azure auto-deploys backend
```

---

## 9. Git — What Not to Commit

Make sure `.gitignore` includes:
```
# Env files
backend/.env
react-frontend/.env.local

# Python cache
__pycache__/
*.pyc

# Model files (large binary)
backend/models/*.pkl

# Node
react-frontend/node_modules/
react-frontend/dist/

# Excel data (optional — keep if small)
# backend/data.xlsx
```
