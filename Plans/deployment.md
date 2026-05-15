# Deployment Guide: Staged Approach

This document outlines the exact, step-by-step process we will follow to get FundScope deployed to the cloud. We will do this in two stages to ensure everything connects properly.

---

## 🔑 Keys & Links Needed
Before we can complete the connections in the code, I will need you to provide me with the following information as you get it during deployment:

1. **Supabase URL & Anon Key:** Required immediately for both frontend and backend to talk to your database.
2. **Vercel Frontend URL:** Required after Stage 1 (e.g., `https://fundscope.vercel.app`). The backend needs this to allow cross-origin requests (CORS).
3. **Azure Backend URL:** Required after Stage 2 (e.g., `https://fundscope-backend.azurewebsites.net`). The frontend needs this to send ML requests.

---

## Stage 1: GitHub & Vercel (Frontend Deployment)

In this stage, we will push the code to GitHub and deploy the React frontend to Vercel. 

### 1. Push to GitHub
Your `.gitignore` is already perfectly configured (it ignores `.env` files, large `*.pkl` models, and `node_modules`). 
- **Action:** Initialize git, commit all files, and push your repository to GitHub.

### 2. Deploy to Vercel
1. Go to [Vercel.com](https://vercel.com) and import your new GitHub repository.
2. Vercel will automatically detect that it's a Vite/React app.
3. **Important:** In the Vercel deployment settings, under "Environment Variables", you must add:
   - `VITE_SUPABASE_URL` = `<your-supabase-url>`
   - `VITE_SUPABASE_ANON_KEY` = `<your-supabase-anon-key>`
   - `VITE_API_URL` = (Leave blank or put a placeholder for now, we will update this in Stage 2 once Azure is deployed).
4. Click **Deploy**.
5. **Action:** Once Vercel finishes, copy the live URL it gives you and provide it to me.

---

## Stage 2: Azure (Backend Deployment)

Once the frontend is live, we will deploy the FastAPI backend to Azure and connect the two.

### 1. Code Updates (What I will do)
Once you provide the Vercel URL, I will update `backend/main.py` to allow CORS (Cross-Origin Resource Sharing) specifically for your Vercel frontend. Without this, browsers will block the frontend from talking to the backend.

### 2. Deploy to Azure App Service
1. In the Azure Portal, create a new Web App (Linux, Python 3.10+ environment).
2. Connect it to your GitHub repository (via Deployment Center or GitHub Actions).
3. **Environment Variables:** In Azure (Configuration -> Application Settings), you must add:
   - `SUPABASE_URL` = `<your-supabase-url>`
   - `SUPABASE_KEY` = `<your-supabase-key>`
   - `FRONTEND_URL` = `<your-vercel-url>`
4. **Startup Command:** Under Configuration -> General Settings -> Startup Command, set it to:
   ```bash
   gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```
5. **Action:** Once Azure deploys, copy the Azure App Service URL and provide it to me.

### 3. Upload the Model Manually
Because GitHub blocked the large `universal_fund_model.pkl` file, you must upload it directly:
1. Go to **Advanced Tools (Kudu)** in your Azure Web App.
2. Navigate to `site/wwwroot/backend/models/`.
3. Drag and drop your `.pkl` file into that folder.

### 4. Final Connection
Once I have the Azure URL, I will instruct you to go back into your Vercel Environment Variables and update `VITE_API_URL` to your new Azure backend URL, fully connecting the two systems.
