# FundScope — Cloud Deployment Strategy

This document outlines the step-by-step strategy for deploying the FundScope application. The workflow is designed so that **GitHub acts as the central source of truth**, **Azure orchestrates the backend and database connections**, and **Streamlit handles the frontend**.

---

## Phase 1: Version Control & Security (GitHub)
**Goal:** Push the entire codebase to GitHub securely so that other services can fetch from it.

1. **Repository Setup**: Both the `frontend/` and `backend/` folders reside in a single GitHub repository.
2. **Security (`.gitignore`)**: 
   Before pushing, ensure sensitive files are ignored. Your `.gitignore` must include:
   ```text
   .env
   __pycache__/
   .venv/
   venv/
   .streamlit/secrets.toml
   ```
3. **Push to GitHub**: Commit and push the full code to your GitHub repository. This repository will now be the source that Azure and Streamlit Cloud pull from.

---

## Phase 2: Database Layer (Supabase)
**Goal:** Set up the database that Azure will communicate with.

1. **Setup**: Create a new project in Supabase.
2. **Security**: The database must **only** accept connections from the Azure Backend. The Streamlit Frontend will **never** speak directly to Supabase. This hides your database credentials completely from the public web app.
3. **Credentials**: Obtain your `SUPABASE_URL` and `SUPABASE_KEY`. Hold onto these for the Azure configuration step.

---

## Phase 3: Backend & Workflow Configuration (Azure Cloud)
**Goal:** Deploy the FastAPI backend and configure it to securely connect the frontend and the database.

1. **Fetch from GitHub**: Create an **Azure App Service (Linux)**. In the Deployment Center, connect your GitHub account and point it to the FundScope repository. Azure will now automatically fetch the code.
2. **Startup Command**: Tell Azure how to run the backend (since it's in a subfolder). Set the startup command to:
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
   ```
3. **Workflow Configuration (Azure Settings)**:
   This is where the magic happens. In the Azure Portal under **Settings > Configuration > Application settings**, you establish the workflow by securely linking the database and defining who can talk to the backend:
   
   * **Database Connection**: Add `SUPABASE_URL` and `SUPABASE_KEY` as environment variables. The backend will use these to talk to Supabase.
   * **External API**: Add `MFAPI_BASE_URL` with the value `https://api.mfapi.in`.
   * **Frontend Connection (CORS)**: Once you deploy your frontend (Phase 4) and get a public URL (e.g., `https://fundscope-app.streamlit.app`), you must ensure that URL is allowed in the `CORSMiddleware` in `backend/main.py`. This ensures *only* your frontend can request data from your backend.

*(For a more detailed breakdown of Azure deployment, see `azure_backend_deployment.md`)*

---

## Phase 4: Frontend Layer (Streamlit Community Cloud)
**Goal:** Deploy the user interface and connect it to Azure.

1. **Deploy**: Log into Streamlit Community Cloud and connect it to your GitHub repository.
2. **Configuration**: Set the primary file path to `frontend/app.py`.
3. **Connect to Azure**: Streamlit needs to know where the backend is. Add your Azure Backend URL to Streamlit's secrets management (Streamlit Secrets):
   ```toml
   FUNDSCOPE_BACKEND_URL="https://your-backend-app.azurewebsites.net"
   ```
4. **Workflow Execution**: When a user interacts with the live Streamlit app, it sends a request to the Azure backend. The Azure backend securely fetches data from Supabase or `mfapi.in`, processes it, and sends it back to the frontend for visualization.
