# Azure Deployment Guide for FastAPI Backend

Since your project is a "monorepo" (meaning both your `frontend/` and `backend/` folders are in the same GitHub repository), the cleanest and most robust way to deploy the backend to Azure is using **Azure App Service (Linux)** combined with **GitHub Actions**.

Here is the exact step-by-step process.

---

## Step 1: Create the Azure App Service
1. Log into the [Azure Portal](https://portal.azure.com/).
2. Search for **App Services** in the top search bar and click **Create -> Web App**.
3. **Project Details:**
   * **Subscription:** (Select yours)
   * **Resource Group:** Create a new one (e.g., `FundScope-RG`)
4. **Instance Details:**
   * **Name:** Give it a unique name (e.g., `fundscope-backend-prod`) - *this will become your URL: `https://fundscope-backend-prod.azurewebsites.net`*
   * **Publish:** Code
   * **Runtime stack:** Python 3.9 (or your current Python version)
   * **Operating System:** Linux
   * **Region:** (Select the closest one to you)
5. **Pricing Plan:** Select **Free F1** or **Basic B1** depending on your budget.
6. Click **Review + Create**, then **Create**.

---

## Step 2: Configure the Startup Command
By default, Azure looks for `app.py` in the root of the repository. Since our FastAPI app is named `main.py` and we are using Gunicorn for production-grade serving:

1. Once the resource is created, go to your new App Service in the Azure Portal.
2. In the left sidebar, under **Settings**, click **Configuration**.
3. Go to the **General settings** tab.
4. In the **Startup Command** field, paste the following exactly:
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
   ```
5. Click **Save** at the top.

---

## Step 3: Add Environment Variables (App Settings)
While still on the **Configuration** page, go to the **Application settings** tab. 
Here, you add your secret variables so you don't commit them to GitHub.

Click **New application setting** and add the following:
* **Name:** `SUPABASE_URL` | **Value:** *(Your Supabase URL)*
* **Name:** `SUPABASE_KEY` | **Value:** *(Your Supabase Anon/Service Key)*
* **Name:** `MFAPI_BASE_URL` | **Value:** `https://api.mfapi.in`

*(Note: We will add CORS frontend URLs here later once Streamlit Cloud gives you your public URL).*
Click **Save**.

---

## Step 4: Link GitHub for Automated Deployment
Because your backend code is inside a subfolder (`/backend`), we need to tell the deployment script to look in that folder.

1. In the left sidebar of your App Service, under **Deployment**, click **Deployment Center**.
2. **Source:** Select **GitHub**.
3. Authorize Azure to access your GitHub if prompted.
4. **Organization / Repository / Branch:** Select your FundScope repository and the `main` branch.
5. **Workflow Option:** Leave it on "Add a workflow" (Azure will automatically generate a GitHub Actions file for you).
6. Click **Save**.

---

## Step 5: Tweak the GitHub Actions Workflow File
When you clicked Save in Step 4, Azure automatically pushed a `.yaml` file to your GitHub repository inside `.github/workflows/`.

1. Go to your repository on **GitHub.com**.
2. Navigate to `.github/workflows/` and open the newly created `main_fundscope...yml` file.
3. Click the pencil icon to **Edit** the file.
4. You need to change the **working directory** so the script knows to install and run from the `backend/` folder, not the root.
   
   Look for the `env:` section at the top and add a variable for your backend folder path. Example of what you need to change:

   ```yaml
   env:
     AZURE_WEBAPP_NAME: fundscope-backend-prod
     PYTHON_VERSION: '3.9'
     WORKING_DIRECTORY: './backend'  # <--- ADD THIS LINE
   ```

5. Further down in the `Build` job, update the `pip install` step to cd into the directory first:
   ```yaml
   - name: Install dependencies
     run: |
       cd ${{ env.WORKING_DIRECTORY }}
       python -m pip install --upgrade pip
       pip install -r requirements.txt
   ```
6. Update the zip creation step so it zips the `backend` folder contents instead of the whole repo:
   ```yaml
   - name: Zip artifact for deployment
     run: |
       cd ${{ env.WORKING_DIRECTORY }}
       zip release.zip ./* -r
       mv release.zip ../
   ```
7. Click **Commit Changes** on GitHub.

*(Alternatively, you can just pull the latest changes to your local machine, edit the workflow file in VS Code, and push it back up!)*

---

## Step 6: Watch it Deploy!
1. In your GitHub repository, click the **Actions** tab at the top.
2. You will see your deployment script running. 
3. Once it turns green, go to your Azure URL: `https://your-app-name.azurewebsites.net/mf/health`
4. You should see `{"status":"ok", ...}`

**Congratulations! Your backend is now live on Azure.**
