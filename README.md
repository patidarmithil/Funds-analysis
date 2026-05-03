# FundScope — Mutual Fund Analytics & Prediction

FundScope is a professional, modern web application built with **Streamlit** for comprehensive analysis, prediction, risk assessment, and simulation of mutual fund Net Asset Value (NAV). 

Designed with a sleek, responsive interface (supporting both Dark and Light modes), it empowers investors to make data-driven decisions using advanced machine learning models and probabilistic simulations.

## 🌟 Key Features

* **📊 Portfolio Overview**: Live snapshots across all tracked mutual funds, featuring KPI metric cards, normalized NAV trend comparisons, and return correlation heatmaps.
* **🤖 Advanced Predictions & Forecasting**: 
  * Forecast future NAVs using **8 different lightweight models** (Prophet, ARIMA, Holt-Winters, Linear/Ridge Regression, Random Forest, SVR, and XGBoost).
  * **🔀 Auto-Optimised Ensemble**: Automatically combines predictions from multiple selected models using `scipy.optimize` (SLSQP) to calculate the best possible weights for minimum error.
* **⚠️ Risk Assessment**: Calculates and visualizes key risk metrics like **Value at Risk (VaR)**, **Conditional Value at Risk (CVaR)**, and historical drawdowns to quantify potential portfolio losses.
* **🔁 Investment Backtesting**: Compares the historical performance of two popular investment strategies: **Buy and Hold** (lump-sum investment) versus a **Systematic Investment Plan (SIP)**.
* **🌀 Monte Carlo Simulation**: Runs thousands of random-walk simulations to project a range of possible future NAV paths, providing a probabilistic view of potential outcomes and confidence intervals.
* **🎨 Professional UI/UX**: Custom CSS styling with full support for seamless **Dark and Light mode** toggling, utilizing modern glassmorphism design principles and interactive Plotly charts.

---

## 🚀 Technology Stack

* **Frontend Framework**: [Streamlit](https://streamlit.io/)
* **Data Visualization**: [Plotly Graph Objects & Express](https://plotly.com/python/)
* **Data Processing**: `pandas`, `numpy`, `scipy`
* **Machine Learning & Forecasting**: 
  * `xgboost`
  * `prophet`
  * `scikit-learn` (Random Forest, SVR, Linear/Ridge Regression)
  * `statsmodels` (ARIMA, Holt-Winters)
* **Financial Data (Optional)**: `yfinance` for benchmark comparisons.

---

## ⚙️ Setup and Local Installation

Follow these steps to run the application on your local machine.

### 1. Prerequisites
* Python 3.9 or higher installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 3. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 🌐 Deployment

This application is built with Streamlit and is best deployed using **Streamlit Community Cloud**. 

**Note:** Deployment platforms meant for static sites or serverless functions (like Vercel) are not recommended, as Streamlit requires a continuously running Python server.

### Deploying to Streamlit Community Cloud (Free & Easy)
1. Push this repository to GitHub. Ensure `app.py`, `config.py`, `requirements.txt`, and the folders (`app_pages/`, `modules/`, `styles/`) are in the root directory.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**, select your repository, branch, and set the Main file path to `app.py`.
4. Click **Deploy**. Streamlit will automatically install the requirements and host your application.

---

## 📂 Project Structure

```text
📁 FundScope
│
├── app.py                  # Main Streamlit application entry point
├── config.py               # Global configurations, theme colors, and layout settings
├── requirements.txt        # Python dependencies
├── data.xlsx               # Default mutual fund dataset
│
├── 📁 app_pages/           # Streamlit page views
│   ├── overview.py
│   ├── analysis.py
│   ├── predictions.py
│   ├── risk_analysis.py
│   ├── backtesting.py
│   ├── simulation.py
│   └── manual.py           # Educational documentation for financial terms
│
├── 📁 modules/             # Backend logic and calculation engines
│   ├── data_loader.py
│   ├── analysis.py
│   ├── predictions.py      # ML models and ensemble logic
│   ├── risk.py
│   ├── backtesting.py
│   └── simulation.py
│
└── 📁 styles/
    └── theme.css           # Custom CSS variables for Dark/Light mode styling
```
