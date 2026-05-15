# Universal ML Model Strategy (`model.md`)

This document explains how to create and deploy the `universal_fund_model.pkl` to drastically reduce backend CPU usage and latency.

## 1. The Problem with Runtime Training

Currently, when a user requests an ML prediction, the backend typically takes the fund's historical NAV data and trains a model (like XGBoost or Prophet) *on the fly* before making a prediction. 

**Issues with this approach:**
- **High CPU Usage:** Training an XGBoost or Random Forest model takes seconds and heavily utilizes the CPU.
- **Scalability:** If 100 users request predictions at the same time, the server will crash due to concurrent heavy model training.
- **Latency:** Users have to wait for the model to train before seeing results.

## 2. The Solution: Pre-trained Universal Model

Instead of training a model per fund, we train **one massive generalized model** offline, save it as a `.pkl` file, and load it into the backend. 

When a user requests a prediction, the backend just extracts features (like 30-day moving average, RSI, momentum) from the user's fund data and passes it to the loaded model for a nearly instant `predict()` call. **No training happens during the API request.**

### Methodology for Creating a Meaningful Model

Instead of relying on a single script, you need a systematic data science approach to build this model. Here is exactly what you need to consider to ensure the model produces proper, meaningful results:

#### 1. Data Requirements (How Much Data?)
To build a universal model, you need a highly diverse dataset. 
- **Volume**: Aim for at least **5 to 10 years of historical daily NAV data**.
- **Diversity**: Include hundreds of different mutual funds spanning various categories (Large Cap, Mid Cap, Small Cap, Debt, Liquid, and Hybrid).
- **Market Cycles**: Ensure the data covers different market conditions—bull runs, bear markets, and sideways movements. If you only train on a 2-year bull market, the model will fail to predict downturns.
- **Total Records**: You should be aiming for a dataset with **at least 500,000 to over 1,000,000 individual daily records** to allow the model to learn deep, generalized patterns.

#### 2. Feature Engineering (Ensuring Meaningful Results)
The model cannot learn from raw NAV prices because a fund with a NAV of $10 is completely different from one with a NAV of $200. You must transform the data into relative features:
- **Momentum Indicators**: Calculate 1-day, 5-day, and 30-day percentage returns.
- **Moving Averages (SMA/EMA)**: Use the ratio of the current NAV to its 50-day and 200-day moving averages.
- **Risk & Volatility**: Calculate rolling standard deviations (e.g., 30-day volatility) and maximum drawdowns.
- **Macro Factors (Optional but Recommended)**: If possible, merge in broad market indices (like S&P 500 or Nifty 50) to give the model context about the overall market climate.

#### 3. Defining the Target
Don't try to predict the exact NAV price. Instead, predict a specific outcome, such as:
- **Expected Return**: The percentage change in NAV 30 days from today.
- **Risk Categorization**: A classification of whether the fund will drop more than 5% in the next month.

#### 4. Avoiding Pitfalls (Validation)
- **Prevent Data Leakage**: Never use future data to predict the past. Use **Time-Series Split** cross-validation (e.g., train on 2015-2020, test on 2021). Standard randomized train/test splits will artificially inflate your accuracy and lead to terrible real-world performance.
- **Hyperparameter Tuning**: Optimize model parameters (like tree depth in XGBoost or learning rate) to prevent overfitting. The model must generalize, not memorize.

Once these steps are carefully executed, you will export the final trained model (e.g., using `joblib` in Python) into a `.pkl` file.

## 3. How to Deploy

1. Run the script above locally on your powerful laptop/desktop to generate `universal_fund_model.pkl`.
2. Move the `universal_fund_model.pkl` file into your `backend/models/` folder.
3. Because the `.gitignore` ignores `*.pkl` files (since they are usually >50MB and too large for GitHub), you must upload this file directly to your production server (e.g., via FTP or Azure Kudu Zip deploy).
4. When FastAPI starts, `backend/services/model_cache.py` will automatically detect the `.pkl`, load it into RAM once, and use it to instantly serve thousands of predictions using only the `predict()` function. CPU usage will drop by over 95%.
