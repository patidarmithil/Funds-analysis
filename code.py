from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk


# Load data for each fund
def load_fund_data(sheet_name):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    data.columns = data.columns.str.strip()
    data.rename(columns={'Date': 'ds', 'NAV': 'y'}, inplace=True)
    data['ds'] = pd.to_datetime(data['ds'])
    data['returns'] = data['y'].pct_change() * 100
    return data

# Prophet Prediction with Fine-Tuning
def predict_nav_with_fine_tuning(data, future_months=6):
    model = Prophet(
        growth="linear",
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.add_seasonality(name="quarterly", period=91.25, fourier_order=8)
    model.fit(data)
    future = model.make_future_dataframe(periods=future_months * 30)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Risk Assessment
def calculate_risk(data, confidence_level=0.95):
    returns = data['returns'].dropna()
    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

# Backtesting Strategies
def buy_and_hold_strategy(data, investment=1000):
    start_nav = data['y'].iloc[0]
    end_nav = data['y'].iloc[-1]
    return (investment / start_nav) * end_nav

def sip_strategy(data, monthly_investment=1000):
    units = 0
    for nav in data['y']:
        units += monthly_investment / nav
    return units * data['y'].iloc[-1]

# Scenario Simulation
def monte_carlo_simulation(data, days=180, iterations=1000):
    returns = data['returns'].dropna()
    mean = np.mean(returns) / 100  # Convert to percentage
    std_dev = np.std(returns) / 100
    last_nav = data['y'].iloc[-1]

    simulated_paths = []
    for _ in range(iterations):
        future_prices = [last_nav]
        for _ in range(days):
            next_price = future_prices[-1] * np.exp(mean + std_dev * np.random.normal())
            future_prices.append(next_price)
        simulated_paths.append(future_prices)

    return simulated_paths


# Main Application
class FundApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mutual Fund Analysis")
        self.root.geometry("1200x800")

        # Fund selection listbox
        self.fund_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, exportselection=0)
        for name in fund_names:
            self.fund_listbox.insert(tk.END, name)
        self.fund_listbox.grid(row=0, column=0, rowspan=6)

        # Buttons
        self.plot_button = tk.Button(root, text="Predict NAV", command=self.plot_predictions)
        self.plot_button.grid(row=6, column=0)

        self.backtest_button = tk.Button(root, text="Backtesting", command=self.run_backtesting)
        self.backtest_button.grid(row=7, column=0)

        self.risk_button = tk.Button(root, text="Risk Assessment", command=self.show_risk_analysis)
        self.risk_button.grid(row=8, column=0)

        self.simulate_button = tk.Button(root, text="Scenario Simulation", command=self.run_simulation)
        self.simulate_button.grid(row=9, column=0)

    def plot_predictions(self):
        selected_funds = [self.fund_listbox.get(i) for i in self.fund_listbox.curselection()]
        for fund_name in selected_funds:
            data = load_fund_data(fund_name)
            forecast = predict_nav_with_fine_tuning(data)
            self.create_prediction_window(fund_name, data, forecast)

    def create_prediction_window(self, fund_name, data, forecast):
        window = tk.Toplevel(self.root)
        window.title(f"{fund_name} - Prediction")
        figure, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['ds'], data['y'], label="Historical NAV", color="blue")
        ax.plot(forecast['ds'], forecast['yhat'], label="Predicted NAV", color="orange")
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color="orange", alpha=0.2)
        ax.set_title(f"{fund_name} NAV Prediction")
        ax.legend()
        canvas = FigureCanvasTkAgg(figure, window)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def run_backtesting(self):
        selected_funds = [self.fund_listbox.get(i) for i in self.fund_listbox.curselection()]
        for fund_name in selected_funds:
            data = load_fund_data(fund_name)
            buy_and_hold = buy_and_hold_strategy(data)
            sip = sip_strategy(data)
            self.show_backtesting_graph(fund_name, data, buy_and_hold, sip)

    def show_backtesting_graph(self, fund_name, data, buy_and_hold, sip):
        window = tk.Toplevel(self.root)
        window.title(f"{fund_name} - Backtesting")
        figure, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['ds'], data['y'], label="NAV", color="blue")
        ax.axhline(buy_and_hold, color="green", linestyle="--", label=f"Buy & Hold: {buy_and_hold:.2f}")
        ax.axhline(sip, color="orange", linestyle="--", label=f"SIP: {sip:.2f}")
        ax.legend()
        canvas = FigureCanvasTkAgg(figure, window)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def show_risk_analysis(self):
        selected_funds = [self.fund_listbox.get(i) for i in self.fund_listbox.curselection()]
        for fund_name in selected_funds:
            data = load_fund_data(fund_name)
            var, cvar = calculate_risk(data)
            self.show_risk_graph(fund_name, data, var, cvar)

    def show_risk_graph(self, fund_name, data, var, cvar):
        window = tk.Toplevel(self.root)
        window.title(f"{fund_name} - Risk Assessment")
        figure, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data['returns'].dropna(), bins=30, color="blue", alpha=0.7, label="Historical Returns")
        ax.axvline(var, color="red", linestyle="--", label=f"VaR (95%): {var:.2f}")
        ax.axvline(cvar, color="orange", linestyle="--", label=f"CVaR (95%): {cvar:.2f}")
        ax.legend()
        canvas = FigureCanvasTkAgg(figure, window)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def run_simulation(self):
        selected_funds = [self.fund_listbox.get(i) for i in self.fund_listbox.curselection()]
        for fund_name in selected_funds:
            data = load_fund_data(fund_name)
            paths = monte_carlo_simulation(data)
            self.show_simulation_graph(fund_name, paths)

    def show_simulation_graph(self, fund_name, paths):
        window = tk.Toplevel(self.root)
        window.title(f"{fund_name} - Scenario Simulation")
        figure, ax = plt.subplots(figsize=(10, 6))
        for path in paths:
            ax.plot(path, color="gray", alpha=0.3)
        ax.set_title(f"{fund_name} - Monte Carlo Simulation (NAV Paths)")
        canvas = FigureCanvasTkAgg(figure, window)
        canvas.get_tk_widget().pack()
        canvas.draw()


# File path and fund names
file_path = r'D:\Coding\Python\Code\Funds\data.xlsx'
fund_names = ['Flexi Cap', 'India PSU', 'Infrastructure', 'Midcap', 'Focused India', 'Large and midcap fund', 'Contra',
              'Multicap', 'Financial Services', 'ESG Integration Strategy', 'ELSS Tax Saver', 'Invesco Pan European',
              'Global Consumer Trends', 'EQQQ NASDAQ-100 ETF']

# Start the application
root = tk.Tk()
app = FundApp(root)
root.mainloop()
