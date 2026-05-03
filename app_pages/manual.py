"""
pages/manual.py — Comprehensive user manual with formulas and explanations.
"""
import streamlit as st


def render(_=None):
    st.markdown('<p class="page-title">📚 User Manual</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Complete guide to every metric, formula, and feature</p>',
                unsafe_allow_html=True)

    # ── Quick Navigation ───────────────────────────────────────────────────
    st.info("Use the sections below to understand every metric used in FundScope. "
            "Each section includes the formula, interpretation, and practical guidance.")

    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("📈 Return Metrics", expanded=True):
        st.markdown("### Absolute Return")
        st.latex(r"\text{Absolute Return} = \frac{P_t - P_0}{P_0} \times 100")
        st.markdown("Simple percentage gain/loss since purchase. Does **not** account for time.")

        st.markdown("### CAGR — Compound Annual Growth Rate")
        st.latex(r"\text{CAGR} = \left(\frac{V_{\text{end}}}{V_{\text{start}}}\right)^{\frac{1}{n}} - 1")
        st.markdown(
            "- **n** = number of years\n"
            "- Represents the smoothed annual return rate assuming compounding.\n"
            "- Best used for comparing funds over different time horizons.\n"
            "- **Good CAGR**: Equity funds in India typically target 12–18% over 5+ years."
        )

        st.markdown("### Annualised Return (from daily data)")
        st.latex(r"\text{Ann. Return} = (1 + \bar{r}_{\text{daily}})^{252} - 1")

        st.markdown("### Rolling Returns")
        st.markdown(
            "Return computed over a rolling window (1M, 3M, 6M, 1Y, 3Y). "
            "Shows how returns change at different starting points — useful for identifying "
            "consistency. A fund with stable positive rolling returns is preferable to one "
            "that only looks good at a specific start date."
        )

    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("⚖️ Risk-Adjusted Metrics"):
        st.markdown("### Sharpe Ratio")
        st.latex(r"S = \frac{R_p - R_f}{\sigma_p} \times \sqrt{252}")
        st.markdown(
            "- **Rp** = portfolio return, **Rf** = risk-free rate (6.5% Indian G-Sec)\n"
            "- **σp** = annualised standard deviation of returns\n"
            "- **Interpretation**: Higher is better.\n"
            "  - < 0: Fund underperforms risk-free asset\n"
            "  - 0 – 1: Acceptable\n"
            "  - 1 – 2: Good\n"
            "  - > 2: Excellent"
        )

        st.markdown("### Sortino Ratio")
        st.latex(r"S_o = \frac{R_p - R_f}{\sigma_d} \times \sqrt{252}")
        st.markdown(
            "- **σd** = downside deviation (only negative returns)\n"
            "- Unlike Sharpe, **only penalises bad volatility**.\n"
            "- Preferred metric for evaluating downside risk management."
        )

        st.markdown("### Calmar Ratio")
        st.latex(r"\text{Calmar} = \frac{\text{Annualised Return}}{|\text{Max Drawdown}|}")
        st.markdown(
            "- Measures return per unit of maximum drawdown risk.\n"
            "- **> 1**: Fund earns more than its worst drawdown — good.\n"
            "- **> 3**: Excellent risk-adjusted performance."
        )

        st.markdown("### Volatility (Annualised Std Dev)")
        st.latex(r"\sigma_{\text{ann}} = \sigma_{\text{daily}} \times \sqrt{252}")
        st.markdown(
            "- Measures price fluctuation. Higher volatility = more risk.\n"
            "- Equity funds: typically 15–30% annual volatility.\n"
            "- Debt funds: typically 1–5%."
        )

    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("📉 Drawdown & Risk Metrics"):
        st.markdown("### Maximum Drawdown (MDD)")
        st.latex(r"\text{MDD} = \min_t \left(\frac{NAV_t - \text{Peak}(t)}{\text{Peak}(t)}\right)")
        st.markdown(
            "- Largest peak-to-trough decline in NAV history.\n"
            "- **Interpretation**: A fund with MDD of −40% means you could have lost 40% "
            "from your high-water mark at worst.\n"
            "- Lower is better. Equity funds: expect −20% to −60% in crashes."
        )

        st.markdown("### VaR — Value at Risk")
        st.latex(r"\text{VaR}_{95\%} = \text{Percentile}_{5\%}(\text{Daily Returns})")
        st.markdown(
            "- **Definition**: Maximum expected daily loss with 95% confidence.\n"
            "- Example: VaR 95% = −2% means on any given day, there's only a 5% chance "
            "of losing more than 2%.\n"
            "- **Historical VaR** used here (non-parametric)."
        )

        st.markdown("### CVaR — Conditional VaR (Expected Shortfall)")
        st.latex(r"\text{CVaR}_{95\%} = \mathbb{E}[R \mid R \leq \text{VaR}_{95\%}]")
        st.markdown(
            "- **Definition**: Average loss **given** that the loss exceeds VaR.\n"
            "- Stricter than VaR — tells you what happens in the worst 5% of cases.\n"
            "- Always more negative than VaR. Preferred by regulators (Basel III)."
        )

    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("🏦 Alpha, Beta & Benchmark"):
        st.markdown("### CAPM Regression Model")
        st.latex(r"R_p - R_f = \alpha + \beta (R_m - R_f) + \varepsilon")
        st.markdown(
            "- **α (Alpha)**: Excess return not explained by market — fund manager skill.\n"
            "  - α > 0: Fund outperforms on a risk-adjusted basis ✅\n"
            "  - α < 0: Fund underperforms ❌\n"
            "\n"
            "- **β (Beta)**: Sensitivity to market movements.\n"
            "  - β = 1: Moves with the market\n"
            "  - β > 1: More volatile than market (higher risk & reward)\n"
            "  - β < 1: Less volatile (defensive)\n"
            "\n"
            "- **R²**: How well the benchmark explains the fund's returns.\n"
            "  - R² = 0.9 → 90% of fund movement explained by the benchmark."
        )

    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("📊 Return Distribution"):
        st.markdown("### Skewness")
        st.latex(r"\text{Skewness} = \frac{1}{n}\sum\left(\frac{r_i - \bar{r}}{\sigma}\right)^3")
        st.markdown(
            "- **0**: Symmetric (normal-like)\n"
            "- **Positive**: More upside surprises — desirable for funds\n"
            "- **Negative**: Heavy left tail — more large losses than gains"
        )

        st.markdown("### Excess Kurtosis")
        st.latex(r"\text{Kurtosis} = \frac{1}{n}\sum\left(\frac{r_i - \bar{r}}{\sigma}\right)^4 - 3")
        st.markdown(
            "- **0**: Normal distribution\n"
            "- **Positive (leptokurtic)**: Fat tails — extreme events more likely than normal\n"
            "- Most financial return series have positive kurtosis (fat tails)"
        )

    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("🤖 Prediction Models"):
        models = {
            "Prophet": (
                "Facebook's time-series model. Decomposes into trend + seasonality + holidays. "
                "Handles missing data and outliers well. Best for funds with clear yearly patterns."
            ),
            "ARIMA (2,1,2)": (
                "AutoRegressive Integrated Moving Average. Uses past values and errors to forecast. "
                "d=1 differencing makes the series stationary. Best for short-term, stable trends."
            ),
            "Holt-Winters": (
                "Exponential Smoothing with trend and seasonal components. "
                "Uses weighted averages where recent data gets higher weight. "
                "Good for funds with smooth, predictable trends."
            ),
            "Linear Regression": (
                "Predicts NAV using lagged values and rolling statistics as features. "
                "Recursive multi-step forecasting. Baseline model, interpretable."
            ),
            "Ridge Regression": (
                "Linear regression with L2 regularisation (penalty on large coefficients). "
                "Reduces overfitting on correlated lag features. Better than plain Linear for volatile funds."
            ),
            "Random Forest": (
                "Ensemble of decision trees. Captures non-linear patterns in lag features. "
                "100 trees, no feature scaling needed. Good for complex, non-trending NAV patterns."
            ),
            "SVR (Support Vector)": (
                "Support Vector Regression with RBF kernel. "
                "Robust to outliers due to epsilon-insensitive loss. "
                "Feature scaling required. Good for medium-range forecasts."
            ),
            "XGBoost": (
                "Gradient boosted trees. State-of-the-art for tabular features. "
                "200 estimators, depth 4, learning rate 0.05. "
                "Usually achieves lowest RMSE among individual models."
            ),
        }
        for name, desc in models.items():
            st.markdown(f"**{name}**")
            st.markdown(f"> {desc}")
            st.markdown("")

    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("🔀 Ensemble Learning"):
        st.markdown("### Auto-Optimised Ensemble")
        st.latex(r"\hat{y}_{\text{ensemble}} = \sum_{i=1}^{n} w_i \hat{y}_i")
        st.markdown(
            "- Combines predictions from multiple models.\n"
            "- Weights **w** are optimised using **SLSQP** (Sequential Least Squares Programming).\n"
            "- Objective: minimise validation RMSE on held-out 20% of data.\n"
            "- Constraints: all weights ≥ 0 and sum to 1 (convex combination).\n"
        )
        st.markdown("**How to interpret the weight chart:**")
        st.markdown(
            "- High weight = that model is most reliable for this fund.\n"
            "- Weight = 0 = the model adds noise; excluded automatically.\n"
            "- Ensemble RMSE improvement over best individual shows the diversification benefit."
        )

    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("🌀 Monte Carlo Simulation"):
        st.markdown("### Geometric Brownian Motion (GBM)")
        st.latex(r"S_{t+1} = S_t \cdot \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\cdot Z\right]")
        st.markdown(
            "- **μ** = mean daily return, **σ** = daily volatility\n"
            "- **Z** ~ N(0,1) random shock each day\n"
            "- Runs 100–2000 independent path simulations\n"
            "- Confidence bands show 5th, 25th, 50th, 75th, 95th percentiles\n"
            "\n"
            "**Probability metrics:**\n"
            "- **Prob. of Profit**: % paths that end above initial investment\n"
            "- **Prob. >20% Loss**: % paths that lose more than 20%\n"
            "\n"
            "⚠️ GBM assumes returns are log-normally distributed and independent — "
            "real markets have fat tails and autocorrelation. Treat as indicative, not predictive."
        )

    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("🔁 Backtesting Strategies"):
        st.markdown("### Buy & Hold")
        st.markdown("Invest a lump sum at the first available NAV. No subsequent purchases. "
                    "Best when markets trend consistently upward.")

        st.markdown("### SIP — Systematic Investment Plan")
        st.markdown(
            "Invest a fixed amount every month regardless of NAV. "
            "Benefits from **rupee cost averaging** — buys more units when NAV is low. "
            "Reduces timing risk. Preferred for retail investors."
        )

        st.markdown("### Value Averaging")
        st.latex(r"\text{Contribution}_t = \text{Target}_t - V_t")
        st.markdown(
            "- Target grows at a fixed rate each month.\n"
            "- Invest **more** when portfolio underperforms, **less** when it outperforms.\n"
            "- Theoretically superior to SIP but requires disciplined, variable contributions."
        )

    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("💥 Stress Testing"):
        st.markdown(
            "Applies hypothetical shocks (−50% to +30%) to the current NAV and computes "
            "the resulting portfolio value and P&L.\n\n"
            "Scenarios are inspired by historical Indian market events:\n"
            "- **−50% (Crash)**: 2008 Global Financial Crisis, COVID-19 (2020)\n"
            "- **−30% (Bear)**: Typical bear market correction\n"
            "- **+30% (Bull)**: Post-COVID recovery (2020–21), Modi wave (2014)\n\n"
            "Use this to understand your **maximum realistic downside** in adverse conditions."
        )

    st.markdown("---")
    st.caption("FundScope v1.0 — Built with Streamlit · All metrics are educational, not financial advice.")
