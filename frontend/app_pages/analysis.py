"""
pages/analysis.py — Deep-dive analysis: NAV, rolling returns, volatility,
                    return distribution, and correlation drill-down.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm

from modules.analysis import (
    calculate_rolling_returns, calculate_rolling_volatility,
    calculate_alpha_beta, get_fund_summary,
)
from modules.data_loader import load_benchmark, align_benchmark
from config import CHART_COLORS, PLOTLY_LAYOUT, BENCHMARKS


def render(all_funds: dict):
    st.markdown('<p class="page-title">📈 Deep-Dive Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Rolling returns, volatility, distribution & correlation</p>',
                unsafe_allow_html=True)

    if not all_funds:
        st.warning("No fund data loaded.")
        return

    # ── Controls ──────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([3, 2, 2])
    with col_a:
        selected = st.multiselect("Select Funds", list(all_funds.keys()),
                                  default=list(all_funds.keys())[:3])
    with col_b:
        bm_label = st.selectbox("Benchmark (Alpha/Beta)", list(BENCHMARKS.keys()))
    with col_c:
        vol_window = st.slider("Rolling Volatility Window (days)", 10, 90, 30)

    if not selected:
        st.info("Please select at least one fund.")
        return

    tab_nav, tab_roll, tab_vol, tab_dist, tab_ab = st.tabs([
        "📊 Historical NAV", "🔄 Rolling Returns",
        "🌊 Volatility", "📉 Distribution", "⚖️ Alpha & Beta",
    ])

    # ─────────────────────────────────────────────────────────────────────
    with tab_nav:
        fig = go.Figure()
        for i, name in enumerate(selected):
            df = all_funds[name]
            fig.add_trace(go.Scatter(
                x=df['ds'], y=df['y'], name=name, mode='lines',
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                hovertemplate='%{x|%d %b %Y}<br>NAV: ₹%{y:.2f}<extra></extra>',
            ))
            # 50-day MA
            ma = df.set_index('ds')['y'].rolling(50).mean().reset_index()
            fig.add_trace(go.Scatter(
                x=ma['ds'], y=ma['y'], name=f"{name[:10]} (50-MA)",
                mode='lines',
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1, dash='dot'),
                opacity=0.6,
            ))
        fig.update_layout(PLOTLY_LAYOUT)
        fig.update_layout(height=450,
                          title='Historical NAV with 50-Day Moving Average',
                          xaxis_title='Date', yaxis_title='NAV (₹)')
        st.plotly_chart(fig, width='stretch')

    # ─────────────────────────────────────────────────────────────────────
    with tab_roll:
        periods = ['1M', '3M', '6M', '1Y', '3Y']
        period_sel = st.radio("Period", periods, horizontal=True)

        fig2 = go.Figure()
        for i, name in enumerate(selected):
            df  = all_funds[name]
            rr  = calculate_rolling_returns(df)
            fig2.add_trace(go.Scatter(
                x=rr['ds'], y=rr[period_sel], name=name, mode='lines',
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.8),
                hovertemplate='%{x|%d %b %Y}<br>Return: %{y:.1f}%<extra></extra>',
            ))
        fig2.add_hline(y=0, line_dash='dash', line_color='#94a3b8', line_width=1)
        fig2.update_layout(PLOTLY_LAYOUT)
        fig2.update_layout(height=400,
                           title=f'{period_sel} Rolling Return (%)',
                           xaxis_title='Date', yaxis_title='Return (%)')
        st.plotly_chart(fig2, width='stretch')

        # Period-return heatmap
        st.markdown('<p class="section-header">Period Returns Heatmap</p>', unsafe_allow_html=True)
        heat_data = {}
        for name in selected:
            df = all_funds[name]
            rr = calculate_rolling_returns(df)
            heat_data[name] = {p: rr[p].iloc[-1] for p in periods}
        heat_df = pd.DataFrame(heat_data).T
        fig3 = px.imshow(heat_df, text_auto='.1f', color_continuous_scale='RdYlGn',
                         aspect='auto', labels=dict(color='Return (%)'))
        fig3.update_layout(**PLOTLY_LAYOUT, height=300)
        st.plotly_chart(fig3, width='stretch')

    # ─────────────────────────────────────────────────────────────────────
    with tab_vol:
        fig4 = go.Figure()
        for i, name in enumerate(selected):
            df  = all_funds[name]
            rv  = calculate_rolling_volatility(df, vol_window)
            fig4.add_trace(go.Scatter(
                x=rv['ds'], y=rv['volatility'] * 100, name=name, mode='lines',
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            ))
        fig4.update_layout(**PLOTLY_LAYOUT, height=400,
                           title=f'{vol_window}-Day Rolling Annualised Volatility (%)',
                           xaxis_title='Date', yaxis_title='Volatility (%)')
        st.plotly_chart(fig4, width='stretch')

    # ─────────────────────────────────────────────────────────────────────
    with tab_dist:
        fig5 = make_subplots(
            rows=1, cols=len(selected),
            subplot_titles=selected,
        )
        for i, name in enumerate(selected, 1):
            df  = all_funds[name]
            r   = df['returns'].dropna() * 100
            mu, std = r.mean(), r.std()
            x_range = np.linspace(r.min(), r.max(), 200)

            fig5.add_trace(go.Histogram(
                x=r, nbinsx=50, name=name, showlegend=False,
                marker_color=CHART_COLORS[(i-1) % len(CHART_COLORS)],
                histnorm='probability density', opacity=0.75,
            ), row=1, col=i)
            fig5.add_trace(go.Scatter(
                x=x_range, y=norm.pdf(x_range, mu, std),
                mode='lines', name='Normal fit', showlegend=(i == 1),
                line=dict(color='#f59e0b', width=2),
            ), row=1, col=i)

        fig5.update_layout(PLOTLY_LAYOUT)
        fig5.update_layout(height=380,
                           title='Daily Return Distribution vs Normal')
        st.plotly_chart(fig5, width='stretch')

        # Distribution stats table
        dist_rows = []
        for name in selected:
            df  = all_funds[name]
            r   = df['returns'].dropna() * 100
            dist_rows.append({
                'Fund': name, 'Mean (%)': round(r.mean(), 4),
                'Std (%)': round(r.std(), 4),
                'Skewness': round(r.skew(), 3),
                'Kurtosis': round(r.kurt(), 3),
                'Min (%)': round(r.min(), 2), 'Max (%)': round(r.max(), 2),
            })
        st.dataframe(pd.DataFrame(dist_rows).set_index('Fund'), width='stretch')

    # ─────────────────────────────────────────────────────────────────────
    with tab_ab:
        bm_ticker = BENCHMARKS[bm_label]
        with st.spinner(f"Fetching {bm_label} data…"):
            # Use first / last dates across selected funds
            all_starts = [all_funds[n]['ds'].min() for n in selected]
            all_ends   = [all_funds[n]['ds'].max() for n in selected]
            bm_df = load_benchmark(bm_ticker,
                                   start=str(min(all_starts).date()),
                                   end=str(max(all_ends).date()))

        if bm_df is None:
            st.error(f"Could not fetch {bm_label} from Yahoo Finance. Check your internet connection.")
        else:
            ab_rows = []
            for name in selected:
                fr, br = align_benchmark(all_funds[name], bm_df)
                ab = calculate_alpha_beta(fr, br)
                ab_rows.append({'Fund': name,
                                'Alpha (ann.)': round(ab['alpha'] * 100, 2),
                                'Beta': round(ab['beta'], 3),
                                'R²': round(ab['r_squared'], 3)})
            ab_df = pd.DataFrame(ab_rows).set_index('Fund')
            st.dataframe(ab_df.style.format({
                'Alpha (ann.)': '{:.2f}%', 'Beta': '{:.3f}', 'R²': '{:.3f}'
            }), width='stretch')

            st.info(
                "**Alpha** > 0 means the fund outperformed the benchmark on a risk-adjusted basis.  \n"
                "**Beta** > 1 means the fund is more volatile than the market.  \n"
                "**R²** close to 1 means the fund closely tracks the benchmark."
            )
