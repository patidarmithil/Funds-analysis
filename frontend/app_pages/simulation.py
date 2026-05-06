"""
pages/simulation.py — Monte Carlo simulation with confidence bands.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules.simulation import monte_carlo, get_percentile_bands, simulation_summary
from config import PLOTLY_LAYOUT, MONTE_CARLO


def render(all_funds: dict):
    st.markdown('<p class="page-title">🌀 Monte Carlo Simulation</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Geometric Brownian Motion — scenario path analysis</p>',
                unsafe_allow_html=True)

    if not all_funds:
        st.warning("No fund data loaded.")
        return

    col_f, col_d, col_i, col_inv = st.columns(4)
    with col_f:
        fund_name = st.selectbox("Select Fund", list(all_funds.keys()))
    with col_d:
        days = st.slider("Forecast Days", 30, 504, MONTE_CARLO['days'])
    with col_i:
        iters = st.select_slider("Simulations",
                                 options=[100, 250, 500, 1000, 2000], value=1000)
    with col_inv:
        investment = st.number_input("Investment (Rs)", value=10_000, step=1_000)

    df = all_funds[fund_name]

    if st.button("Run Simulation", width='stretch'):
        with st.spinner("Simulating paths…"):
            paths = monte_carlo(df, days=days, iterations=iters)
        bands = get_percentile_bands(paths, [5, 25, 50, 75, 95])
        summary = simulation_summary(paths, investment, df['y'].iloc[-1])

        x_axis = list(range(days + 1))

        # ── Path chart ──────────────────────────────────────────────────────
        fig = go.Figure()

        # Sample 200 paths (grey)
        sample_idx = np.random.choice(iters, min(200, iters), replace=False)
        for idx in sample_idx:
            fig.add_trace(go.Scatter(
                x=x_axis, y=paths[idx], mode='lines',
                line=dict(color='rgba(148,163,184,0.06)', width=1),
                showlegend=False, hoverinfo='skip',
            ))

        # Confidence bands
        pct_colors = {95: '#10b981', 75: '#00d4ff', 50: '#7c3aed', 25: '#f59e0b', 5: '#ef4444'}
        pct_labels = {95: '95th Pct', 75: '75th Pct', 50: 'Median', 25: '25th Pct', 5: '5th Pct'}
        for p in [5, 25, 50, 75, 95]:
            fig.add_trace(go.Scatter(
                x=x_axis, y=bands[p], name=pct_labels[p], mode='lines',
                line=dict(color=pct_colors[p], width=2.5 if p == 50 else 1.5),
            ))

        # Fill between 25–75
        fig.add_trace(go.Scatter(
            x=x_axis + x_axis[::-1],
            y=list(bands[75]) + list(bands[25])[::-1],
            fill='toself', fillcolor='rgba(0,212,255,0.06)',
            line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip',
        ))

        fig.update_layout(PLOTLY_LAYOUT)
        fig.update_layout(height=440,
                          title=f'{fund_name} — Monte Carlo ({iters:,} paths, {days} days)',
                          xaxis_title='Days from Today', yaxis_title='NAV (Rs)')
        st.plotly_chart(fig, width='stretch')

        # ── Summary KPIs ────────────────────────────────────────────────────
        st.markdown('<p class="section-header">Simulation Summary</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median Final Value",  f"Rs {summary['median']:,.0f}")
        c2.metric("Best Case (95th)",    f"Rs {summary['p95']:,.0f}")
        c3.metric("Worst Case (5th)",    f"Rs {summary['p5']:,.0f}")
        c4.metric("Prob. of Profit",     f"{summary['prob_profit']:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Mean Final Value",    f"Rs {summary['mean']:,.0f}")
        c6.metric("25th Percentile",     f"Rs {summary['p25']:,.0f}")
        c7.metric("75th Percentile",     f"Rs {summary['p75']:,.0f}")
        c8.metric("Prob. >20% Loss",     f"{summary['prob_loss20']:.1f}%")

        # ── Final value histogram ────────────────────────────────────────────
        portfolio_finals = (paths[:, -1] / paths[:, 0]) * investment
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=portfolio_finals, nbinsx=60,
            marker_color='#00d4ff', opacity=0.75,
            histnorm='probability density', name='Final Distribution',
        ))
        fig2.add_vline(x=investment, line_dash='dash', line_color='#f59e0b',
                       annotation_text='Invested Amount')
        fig2.add_vline(x=summary['median'], line_dash='dot', line_color='#10b981',
                       annotation_text='Median')
        fig2.update_layout(PLOTLY_LAYOUT)
        fig2.update_layout(height=300,
                           title='Distribution of Final Portfolio Values',
                           xaxis_title='Portfolio Value (Rs)',
                           yaxis_title='Probability Density')
        st.plotly_chart(fig2, width='stretch')
