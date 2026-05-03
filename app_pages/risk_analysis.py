"""
pages/risk_analysis.py — VaR, CVaR, Drawdown, Stress Testing.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

from modules.risk import (
    calculate_var, calculate_cvar, calculate_drawdown_series,
    calculate_recovery_periods, stress_test, get_risk_summary,
)
from config import CHART_COLORS, PLOTLY_LAYOUT, hex_to_rgba


def render(all_funds: dict):
    st.markdown('<p class="page-title">⚠️ Risk Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">VaR, CVaR, Drawdown analysis & stress testing</p>',
                unsafe_allow_html=True)

    if not all_funds:
        st.warning("No fund data loaded.")
        return

    col_a, col_b, col_c = st.columns([3, 1, 1])
    with col_a:
        selected = st.multiselect("Select Funds", list(all_funds.keys()),
                                  default=list(all_funds.keys())[:2])
    with col_b:
        confidence = st.selectbox("Confidence Level", [0.90, 0.95, 0.99],
                                  index=1, format_func=lambda x: f"{x*100:.0f}%")
    with col_c:
        investment = st.number_input("Investment (₹)", value=100_000, step=10_000)

    if not selected:
        st.info("Select at least one fund.")
        return

    tab_var, tab_dd, tab_stress = st.tabs(["📊 VaR & CVaR", "📉 Drawdown", "💥 Stress Test"])

    # ─────────────────────────────────────────────────────────────────────
    with tab_var:
        # Risk summary metrics
        st.markdown('<p class="section-header">Risk Metrics Summary</p>', unsafe_allow_html=True)
        risk_rows = []
        for name in selected:
            df  = all_funds[name]
            rs  = get_risk_summary(df, confidence)
            risk_rows.append({
                'Fund':    name,
                'VaR 95%': f"{rs['var_95']*100:.2f}%",
                'VaR 99%': f"{rs['var_99']*100:.2f}%",
                'CVaR 95%': f"{rs['cvar_95']*100:.2f}%",
                'CVaR 99%': f"{rs['cvar_99']*100:.2f}%",
                'Max DD':   f"{rs['max_dd']:.1f}%",
            })
        st.dataframe(pd.DataFrame(risk_rows).set_index('Fund'), width='stretch')

        st.markdown("---")

        # Per-fund histogram with VaR/CVaR lines
        for i, name in enumerate(selected):
            df  = all_funds[name]
            r   = df['returns'].dropna() * 100
            var = calculate_var(df, confidence) * 100
            cvar= calculate_cvar(df, confidence) * 100
            mu, sigma = r.mean(), r.std()

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=r, nbinsx=60, histnorm='probability density',
                marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                opacity=0.7, name='Daily Returns',
            ))
            x_range = np.linspace(r.min(), r.max(), 300)
            fig.add_trace(go.Scatter(
                x=x_range, y=norm.pdf(x_range, mu, sigma),
                name='Normal Fit', line=dict(color='#f59e0b', width=2),
            ))
            fig.add_vline(x=var,  line_color='#ef4444', line_dash='dash',
                          annotation_text=f'VaR {confidence*100:.0f}%: {var:.2f}%',
                          annotation_position='top right')
            fig.add_vline(x=cvar, line_color='#f97316', line_dash='dot',
                          annotation_text=f'CVaR: {cvar:.2f}%',
                          annotation_position='top left')
            fig.update_layout(PLOTLY_LAYOUT)
            fig.update_layout(height=340,
                              title=f'{name} — Return Distribution & Risk Thresholds')
            st.plotly_chart(fig, width='stretch')

            col1, col2, col3 = st.columns(3)
            col1.metric(f"Daily VaR ({confidence*100:.0f}%)", f"{var:.2f}%",
                        help="Maximum expected daily loss at this confidence level")
            col2.metric(f"Daily CVaR ({confidence*100:.0f}%)", f"{cvar:.2f}%",
                        help="Average loss when loss exceeds VaR (Expected Shortfall)")
            col3.metric("₹ at Risk (daily)", f"₹{abs(var/100*investment):,.0f}",
                        help=f"Rupee value at risk for ₹{investment:,.0f} investment")

    # ─────────────────────────────────────────────────────────────────────
    with tab_dd:
        fig_dd = go.Figure()
        for i, name in enumerate(selected):
            dd_df = calculate_drawdown_series(all_funds[name])
            fig_dd.add_trace(go.Scatter(
                x=dd_df['ds'], y=dd_df['drawdown'], name=name,
                mode='lines', fill='tozeroy',
                fillcolor=hex_to_rgba(CHART_COLORS[i % len(CHART_COLORS)], 0.15),
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
            ))
        fig_dd.update_layout(PLOTLY_LAYOUT)
        fig_dd.update_layout(height=380,
                             title='Underwater (Drawdown) Chart',
                             xaxis_title='Date', yaxis_title='Drawdown (%)')
        st.plotly_chart(fig_dd, width='stretch')

        # Recovery periods table
        for name in selected:
            st.markdown(f'<p class="section-header">🔄 {name} — Drawdown Episodes</p>',
                        unsafe_allow_html=True)
            episodes = calculate_recovery_periods(all_funds[name])
            if episodes:
                ep_df = pd.DataFrame(episodes)
                st.dataframe(ep_df, width='stretch')
            else:
                st.info("No significant drawdown episodes found.")

    # ─────────────────────────────────────────────────────────────────────
    with tab_stress:
        st.markdown('<p class="section-header">Stress Test — Portfolio Impact</p>',
                    unsafe_allow_html=True)
        for name in selected:
            st.markdown(f"**{name}**")
            st_df = stress_test(all_funds[name], investment)
            fig_s  = px.bar(
                st_df, x='Scenario', y='PnL',
                color='PnL',
                color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'],
                text=st_df['PnL'].map(lambda v: f"₹{v:,.0f}"),
                title=f"{name} — P&L under Stress Scenarios (₹{investment:,} invested)",
            )
            fig_s.add_hline(y=0, line_dash='dash', line_color='#94a3b8')
            fig_s.update_layout(PLOTLY_LAYOUT)
            fig_s.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(fig_s, width='stretch')
            st.dataframe(st_df.set_index('Scenario'), width='stretch')
            st.markdown("---")
