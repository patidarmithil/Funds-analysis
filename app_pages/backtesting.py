"""
pages/backtesting.py — Strategy comparison: Buy & Hold, SIP, Value Averaging.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from modules.backtesting import compare_strategies
from config import CHART_COLORS, PLOTLY_LAYOUT, BACKTEST


def render(all_funds: dict):
    st.markdown('<p class="page-title">🔁 Strategy Backtesting</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Compare Buy & Hold, SIP and Value Averaging</p>',
                unsafe_allow_html=True)

    if not all_funds:
        st.warning("No fund data loaded.")
        return

    col_f, col_l, col_s = st.columns(3)
    with col_f:
        selected = st.multiselect("Select Funds", list(all_funds.keys()),
                                  default=list(all_funds.keys())[:2])
    with col_l:
        lumpsum = st.number_input("Lumpsum Investment (Rs)", value=BACKTEST['lumpsum'], step=5_000)
    with col_s:
        monthly = st.number_input("Monthly SIP (Rs)", value=BACKTEST['monthly_sip'], step=500)

    if not selected:
        st.info("Select at least one fund.")
        return

    for fund_name in selected:
        df = all_funds[fund_name]
        combined, summary = compare_strategies(df, lumpsum, monthly)

        st.markdown(f'<p class="section-header">{fund_name}</p>', unsafe_allow_html=True)

        colors_map = {
            'Buy & Hold':      CHART_COLORS[0],
            'SIP':             CHART_COLORS[1],
            'Value Averaging': CHART_COLORS[2],
        }
        fig = go.Figure()
        for strategy, grp in combined.groupby('strategy'):
            fig.add_trace(go.Scatter(
                x=grp['ds'], y=grp['portfolio_value'],
                name=strategy, mode='lines',
                line=dict(color=colors_map.get(strategy, '#ffffff'), width=2),
            ))
        fig.update_layout(PLOTLY_LAYOUT)
        fig.update_layout(height=360,
                          title=f'{fund_name} — Portfolio Growth',
                          xaxis_title='Date', yaxis_title='Portfolio Value (Rs)')
        st.plotly_chart(fig, width='stretch')

        col_tab, col_bar = st.columns([2, 1])
        with col_tab:
            st.dataframe(summary.set_index('Strategy').style.format({
                'total_invested': '{:,.0f}', 'final_value': '{:,.0f}',
                'total_return': '{:.1f}%', 'cagr': '{:.2f}%', 'max_drawdown': '{:.1f}%',
            }), width='stretch')
        with col_bar:
            fig_bar = px.bar(summary, x='Strategy', y='final_value',
                             color='Strategy',
                             color_discrete_sequence=list(colors_map.values()),
                             text=summary['final_value'].map(lambda v: f'{v:,.0f}'))
            fig_bar.update_layout(PLOTLY_LAYOUT)
            fig_bar.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_bar, width='stretch')

        st.markdown("---")
