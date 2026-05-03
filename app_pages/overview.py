"""
pages/overview.py — Dashboard overview: KPI cards, fund summary table, top performers.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from modules.analysis import get_fund_summary, calculate_correlation_matrix
from config import CHART_COLORS, PLOTLY_LAYOUT


def render(all_funds: dict):
    st.markdown('<p class="page-title">📊 Portfolio Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Live snapshot across all tracked mutual funds</p>',
                unsafe_allow_html=True)

    if not all_funds:
        st.warning("No fund data loaded. Please upload a data file.")
        return

    # ── Build summary table ────────────────────────────────────────────────
    rows = []
    for name, df in all_funds.items():
        s = get_fund_summary(df)
        s['Fund'] = name
        rows.append(s)
    summary = pd.DataFrame(rows).set_index('Fund')

    # ── Top KPI row ────────────────────────────────────────────────────────
    cols = st.columns(5)
    _kpi(cols[0], "Total Funds",      len(all_funds),    suffix="",  fmt=".0f")
    best = summary['ret_1y'].dropna().idxmax()
    _kpi(cols[1], "Best Fund (1Y)",   summary.loc[best, 'ret_1y'], suffix="%", name=best[:12])
    worst = summary['ret_1y'].dropna().idxmin()
    _kpi(cols[2], "Worst Fund (1Y)",  summary.loc[worst,'ret_1y'], suffix="%", name=worst[:12])
    _kpi(cols[3], "Avg Sharpe",       summary['sharpe'].mean(),    fmt=".2f")
    _kpi(cols[4], "Avg Max Drawdown", summary['max_drawdown'].mean()*100, suffix="%", fmt=".1f")

    st.markdown("---")

    # ── Fund summary table ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">Fund Summary Table</p>', unsafe_allow_html=True)
    display = summary[[
        'current_nav','ret_1m','ret_3m','ret_6m','ret_1y',
        'cagr','sharpe','sortino','calmar','max_drawdown','volatility',
    ]].copy()
    display.columns = [
        'NAV','1M %','3M %','6M %','1Y %',
        'CAGR','Sharpe','Sortino','Calmar','Max DD','Volatility',
    ]

    def _color(val, col):
        if col in ('1M %','3M %','6M %','1Y %','CAGR','Sharpe','Sortino','Calmar'):
            if isinstance(val, float):
                return 'color: #10b981' if val > 0 else 'color: #ef4444'
        if col in ('Max DD','Volatility') and isinstance(val, float):
            return 'color: #ef4444' if val < -20 else ''
        return ''

    fmt_map = {
        'NAV': '{:.2f}', '1M %': '{:.1f}%', '3M %': '{:.1f}%',
        '6M %': '{:.1f}%', '1Y %': '{:.1f}%',
        'CAGR': '{:.1%}', 'Sharpe': '{:.2f}', 'Sortino': '{:.2f}',
        'Calmar': '{:.2f}', 'Max DD': '{:.1%}', 'Volatility': '{:.1%}',
    }

    styled = display.style.format(fmt_map, na_rep='—')
    st.dataframe(styled, width='stretch')

    st.markdown("---")

    # ── Top & Worst performers ──────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<p class="section-header">🏆 Top Performers (1Y)</p>', unsafe_allow_html=True)
        top5 = summary['ret_1y'].dropna().nlargest(5).reset_index()
        top5.columns = ['Fund', '1Y Return (%)']
        fig = px.bar(top5, x='1Y Return (%)', y='Fund', orientation='h',
                     color='1Y Return (%)', color_continuous_scale='Teal',
                     text_auto='.1f')
        fig.update_layout(PLOTLY_LAYOUT)
        fig.update_layout(height=280, coloraxis_showscale=False)
        fig.update_yaxes(categoryorder='total ascending')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width='stretch')

    with col_r:
        st.markdown('<p class="section-header">📉 Worst Performers (1Y)</p>', unsafe_allow_html=True)
        bot5 = summary['ret_1y'].dropna().nsmallest(5).reset_index()
        bot5.columns = ['Fund', '1Y Return (%)']
        fig2 = px.bar(bot5, x='1Y Return (%)', y='Fund', orientation='h',
                      color='1Y Return (%)', color_continuous_scale='Reds_r',
                      text_auto='.1f')
        fig2.update_layout(PLOTLY_LAYOUT)
        fig2.update_layout(height=280, coloraxis_showscale=False)
        fig2.update_yaxes(categoryorder='total ascending')
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, width='stretch')

    st.markdown("---")

    # ── NAV Trend (all funds normalised to 100) ────────────────────────────
    st.markdown('<p class="section-header">📈 Normalised NAV Trends (Base = 100)</p>',
                unsafe_allow_html=True)
    fig3 = go.Figure()
    for i, (name, df) in enumerate(all_funds.items()):
        base = df['y'].iloc[0]
        fig3.add_trace(go.Scatter(
            x=df['ds'], y=df['y'] / base * 100,
            name=name, mode='lines',
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
        ))
    fig3.update_layout(PLOTLY_LAYOUT)
    fig3.update_layout(height=380,
                       xaxis_title='Date', yaxis_title='Indexed NAV',
                       title='All Funds — Normalised Performance')
    st.plotly_chart(fig3, width='stretch')

    # ── Correlation heatmap ────────────────────────────────────────────────
    st.markdown('<p class="section-header">🔗 Return Correlation Matrix</p>', unsafe_allow_html=True)
    corr = calculate_correlation_matrix(all_funds)
    fig4 = px.imshow(
        corr, text_auto='.2f', aspect='auto',
        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
    )
    fig4.update_layout(PLOTLY_LAYOUT)
    fig4.update_layout(height=480)
    fig4.update_traces(textfont_size=10)
    st.plotly_chart(fig4, width='stretch')


# ─── Helper ──────────────────────────────────────────────────────────────────

def _kpi(col, label, value, suffix="", fmt=".1f", name=None):
    with col:
        if isinstance(value, float):
            disp = f"{value:{fmt}}{suffix}"
        else:
            disp = f"{value}{suffix}"
        sub = f"<small style='color:#94a3b8'>{name}</small>" if name else ""
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{disp}</div>
          {sub}
        </div>
        """, unsafe_allow_html=True)
