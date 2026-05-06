"""
pages/predictions.py — Model selection, forecasting, comparison & ensemble optimizer.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from modules.predictions import run_model, run_ensemble, _MODEL_FNS
from config import CHART_COLORS, PLOTLY_LAYOUT, PREDICTION_MODELS, hex_to_rgba


def render(all_funds: dict):
    st.markdown('<p class="page-title">🤖 Predictions & Forecasting</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Select models, tune parameters, compare forecasts</p>',
                unsafe_allow_html=True)

    if not all_funds:
        st.warning("No fund data loaded.")
        return

    # ── Controls ──────────────────────────────────────────────────────────
    col_f, col_p = st.columns([2, 1])
    with col_f:
        fund_name = st.selectbox("Select Fund", list(all_funds.keys()))
    with col_p:
        future_days = st.slider("Forecast Horizon (days)", 30, 365, 90)

    df = all_funds[fund_name]

    # ── Model selection ────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Select Models to Run</p>', unsafe_allow_html=True)
    model_cols = st.columns(4)
    selected_models = []
    for i, m in enumerate(PREDICTION_MODELS):
        with model_cols[i % 4]:
            if st.checkbox(m, value=(m in ['Prophet', 'XGBoost']), key=f"chk_{m}"):
                selected_models.append(m)

    run_ensemble_flag = st.checkbox("🔀 Auto-Optimise Ensemble (runs all selected models)", value=False)

    if not selected_models:
        st.info("Please select at least one model.")
        return

    if st.button("▶  Run Predictions", width='stretch'):
        _run_and_show(df, fund_name, selected_models, future_days, run_ensemble_flag)


def _run_and_show(df, fund_name, selected_models, future_days, run_ensemble_flag):
    results    = {}
    metrics    = {}
    fig        = go.Figure()

    # Historical NAV
    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['y'], name='Historical NAV', mode='lines',
        line=dict(color='#94a3b8', width=1.5),
        hovertemplate='%{x|%d %b %Y}<br>NAV: ₹%{y:.2f}<extra></extra>',
    ))

    progress = st.progress(0, "Initialising…")
    n = len(selected_models)

    for i, model_name in enumerate(selected_models):
        progress.progress((i) / n, f"Training {model_name}…")
        try:
            fc, m = run_model(df, model_name, future_days)
            results[model_name] = fc
            metrics[model_name] = m

            color = CHART_COLORS[(i + 1) % len(CHART_COLORS)]
            # Only plot the future portion
            future_fc = fc.tail(future_days)

            fig.add_trace(go.Scatter(
                x=future_fc['ds'], y=future_fc['yhat'],
                name=model_name, mode='lines',
                line=dict(color=color, width=2),
                hovertemplate='%{x|%d %b %Y}<br>%{name}: ₹%{y:.2f}<extra></extra>',
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([future_fc['ds'], future_fc['ds'][::-1]]),
                y=pd.concat([future_fc['yhat_upper'], future_fc['yhat_lower'][::-1]]),
                fill='toself', fillcolor=hex_to_rgba(color, 0.08),
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False, hoverinfo='skip',
            ))
        except Exception as e:
            st.warning(f"⚠ {model_name} failed: {e}")

    progress.progress(1.0, "Done!")

    # Ensemble
    ens_result = None
    if run_ensemble_flag and len(selected_models) >= 2:
        st.info("🔀 Running ensemble optimisation…")
        prog2 = st.progress(0)
        ens_result = run_ensemble(
            df, selected_models, future_days,
            progress_cb=lambda v, msg: prog2.progress(v, msg),
        )
        ef = ens_result['ensemble_forecast']
        if not ef.empty:
            fig.add_trace(go.Scatter(
                x=ef['ds'], y=ef['yhat'], name='Ensemble (Optimised)',
                mode='lines', line=dict(color='#f59e0b', width=3, dash='dash'),
                hovertemplate='%{x|%d %b %Y}<br>Ensemble: ₹%{y:.2f}<extra></extra>',
            ))

    # ── Add vertical "today" line ──────────────────────────────────────────
    today = df['ds'].iloc[-1]
    fig.add_shape(
        type="line", x0=today, x1=today, y0=0, y1=1, yref='paper',
        line=dict(color="#94a3b8", width=1.5, dash="dot")
    )
    fig.add_annotation(
        x=today, y=1, yref='paper', text="Today", showarrow=False,
        xanchor='left', yanchor='top', font=dict(color="#94a3b8", size=10)
    )

    fig.update_layout(PLOTLY_LAYOUT)
    fig.update_layout(height=480,
                      title=f'{fund_name} — NAV Forecast ({future_days} days)',
                      xaxis_title='Date', yaxis_title='NAV (₹)',
                      legend=dict(orientation='h', yanchor='bottom', y=-0.25))
    st.plotly_chart(fig, width='stretch')

    # ── Accuracy metrics table ─────────────────────────────────────────────
    if metrics:
        st.markdown('<p class="section-header">📊 Model Accuracy (Hold-out Test Set)</p>',
                    unsafe_allow_html=True)
        met_df = pd.DataFrame(metrics).T.reset_index()
        met_df.columns = ['Model', 'RMSE', 'MAE', 'MAPE (%)', 'R²']
        best_rmse = met_df['RMSE'].min()
        met_df = met_df.sort_values('RMSE')

        def _badge(row):
            if row['RMSE'] == best_rmse:
                return [''] + ['background-color: #10b98115; color: #10b981'] * 4
            return [''] * 5

        styled_met = met_df.style.apply(_badge, axis=1).format(
            {'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'MAPE (%)': '{:.2f}%', 'R²': '{:.4f}'}
        )
        st.dataframe(styled_met, width='stretch')

    # ── Ensemble recommendations ───────────────────────────────────────────
    if ens_result:
        rec = ens_result['recommendations']
        st.markdown('<p class="section-header">🎯 Ensemble Recommendations</p>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Best Individual Model", rec['best_individual'],
                      delta=f"RMSE {rec['best_individual_rmse']:.4f}")
        with c2:
            st.metric("Ensemble RMSE", f"{rec['ensemble_rmse']:.4f}",
                      delta=f"{rec['improvement_pct']:+.1f}% vs best solo"
                      if rec['improvement_pct'] is not None else None)
        with c3:
            top = rec['top_weights']
            top_str = "\n".join(f"• {n}: {w*100:.1f}%" for n, w in top)
            st.info(f"**Top Weights**\n\n{top_str}")

        # Full weight bar chart
        wdf = pd.DataFrame(list(ens_result['weights'].items()),
                           columns=['Model', 'Weight'])
        wdf = wdf[wdf['Weight'] > 0.001]
        import plotly.express as px
        fig_w = px.bar(wdf, x='Model', y='Weight', color='Weight',
                       color_continuous_scale='Blues',
                       text=wdf['Weight'].map(lambda v: f'{v*100:.1f}%'))
        fig_w.update_layout(PLOTLY_LAYOUT)
        fig_w.update_layout(height=280, coloraxis_showscale=False,
                            title='Optimal Ensemble Weights (SLSQP)')
        st.plotly_chart(fig_w, width='stretch')
