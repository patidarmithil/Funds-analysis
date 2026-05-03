"""
app.py — FundScope: Professional Mutual Fund Analytics Dashboard
Run: streamlit run app.py
"""
import os
import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="FundScope — Mutual Fund Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS theme ───────────────────────────────────────────────────────────
_CSS_PATH = os.path.join(os.path.dirname(__file__), "styles", "theme.css")
if os.path.exists(_CSS_PATH):
    with open(_CSS_PATH, encoding="utf-8") as f:
        theme_css = f.read()
        # Inject theme class into body via a wrapper or by replacing stApp class
        st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)

# ── Imports (after page config) ──────────────────────────────────────────────
from modules.data_loader import load_all_funds
import app_pages.overview      as pg_overview
import app_pages.analysis      as pg_analysis
import app_pages.predictions   as pg_predictions
import app_pages.risk_analysis as pg_risk
import app_pages.backtesting   as pg_backtest
import app_pages.simulation    as pg_simulation
import app_pages.manual        as pg_manual
import config

# ── Navigation definition ────────────────────────────────────────────────────
PAGES = {
    "📊  Overview":         pg_overview,
    "📈  Analysis":         pg_analysis,
    "🤖  Predictions":      pg_predictions,
    "⚠️  Risk Analysis":    pg_risk,
    "🔁  Backtesting":      pg_backtest,
    "🌀  Simulation":       pg_simulation,
    "📚  User Manual":      pg_manual,
}

# ── Session state defaults ────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "📊  Overview"
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# ── Theme CSS Injection ──────────────────────────────────────────────────────
_DARK_VARS = """
:root {
    --bg-color:        #0a0e1a;
    --sidebar-bg:      #0f1629;
    --surface-color:   #0f1629;
    --surface-color-2: #162040;
    --border-color:    #1e2d4d;
    --text-main:       #e2e8f0;
    --text-muted:      #94a3b8;
    --primary-color:   #00d4ff;
    --btn-text:        #00d4ff;
    --tab-bg:          #0f1629;
    --input-bg:        #162040;
    --card-shadow:     rgba(0,0,0,0.4);
    --hover-shadow:    rgba(0,212,255,0.18);
}
"""

_LIGHT_VARS = """
:root {
    --bg-color:        #f0f4f8;
    --sidebar-bg:      #ffffff;
    --surface-color:   #ffffff;
    --surface-color-2: #e8eef6;
    --border-color:    #cbd5e1;
    --text-main:       #0f172a;
    --text-muted:      #475569;
    --primary-color:   #0284c7;
    --btn-text:        #0284c7;
    --tab-bg:          #ffffff;
    --input-bg:        #f1f5f9;
    --card-shadow:     rgba(0,0,0,0.08);
    --hover-shadow:    rgba(2,132,199,0.14);
}
"""

if st.session_state.theme == "Light":
    st.markdown(f"<style>{_LIGHT_VARS}</style>", unsafe_allow_html=True)
    config.PLOTLY_LAYOUT.update(config.PLOTLY_LIGHT)
else:
    st.markdown(f"<style>{_DARK_VARS}</style>", unsafe_allow_html=True)
    config.PLOTLY_LAYOUT.update(config.PLOTLY_DARK)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Logo / Brand ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.5rem 0;">
        <div style="font-size:2.2rem;">📈</div>
        <div style="font-size:1.3rem; font-weight:700; color:var(--text-main); letter-spacing:0.05em;">
            FundScope
        </div>
        <div style="font-size:0.72rem; color:var(--text-muted); letter-spacing:0.1em; text-transform:uppercase;">
            Mutual Fund Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Theme Toggle ───────────────────────────────────────────────────────
    t1, t2 = st.columns([2, 1])
    with t1:
        st.markdown('<p style="font-size:0.8rem; font-weight:600; margin-top:5px;">APPEARANCE</p>', unsafe_allow_html=True)
    with t2:
        new_theme = st.selectbox("Theme", ["Dark", "Light"], 
                                 index=0 if st.session_state.theme == "Dark" else 1,
                                 label_visibility="collapsed")
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

    st.markdown("---")
    st.markdown('<p class="section-header">Navigation</p>', unsafe_allow_html=True)

    # ── Nav buttons ────────────────────────────────────────────────────────
    for page_name in PAGES:
        is_active = st.session_state.page == page_name
        # Note: custom CSS classes are hard with st.button, so we use session state logic
        if st.button(
            page_name,
            key=f"nav_{page_name}",
            use_container_width=True,
            type="secondary" if not is_active else "primary"
        ):
            st.session_state.page = page_name
            st.rerun()

    st.markdown("---")

    # ── Data source ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Data Source</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload custom data.xlsx",
        type=["xlsx"],
        help="Sheets must be named after funds with 'Date' and 'NAV' columns.",
    )
    if uploaded is not None:
        st.session_state.uploaded_bytes = uploaded.getvalue()
        st.success("Custom file loaded ✓")
    else:
        if st.session_state.uploaded_bytes is None:
            if os.path.exists(config.DEFAULT_FILE):
                st.info("Using default data.xlsx")
            else:
                st.warning("No data file found. Please upload.")

    st.markdown("---")
    st.caption("v1.1 · Professional Analytics")

# ═══════════════════════════════════════════════════════════════════════════════
# RENDER CONTENT
# ═══════════════════════════════════════════════════════════════════════════════


with st.spinner("Loading fund data…"):
    # load_all_funds is already cached in data_loader.py
    all_funds = load_all_funds(st.session_state.uploaded_bytes)

if not all_funds:
    st.error("⚠️ Could not load any fund data.")
    st.stop()

# RENDER SELECTED PAGE
current_page = PAGES[st.session_state.page]
current_page.render(all_funds)


