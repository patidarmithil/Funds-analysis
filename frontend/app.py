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
import app_pages.home          as pg_home
import app_pages.overview      as pg_overview
import app_pages.analysis      as pg_analysis
import app_pages.predictions   as pg_predictions
import app_pages.risk_analysis as pg_risk
import app_pages.backtesting   as pg_backtest
import app_pages.simulation    as pg_simulation
import app_pages.manual        as pg_manual
import app_pages.new_analysis  as pg_new_analysis
import config

# ── Navigation definition ────────────────────────────────────────────────────
PAGES = {
    "🏠  Home":             pg_home,
    "📊  Overview":         pg_overview,
    "📈  Analysis":         pg_analysis,
    "🤖  Predictions":      pg_predictions,
    "⚠️  Risk Analysis":    pg_risk,
    "🔁  Backtesting":      pg_backtest,
    "🌀  Simulation":       pg_simulation,
    "📚  User Manual":      pg_manual,
    "🔍  New Analysis":     pg_new_analysis,
}

# ── Session state defaults ────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "🏠  Home"
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
# Live analysis mode
if "data_mode" not in st.session_state:
    st.session_state.data_mode = "default"
if "live_funds_data" not in st.session_state:
    st.session_state.live_funds_data = {}
if "live_selected_funds" not in st.session_state:
    st.session_state.live_selected_funds = []
if "live_date_range" not in st.session_state:
    st.session_state.live_date_range = (None, None)

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
# TOP HEADER & NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
head_c1, head_c2, head_c3 = st.columns([5, 3, 2])
with head_c1:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; padding: 0.5rem 0;">
        <div style="font-size:2.5rem; line-height: 1;">📈</div>
        <div>
            <div style="font-size:1.6rem; font-weight:800; color:var(--text-main); letter-spacing:0.05em; line-height: 1.1;">
                FundScope
            </div>
            <div style="font-size:0.8rem; color:var(--text-muted); letter-spacing:0.1em; text-transform:uppercase;">
                Professional Mutual Fund Analytics
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with head_c2:
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    if st.session_state.data_mode == "default":
        st.info("📂 Data Mode: **Sample Report**")
    elif st.session_state.live_funds_data:
        n = len(st.session_state.live_funds_data)
        st.success(f"🟢 Data Mode: **Live** ({n} fund{'s' if n > 1 else ''} loaded)")
    else:
        st.warning("🟡 Data Mode: **Live** (No data loaded)")

with head_c3:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    new_theme = st.selectbox("Appearance", ["Dark", "Light"], 
                             index=0 if st.session_state.theme == "Dark" else 1,
                             label_visibility="collapsed")
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

st.markdown("---")

# ── Nav buttons (Top Header) ──────────────────────────────────────────────
# Calculate number of pages and create columns
page_names = list(PAGES.keys())
# Break into 2 rows if too many pages, or just 1 row. 9 pages is a lot for 1 row, let's use columns
cols = st.columns(len(page_names))
for i, page_name in enumerate(page_names):
    is_active = st.session_state.page == page_name
    with cols[i]:
        if st.button(
            page_name,
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state.page = page_name
            st.rerun()

st.markdown("---")

# ── Data source upload (default mode only) ────────────────────────────────
if st.session_state.data_mode == "default" and st.session_state.page != "🏠  Home":
    with st.expander("⚙️ Data Settings & Custom Upload"):
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

# ═══════════════════════════════════════════════════════════════════════════════
# RENDER CONTENT
# ═══════════════════════════════════════════════════════════════════════════════


# ── Resolve which fund dict to pass to the current page ──────────────────────
with st.spinner("Loading fund data…"):
    if st.session_state.data_mode == "live" and st.session_state.live_funds_data:
        # Live mode: use data fetched from mfapi.in via FastAPI backend
        all_funds = st.session_state.live_funds_data
    else:
        # Default mode (or live mode before any data is loaded)
        all_funds = load_all_funds(st.session_state.uploaded_bytes)

# For the New Analysis page, all_funds being empty is fine (the page handles it)
if not all_funds and st.session_state.page != "🔍  New Analysis":
    st.error("⚠️ Could not load any fund data.")
    st.stop()

# RENDER SELECTED PAGE
current_page = PAGES[st.session_state.page]
current_page.render(all_funds)


