"""
app_pages/new_analysis.py — Live fund search & data loading via mfapi.in.
Users search for any Indian mutual fund, pick a date range, and load
NAV data that is fed to all existing analysis pages.
"""
import datetime
import streamlit as st

from modules.data_loader import (
    search_schemes_api,
    load_funds_from_api,
    check_backend_health,
)

# ── Popular funds shown as quick-pick shortcuts ───────────────────────────────
POPULAR_FUNDS = [
    {"schemeCode": 122639, "schemeName": "Parag Parikh Flexi Cap Fund - Direct Plan - Growth"},
    {"schemeCode": 120503, "schemeName": "HDFC Infrastructure Fund - Direct Plan - Growth"},
    {"schemeCode": 125497, "schemeName": "HDFC Top 100 Fund - Direct Plan - Growth"},
    {"schemeCode": 118989, "schemeName": "Mirae Asset Large Cap Fund - Direct Plan - Growth"},
    {"schemeCode": 120716, "schemeName": "SBI Blue Chip Fund - Direct Plan - Growth"},
    {"schemeCode": 119598, "schemeName": "SBI Banking & Financial Services Fund - Direct Plan - Growth"},
]

MAX_FUNDS = 10
DEFAULT_YEARS_BACK = 3


def render(all_funds: dict):  # noqa — all_funds not used here but signature must match
    st.markdown('<p class="page-title">🔍 New Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Search any Indian mutual fund, pick a date range, '
        'and run the full FundScope analysis on live data.</p>',
        unsafe_allow_html=True,
    )

    # ── Backend health check ──────────────────────────────────────────────────
    backend_ok = check_backend_health()
    if not backend_ok:
        st.error(
            "⚠️ **FundScope backend is not running.**\n\n"
            "Start it in a separate terminal:\n"
            "```bash\n"
            "cd backend\n"
            "pip install -r requirements.txt\n"
            "uvicorn main:app --reload --port 8000\n"
            "```\n"
            "Then refresh this page."
        )
        _show_current_live_funds()
        return

    st.success("✅ Backend connected — `http://localhost:8000`")

    # ── Step 1: Search & Select ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 1 — Search & Select Funds")
    st.caption(f"You can select up to {MAX_FUNDS} funds per analysis.")

    # Quick-pick popular funds
    with st.expander("⚡ Quick-pick popular funds", expanded=False):
        st.caption("Click any fund to pre-fill the search and add it to your selection.")
        cols = st.columns(3)
        for i, fund in enumerate(POPULAR_FUNDS):
            with cols[i % 3]:
                short_name = fund["schemeName"].replace(" - Direct Plan - Growth", "").replace(" Fund", "")
                if st.button(short_name, key=f"popular_{fund['schemeCode']}", use_container_width=True):
                    _add_to_selection(fund)

    # Search bar
    search_query = st.text_input(
        "Search by fund name",
        placeholder="e.g. HDFC Midcap, Axis Small Cap, Nippon India...",
        key="live_search_query",
    )

    search_results: list[dict] = []
    if search_query and len(search_query.strip()) >= 3:
        with st.spinner("Searching…"):
            search_results = search_schemes_api(search_query)

        if search_results:
            # Build label → fund dict
            options_map = {
                f"{r['schemeName']}  [{r['schemeCode']}]": r
                for r in search_results[:100]   # cap display at 100 results
            }
            chosen_labels = st.multiselect(
                f"Select from {len(search_results)} result(s)",
                list(options_map.keys()),
                key="live_search_results",
            )
            for label in chosen_labels:
                _add_to_selection(options_map[label])
        elif search_query:
            st.info("No results. Try a different keyword.")

    # Current selection display
    selected: list[dict] = st.session_state.get("live_selected_funds", [])
    _render_selection_box(selected)

    # ── Step 2: Date Range ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 2 — Select Date Range")
    st.caption(
        "Minimum 3 months recommended for rolling metrics. "
        "⚠️ International funds are not available via mfapi.in (Indian MFs only)."
    )

    today = datetime.date.today()
    default_start = today.replace(year=today.year - DEFAULT_YEARS_BACK)

    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            min_value=datetime.date(1995, 1, 1),
            max_value=today,
            key="live_start_date",
        )
    with col_e:
        end_date = st.date_input(
            "End Date",
            value=today,
            min_value=datetime.date(1995, 1, 2),
            max_value=today,
            key="live_end_date",
        )

    if end_date <= start_date:
        st.warning("End date must be after start date.")
        return

    date_span_days = (end_date - start_date).days
    if date_span_days < 30:
        st.warning(f"Date range is only {date_span_days} days — analysis needs at least 30 days of data.")

    # ── Step 3: Load ─────────────────────────────────────────────────────────
    st.markdown("---")

    if not selected:
        st.info("☝️ Search and select at least one fund above, then click **Load Data**.")
    else:
        n = len(selected)
        st.markdown(
            f"**{n} fund(s) selected** — "
            f"`{start_date}` → `{end_date}` "
            f"({date_span_days} days)"
        )

        col_load, col_clear = st.columns([3, 1])
        with col_load:
            load_clicked = st.button(
                f"📥 Load Data & Run Analysis ({n} fund{'s' if n > 1 else ''})",
                type="primary",
                use_container_width=True,
                disabled=(date_span_days < 7),
            )
        with col_clear:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.live_selected_funds = []
                st.rerun()

        if load_clicked:
            payload = [
                {"schemeCode": f["schemeCode"], "schemeName": f["schemeName"]}
                for f in selected
            ]
            with st.spinner(
                f"Fetching NAV data for {n} fund(s) in parallel… "
                f"(date range: {start_date} → {end_date})"
            ):
                live_data = load_funds_from_api(payload, str(start_date), str(end_date))

            if not live_data:
                st.error("❌ No data returned. Check backend logs or try different funds / date range.")
            else:
                st.session_state.live_funds_data = live_data
                st.session_state.data_mode = "live"
                st.session_state.live_date_range = (str(start_date), str(end_date))

                loaded_n = len(live_data)
                st.success(
                    f"✅ Successfully loaded **{loaded_n}** fund(s). "
                    f"Navigate using the sidebar to run analysis."
                )

                # Show summary of what was loaded
                _show_current_live_funds()

    # ── Show already-loaded data ──────────────────────────────────────────────
    if st.session_state.get("live_funds_data") and not selected:
        _show_current_live_funds()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_to_selection(fund: dict) -> None:
    """Add a fund to the live_selected_funds list (deduplicated, max MAX_FUNDS)."""
    if "live_selected_funds" not in st.session_state:
        st.session_state.live_selected_funds = []

    existing_codes = {f["schemeCode"] for f in st.session_state.live_selected_funds}
    if fund["schemeCode"] in existing_codes:
        return  # already added
    if len(st.session_state.live_selected_funds) >= MAX_FUNDS:
        st.warning(f"Maximum {MAX_FUNDS} funds reached. Remove one before adding another.")
        return

    st.session_state.live_selected_funds.append({
        "schemeCode": fund["schemeCode"],
        "schemeName": fund["schemeName"],
    })
    st.rerun()


def _render_selection_box(selected: list[dict]) -> None:
    """Show the current selection with remove buttons."""
    if not selected:
        return

    st.markdown(f"**Selected ({len(selected)}/{MAX_FUNDS}):**")
    for i, fund in enumerate(selected):
        col_name, col_rm = st.columns([9, 1])
        with col_name:
            short = fund["schemeName"][:80] + ("…" if len(fund["schemeName"]) > 80 else "")
            st.markdown(f"`{fund['schemeCode']}`  {short}")
        with col_rm:
            if st.button("✕", key=f"rm_{fund['schemeCode']}_{i}"):
                st.session_state.live_selected_funds = [
                    f for f in selected if f["schemeCode"] != fund["schemeCode"]
                ]
                st.rerun()


def _show_current_live_funds() -> None:
    """Render a summary table of currently loaded live funds."""
    live = st.session_state.get("live_funds_data", {})
    if not live:
        return

    st.markdown("---")
    st.markdown("### Currently Loaded Live Funds")
    date_range = st.session_state.get("live_date_range", ("—", "—"))
    st.caption(f"Date range: `{date_range[0]}` → `{date_range[1]}`")

    for name, df in live.items():
        dr_start = df["ds"].min().date() if not df.empty else "—"
        dr_end   = df["ds"].max().date() if not df.empty else "—"
        latest_nav = df["y"].iloc[-1] if not df.empty else 0
        records    = len(df)
        st.markdown(
            f"- **{name}** — {records:,} records &nbsp;|&nbsp; "
            f"`{dr_start}` → `{dr_end}` &nbsp;|&nbsp; "
            f"Latest NAV: ₹`{latest_nav:.2f}`"
        )

    st.info(
        "👈 Use the **sidebar** to switch between Overview, Analysis, Predictions, "
        "Risk Analysis, Backtesting, or Simulation. All pages will use this live data."
    )
