import streamlit as st

def render(all_funds: dict):
    # Description of what the website does
    st.markdown("""
        <div style="text-align: center; padding: 4rem 0;">
            <h1 style="font-size: 3.5rem; color: var(--primary-color); font-weight: 800; margin-bottom: 1rem;">Welcome to FundScope</h1>
            <p style="font-size: 1.25rem; color: var(--text-muted); max-width: 800px; margin: 0 auto; line-height: 1.8;">
                FundScope is a professional Mutual Fund Analytics Dashboard. 
                It provides deep insights into your mutual fund portfolio, 
                offering features like performance tracking, risk analysis, 
                predictive modeling, and backtesting. Whether you are analyzing 
                historical data or evaluating live funds, FundScope delivers 
                the tools you need to make informed investment decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Adding some space for scrolling effect
    st.markdown("<br>" * 15, unsafe_allow_html=True)

    # Two options
    st.markdown("<h3 style='text-align: center; margin-bottom: 2rem;'>Choose an Option to Get Started</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h4 style="color: var(--text-main);">Explore Sample Data</h4>
            <p style="color: var(--text-muted); font-size: 0.95rem;">View an analysis based on our pre-loaded sample report (data.xlsx) to understand the platform's capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("See Sample Report", use_container_width=True, type="primary"):
            st.session_state.data_mode = "default"
            st.session_state.page = "📊  Overview"
            st.rerun()

    with col3:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h4 style="color: var(--text-main);">Try Your Own Analysis</h4>
            <p style="color: var(--text-muted); font-size: 0.95rem;">Search for live mutual funds, specify a date range, and perform a real-time customized analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Try Your Own Analysis", use_container_width=True, type="primary"):
            st.session_state.data_mode = "live"
            st.session_state.page = "🔍  New Analysis"
            st.rerun()
