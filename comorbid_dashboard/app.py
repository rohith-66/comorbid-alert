"""
ComorbidAlert Dashboard — Week 6
US County-Level Diabetes-Cardiac Comorbidity Forecasting
"""

import streamlit as st

st.set_page_config(
    page_title="ComorbidAlert",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --navy:    #0B1426;
    --navy2:   #111D35;
    --navy3:   #162040;
    --teal:    #00B4B4;
    --teal2:   #00D4D4;
    --teal-dk: #007A7A;
    --red:     #FF4D4D;
    --orange:  #FF8C00;
    --yellow:  #FFD700;
    --green:   #28E08A;
    --text:    #E8F0FE;
    --text2:   #9BB0CC;
    --border:  rgba(0,180,180,0.18);
    --card-bg: rgba(17,29,53,0.85);
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text);
}

/* Remove default Streamlit padding */
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stRadio > label {
    color: var(--text2) !important;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text2);
    font-size: 0.8rem;
}

/* Nav radio buttons styled as menu items */
[data-testid="stSidebar"] .stRadio [role="radio"] {
    padding: 8px 12px !important;
    border-radius: 6px;
    margin-bottom: 2px;
}

/* Metric cards */
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    backdrop-filter: blur(8px);
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--teal);
}
.metric-card .label {
    font-size: 0.75rem;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Tier badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-critical { background: rgba(255,77,77,0.18); color: #FF6B6B; border: 1px solid rgba(255,77,77,0.4); }
.badge-high     { background: rgba(255,140,0,0.15); color: #FFA040; border: 1px solid rgba(255,140,0,0.4); }
.badge-moderate { background: rgba(255,215,0,0.12); color: #FFD700; border: 1px solid rgba(255,215,0,0.35); }
.badge-low      { background: rgba(40,224,138,0.12); color: #28E08A; border: 1px solid rgba(40,224,138,0.35); }

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--teal);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 12px;
}

/* Streamlit selectbox, multiselect */
.stSelectbox > div, .stMultiSelect > div {
    background: var(--navy2) !important;
    border-color: var(--border) !important;
}

/* Plotly chart backgrounds transparent */
.js-plotly-plot { background: transparent !important; }

/* Alert callout */
.alert-callout {
    background: linear-gradient(135deg, rgba(255,215,0,0.08), rgba(255,140,0,0.08));
    border: 1px solid rgba(255,215,0,0.3);
    border-left: 4px solid var(--yellow);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}

/* Page title */
.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--teal);
    margin-bottom: 0.25rem;
}
.page-subtitle {
    font-size: 0.85rem;
    color: var(--text2);
    margin-bottom: 1.5rem;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem; font-weight:600; color:#00B4B4;">
            🫀 ComorbidAlert
        </div>
        <div style="font-size:0.72rem; color:#9BB0CC; margin-top:4px; letter-spacing:0.05em;">
            US County Comorbidity Forecasting
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🗺️  Map", "🔍  County Drill-Down", "🚨  Alerts", "💡  Insights"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#9BB0CC; line-height:1.6;">
        <div style="color:#00B4B4; font-weight:600; margin-bottom:6px;">DATA SOURCES</div>
        CDC PLACES 2024<br>
        Census ACS 5-Year<br>
        BRFSS 2023<br>
        <br>
        <div style="color:#00B4B4; font-weight:600; margin-bottom:6px;">COVERAGE</div>
        3,144 US counties<br>
        Forecast: 2025–2027<br>
        <br>
        <div style="color:#00B4B4; font-weight:600; margin-bottom:6px;">MODELS</div>
        Prophet · LightGBM<br>
        Weighted Ensemble<br>
        WAPE: 0.46%
    </div>
    """, unsafe_allow_html=True)

# ── Page routing ─────────────────────────────────────────────────────────────
if "🗺️" in page:
    import _pages.page_map as map_page
    map_page.render()
elif "🔍" in page:
    import _pages.page_county as drilldown_page
    drilldown_page.render()
elif "🚨" in page:
    import _pages.page_alerts as alerts_page
    alerts_page.render()
elif "💡" in page:
    import _pages.page_insights as insights_page
    insights_page.render()