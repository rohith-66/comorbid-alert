# ComorbidAlert Dashboard — Week 6

Streamlit dashboard for the US county-level diabetes–cardiac comorbidity forecasting system.

## Structure

```
comorbid_dashboard/
├── app.py                     ← Streamlit entry point + global CSS
├── requirements.txt
├── .streamlit/
│   └── config.toml            ← Dark theme + server config
├── data/
│   └── loader.py              ← S3 data access layer (all @st.cache_data)
├── ui/
│   └── sidebar.py             ← Navigation sidebar + routing
└── pages/
    ├── page_map.py            ← National choropleth map
    ├── page_county.py         ← County drill-down panel
    ├── page_alerts.py         ← Alert table + filters
    └── page_insights.py       ← 5 key insights with charts
```

## Setup

```bash
# From project root (comorbid_alret/)
cd comorbid_dashboard

# Install deps into existing venv
../.venv/bin/pip install -r requirements.txt

# Make sure AWS creds are set (same .env as main pipeline)
# COMORBID_S3_BUCKET=comorbid-alert-data must be set

# Run
../.venv/bin/streamlit run app.py
```

Or from the project root:

```bash
source .venv/bin/activate
pip install -r comorbid_dashboard/requirements.txt
streamlit run comorbid_dashboard/app.py
```

## S3 Data Expected

The dashboard reads these keys from `comorbid-alert-data`:

| Key | Used for |
|-----|----------|
| `processed/comorbidity_index.parquet` | Base index, L1/L2/L3, risk tiers |
| `models/ensemble_forecasts.parquet` | Year-level ensemble forecasts 2025–2027 |
| `models/prophet_forecasts.parquet` | Prophet per-county with CI (optional) |
| `models/lgbm_forecasts.parquet` | LightGBM per-county (optional) |
| `models/shap_values.parquet` | SHAP feature importance per county (optional) |
| `alerts/alert_log.csv` | 830 alerts from Week 5 |

Missing optional files degrade gracefully — the chart simply won't appear.

## Pages

### 🗺️ National Map
- Choropleth colored by comorbidity index (0–1 scale, dark blue → deep red)
- Toggle alert overlays on/off
- Switch between 2023 observed / 2025 / 2026 / 2027 forecast views
- County selector → navigates to drill-down

### 🏥 County Detail
- Risk tier badge + comorbidity index (monospace, tier-colored)
- L1 / L2 / L3 bar breakdown with plain-English labels
- Forecast chart: Prophet (dotted) + LightGBM (dashed) + Ensemble (solid) + CI band
- Alert status + reason
- SHAP top-5 features (red = increases risk, blue = reduces risk)

### 🚨 Alerts
- Full 830-alert table, sortable by tier / slope / change % / state
- Filter by tier and/or state
- Great Plains cluster highlighted as novel finding
- Bar chart: top 20 states by alert count (stacked by tier)

### 💡 Insights
- 5 finding cards in plain English (no jargon)
- Each card has an embedded supporting chart:
  1. L1 vs L2 by risk tier (grouped bar)
  2. Two-archetype comparison (Deep South vs Native American)
  3. SE coast cluster trajectory vs national average
  4. Suburban obesity trend by county type
  5. Critical county growth 2023→2027 with Great Plains highlighted

## Optional: streamlit-plotly-events

For click-to-drill-down on the choropleth:

```bash
pip install streamlit-plotly-events
```

If not installed, the choropleth renders normally and the county selector handles navigation.

## Design Principles

- IBM Plex Sans / IBM Plex Mono — readable, authoritative, clinical
- Dark background (#0f1117) — reduces eye strain for long analysis sessions
- Tier colors consistent everywhere: Critical=red, High=orange, Moderate=amber, Low=green
- Every chart is `displayModeBar: False` — clean, no toolbar clutter
- All S3 reads cached for 1 hour with `@st.cache_data(ttl=3600)`
