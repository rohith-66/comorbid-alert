# 🫀 ComorbidAlert

**ML pipeline forecasting diabetes-cardiac risk across all 3,144 US counties · WAPE 0.46% · Live Streamlit dashboard**

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-comorbid--alert.streamlit.app-00B4B4?style=flat-square&logo=streamlit)](https://comorbid-alert.streamlit.app)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)](https://python.org)
[![Data: CDC PLACES](https://img.shields.io/badge/Data-CDC%20PLACES%202024-red?style=flat-square)](https://places.cdc.gov)

---

## What is this?

ComorbidAlert forecasts which US counties are on a worsening trajectory toward diabetes-cardiac comorbidity — the co-occurrence of diabetes and cardiovascular disease, which is the leading driver of preventable mortality in the US.

The system goes beyond current risk scores. A county can look "Moderate" today while silently accelerating toward crisis. ComorbidAlert catches that — tracking **direction**, not just level — and flags counties 2–3 years before they cross critical thresholds.

**Live dashboard → [comorbid-alert.streamlit.app](https://comorbid-alert.streamlit.app)**

---

## Key Findings

1. **Diabetes is the engine.** Diabetes prevalence is 3.5× more predictive than the next strongest risk factor — driving cardiac risk more than poverty, obesity, and insurance gaps combined.

2. **Two completely different crises.** The 33 Critical counties split into two archetypes — Deep South Black Belt counties (structural poverty + clinical burden) and Native American reservation counties (geographic isolation + federal healthcare gaps) — each requiring entirely different interventions.

3. **The Great Plains are the blind spot.** Nebraska, Iowa, and South Dakota look Moderate today but are accelerating faster than almost anywhere in the country. Novel finding not documented in prior public health literature.

4. **994 suburban counties are hiding a time bomb.** Low current scores but rapidly rising obesity and stagnant healthcare access — over 150 projected to cross into High risk by 2027.

5. **The intervention window is closing.** 102 Critical counties in 2025 → 233 by 2027. Today's Watch alerts are tomorrow's Critical counties.

---

## Pipeline

```
CDC PLACES 2024 + Census ACS 5-Year + BRFSS 2023
                    ↓
           3,144 US Counties
                    ↓
     3-Layer Comorbidity Index
     ├── L1: Clinical burden (diabetes, CHD, obesity, hypertension)
     ├── L2: Social vulnerability (poverty, uninsured, access)
     └── L3: Trajectory (rate of change vs baseline)
                    ↓
     Risk Tiers: Critical (33) · High (381) · Moderate (2,164) · Low (566)
                    ↓
     Forecasting 2025–2027
     ├── Prophet baseline       WAPE: 5.07%
     ├── LightGBM               WAPE: 0.96%  (wins 2,101 counties)
     └── Weighted Ensemble      WAPE: 0.46%  ✓
                    ↓
     Early Warning Alerts: 830 total
     ├── Critical: 29  (already Critical, still worsening)
     ├── Warning:  129 (High tier, crossing Critical by 2027)
     └── Watch:    672 (Moderate, accelerating toward High)
```

---

## Dashboard Pages

### 🗺️ Map
US choropleth colored by comorbidity index. Toggle alert overlays (Critical / Warning / Watch). Switch between 2023 observed and 2025 / 2026 / 2027 forecast views. Click any county to open its risk profile.

### 🔍 County Drill-Down
Full county profile — risk tier badge, comorbidity index, L1/L2/L3 layer breakdown, forecast trajectory with all three model lines and confidence intervals, alert status with plain-English reason, and SHAP top-5 risk drivers.

### 🚨 Alerts
Sortable table of all 830 alerts. Filter by tier and state. Great Plains cluster highlighted as a novel finding. Alert distribution by state.

### 💡 Insights
5 key findings written in plain English for public health officials — no data science jargon. Each finding has a supporting chart.

---

## Project Structure

```
comorbid-alert/
├── comorbid_dashboard/         # Streamlit dashboard
│   ├── app.py                  # Entry point + global CSS
│   ├── data_loader.py          # All S3 reads with st.cache_data
│   ├── requirements.txt
│   ├── .streamlit/
│   │   └── config.toml         # Dark navy theme
│   └── _pages/
│       ├── page_map.py         # Page 1: Choropleth map
│       ├── page_county.py      # Page 2: County drill-down
│       ├── page_alerts.py      # Page 3: Alerts table
│       └── page_insights.py    # Page 4: Key findings
├── src/                        # Data pipeline
│   ├── ingest/                 # CDC PLACES, Census ACS, BRFSS ingestors
│   ├── transform/              # Cleaning, joining, scoring
│   ├── storage/                # S3 writer (versioned Parquet)
│   └── aws_session.py
├── eda/                        # Exploratory analysis scripts
├── forecasting/                # Prophet, LightGBM, ensemble, alerting
├── tests/                      # Pipeline tests (14 passing)
└── pipeline.py                 # Full pipeline runner
```

---

## Stack

| Layer | Tools |
|-------|-------|
| Data ingestion | CDC PLACES API, Census ACS API, BRFSS |
| Storage | AWS S3 (versioned Parquet via PyArrow) |
| Scoring | pandas, numpy |
| Forecasting | Prophet, LightGBM, weighted ensemble |
| Explainability | SHAP |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Community Cloud |

---

## Running Locally

```bash
git clone https://github.com/rohith-66/comorbid-alert.git
cd comorbid-alert

python -m venv .venv
source .venv/bin/activate
pip install -r comorbid_dashboard/requirements.txt

export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"

python -m streamlit run comorbid_dashboard/app.py
```

---

## Data Sources

- **CDC PLACES 2024** — county-level health outcomes (diabetes, CHD, obesity, hypertension prevalence)
- **Census ACS 5-Year** — social determinants (poverty rate, uninsured rate, median income)
- **BRFSS 2023** — behavioral risk factors
- **Coverage:** 3,144 US counties · Forecast horizon: 2025–2027

*Note: Oglala Lakota County (FIPS 46113) is missing from CDC PLACES 2024 — a documented data gap for reservation counties.*

---

## Model Performance

| Model | Median WAPE | Counties Won |
|-------|------------|-------------|
| Prophet | 5.07% | 823 |
| LightGBM | 0.96% | 2,101 |
| Weighted Ensemble | **0.46%** | — |

Ensemble weights are per-county, based on inverse holdout WAPE — counties where LightGBM performed better get higher LightGBM weight, and vice versa.
