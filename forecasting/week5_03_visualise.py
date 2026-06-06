"""
ComorbidAlert — Week 5, Step 3: Validation & Visualisation
==========================================================
Produces four Plotly HTML outputs saved to S3.

Outputs → s3://comorbid-alert-data/comorbid_alert/week5/
  alert_map.html
  critical_trajectories.html
  watch_trajectories.html
  wape_comparison.html
"""

import io, sys, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aws_session import get_s3_client

warnings.filterwarnings("ignore")

BUCKET = "comorbid-alert-data"
s3     = get_s3_client()

FORECAST_YEARS = [2025, 2026, 2027]
ALERT_COLORS   = {"CRITICAL": "#d62728", "WARNING": "#ff7f0e", "WATCH": "#f7c244"}

def s3_read_parquet(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def s3_read_csv(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def s3_write_html(fig, key):
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    s3.put_object(Bucket=BUCKET, Key=key, Body=html.encode("utf-8"),
                  ContentType="text/html")
    print(f"  ✓ s3://{BUCKET}/{key}")

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading data …")
alerts  = s3_read_csv("comorbid_alert/week5/alert_log.csv")
alerts["fips"] = alerts["fips"].astype(str).str.zfill(5)

fc      = s3_read_parquet("comorbid_alert/week5/ensemble_forecasts.parquet")
fc["fips"] = fc["fips"].astype(str).str.zfill(5)

panel   = s3_read_parquet("comorbid_alert/panel/comorbid_panel.parquet")
panel["fips"] = panel["fips"].astype(str).str.zfill(5)

weights = s3_read_parquet("comorbid_alert/week5/ensemble_weights.parquet")
weights["fips"] = weights["fips"].astype(str).str.zfill(5)

# ── 1. Alert choropleth ───────────────────────────────────────────────────────

print("Building alert choropleth …")
level_num = {"WATCH": 1, "WARNING": 2, "CRITICAL": 3}
alerts["level_num"] = alerts["alert_level"].map(level_num)

fig_map = go.Figure(go.Choropleth(
    geojson   = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    locations = alerts["fips"],
    z         = alerts["level_num"],
    colorscale= [[0.0, "#f7c244"], [0.5, "#ff7f0e"], [1.0, "#d62728"]],
    zmin=1, zmax=3,
    colorbar  = dict(title="Alert Level",
                     tickvals=[1,2,3], ticktext=["WATCH","WARNING","CRITICAL"]),
    hovertemplate=(
        "<b>%{customdata[0]}, %{customdata[1]}</b><br>"
        "Alert: <b>%{customdata[2]}</b><br>"
        "Current Tier: %{customdata[3]}<br>"
        "Comorbidity Index 2024: %{customdata[4]:.3f}<br>"
        "Ensemble 2027: %{customdata[5]:.3f}<br>"
        "Forecast Δ: %{customdata[6]:+.1f}%<br>"
        "<extra></extra>"
    ),
    customdata=alerts[["county_name","stateabbr","alert_level","current_tier",
                        "comorbid_index_2024","ens_2027","forecast_pct_chg"]].values,
))
fig_map.update_layout(
    title_text="ComorbidAlert — Early Warning Alert Map (2025–2027 Ensemble)",
    title_x=0.5, geo_scope="usa",
    margin=dict(l=0, r=0, t=50, b=0), height=580,
    annotations=[dict(
        text=(f"CRITICAL: {(alerts['alert_level']=='CRITICAL').sum()} | "
              f"WARNING: {(alerts['alert_level']=='WARNING').sum()} | "
              f"WATCH: {(alerts['alert_level']=='WATCH').sum()}"),
        xref="paper", yref="paper", x=0.5, y=-0.02,
        showarrow=False, font=dict(size=13)
    )]
)
s3_write_html(fig_map, "comorbid_alert/week5/alert_map.html")

# ── 2. Trajectory subplots ────────────────────────────────────────────────────

def trajectory_fig(top_df, title, color):
    n = min(5, len(top_df))
    if n == 0:
        return None
    top_df = top_df.head(n).reset_index(drop=True)

    subplot_titles = []
    for _, r in top_df.iterrows():
        subplot_titles.append(f"{r['county_name']}, {r['stateabbr']}<br>({r['fips']})")

    fig = make_subplots(rows=1, cols=n, subplot_titles=subplot_titles,
                        shared_yaxes=True)

    for i, (_, row) in enumerate(top_df.iterrows(), 1):
        fips = row["fips"]

        # Historical from panel
        hist = (panel[panel["fips"] == fips]
                .sort_values("release_year")[["release_year","comorbid_index"]])
        if not hist.empty:
            fig.add_trace(go.Scatter(
                x=hist["release_year"], y=hist["comorbid_index"],
                mode="lines+markers", name="Historical",
                line=dict(color="#aaaaaa", dash="dot"),
                marker=dict(size=5),
                showlegend=(i == 1)
            ), row=1, col=i)

        # Ensemble forecast
        ens = (fc[fc["fips"] == fips]
               .sort_values("forecast_year")[["forecast_year","ensemble_forecast"]])
        if not ens.empty:
            # Connect from last historical point
            if not hist.empty:
                last_hist = hist.iloc[-1]
                connect_x = [last_hist["release_year"]] + ens["forecast_year"].tolist()
                connect_y = [last_hist["comorbid_index"]] + ens["ensemble_forecast"].tolist()
            else:
                connect_x = ens["forecast_year"].tolist()
                connect_y = ens["ensemble_forecast"].tolist()

            fig.add_trace(go.Scatter(
                x=connect_x, y=connect_y,
                mode="lines+markers", name="Ensemble Forecast",
                line=dict(color=color, width=2.5),
                marker=dict(size=7),
                showlegend=(i == 1)
            ), row=1, col=i)

        # Threshold line
        fig.add_hline(y=0.65, line_dash="dash", line_color="#d62728",
                      line_width=1, opacity=0.6, row=1, col=i)

    fig.update_layout(
        title_text=title, title_x=0.5,
        height=400, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center")
    )
    return fig

print("Building trajectory plots …")

critical_top = s3_read_csv("comorbid_alert/week5/critical_top10.csv")
critical_top["fips"] = critical_top["fips"].astype(str).str.zfill(5)
fig_crit = trajectory_fig(
    critical_top,
    "Top 5 CRITICAL Counties — Worsening Despite Critical Status",
    "#d62728"
)
if fig_crit:
    s3_write_html(fig_crit, "comorbid_alert/week5/critical_trajectories.html")

watch_top = s3_read_csv("comorbid_alert/week5/watch_top10.csv")
watch_top["fips"] = watch_top["fips"].astype(str).str.zfill(5)
fig_watch = trajectory_fig(
    watch_top,
    "Top 5 WATCH Counties — Accelerating Moderate Tier (Early Warning)",
    "#f7c244"
)
if fig_watch:
    s3_write_html(fig_watch, "comorbid_alert/week5/watch_trajectories.html")

# ── 3. Model comparison: WAPE proxy by tier ───────────────────────────────────

print("Building WAPE comparison chart …")

# Use weights file — it has per-county err_prophet & err_lgbm
wt = weights.copy()
wt = wt.merge(
    fc[fc["forecast_year"]==2025][["fips","ensemble_forecast"]],
    on="fips", how="left"
)
wt["err_ensemble"] = (
    np.abs(wt["ensemble_forecast"] - wt["actual_2024"])
    / wt["actual_2024"].clip(lower=1e-6) * 100
)

tier_order = ["Critical","High","Moderate","Low"]
palette    = {"Prophet":"#1f77b4","LightGBM":"#2ca02c","Ensemble":"#9467bd"}

fig_wape = go.Figure()
for model, col in [("Prophet","err_prophet"),("LightGBM","err_lgbm"),("Ensemble","err_ensemble")]:
    for tier in tier_order:
        subset = wt[wt["current_tier"] == tier][col].dropna()
        subset = subset[subset < subset.quantile(0.99)]
        if subset.empty:
            continue
        fig_wape.add_trace(go.Box(
            y=subset,
            name=f"{model}<br>{tier}",
            boxmean=True,
            marker_color=palette[model],
            legendgroup=model,
            showlegend=(tier == "Critical"),
        ))

fig_wape.update_layout(
    title_text="Proxy WAPE by Model and Risk Tier (2025 forecast vs 2024 actual)",
    title_x=0.5,
    yaxis_title="Absolute % Error",
    template="plotly_white",
    height=500,
    boxmode="group",
)
s3_write_html(fig_wape, "comorbid_alert/week5/wape_comparison.html")

print("\nWeek 5 Step 3 complete.")
print("\n   Outputs:")
print("     s3://comorbid-alert-data/comorbid_alert/week5/alert_map.html")
print("     s3://comorbid-alert-data/comorbid_alert/week5/critical_trajectories.html")
print("     s3://comorbid-alert-data/comorbid_alert/week5/watch_trajectories.html")
print("     s3://comorbid-alert-data/comorbid_alert/week5/wape_comparison.html")