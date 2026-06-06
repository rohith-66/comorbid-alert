"""
forecast_prophet.py
===================
Week 3/4 — Prophet baseline forecasting on comorbid_index per county.

Temporal cross-validation setup (no data leakage):
  - Train: years 2021, 2022, 2023
  - Test:  year 2024
  - Forecast horizon: 3 years (2025, 2026, 2027)

Evaluation metrics: MAE, RMSE, WAPE

Usage:
    cd ~/Documents/comorbid_alret
    source .venv/bin/activate
    pip install prophet
    python forecasting/forecast_prophet.py

Output:
    forecasting/outputs/forecast_results.parquet   ← all county forecasts
    forecasting/outputs/forecast_eval.csv          ← per-county MAE/RMSE/WAPE
    forecasting/outputs/forecast_summary.csv       ← aggregate metrics
    forecasting/outputs/forecast_plots/            ← HTML charts for top counties
"""
from dotenv import load_dotenv
load_dotenv()
import io
import logging
import warnings
from pathlib import Path

import boto3
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

warnings.filterwarnings("ignore")  # suppress Prophet/Stan noise
from prophet import Prophet

load_dotenv("/Users/rohithsrinivasa/Documents/comorbid_alret/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR   = Path(__file__).parent / "outputs"
PLOTS_DIR    = OUTPUT_DIR / "forecast_plots"
S3_BUCKET    = "comorbid-alert-data"
PANEL_KEY    = "comorbid_alert/panel/comorbid_panel.parquet"
FORECAST_KEY = "comorbid_alert/panel/forecast_results.parquet"

TRAIN_YEARS    = [2021, 2022, 2023]
TEST_YEAR      = 2024
HORIZON_YEARS  = [2025, 2026, 2027]
MIN_TRAIN_PTS  = 3   # need at least 3 points to fit Prophet

# Counties to plot individually (Critical + interesting cases)
HIGHLIGHT_FIPS = {
    "46113": "Oglala Lakota, SD",
    "28027": "Coahoma, MS",
    "01085": "Lowndes, AL",
    "48427": "Starr, TX",
    "13153": "Jeff Davis, GA",
}


# ── Load panel ────────────────────────────────────────────────────────────────
def load_panel() -> pd.DataFrame:
    log.info("Loading panel from S3...")
    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=PANEL_KEY)
    df  = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    log.info("  Panel: %d rows | %d counties | years: %s",
             len(df),
             df["fips"].nunique(),
             sorted(df["release_year"].unique().tolist()))
    return df


# ── Metrics ───────────────────────────────────────────────────────────────────
def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def wape(actual, predicted):
    """Weighted Absolute Percentage Error — robust to near-zero actuals."""
    return np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual))


# ── Prophet per county ────────────────────────────────────────────────────────
def fit_prophet_county(ts: pd.DataFrame) -> dict:
    """
    Fit Prophet on TRAIN_YEARS, evaluate on TEST_YEAR,
    forecast HORIZON_YEARS.
    """
    ts = ts.sort_values("release_year").copy()
    # Jan 1 — avoids YE freq boundary issues
    ts["ds"] = pd.to_datetime(ts["release_year"].astype(str) + "-01-01")
    ts["y"]  = ts["comorbid_index"].clip(0, 1)

    train = ts[ts["release_year"].isin(TRAIN_YEARS)]
    test  = ts[ts["release_year"] == TEST_YEAR]

    if len(train) < MIN_TRAIN_PTS:
        return None

    # ── Fit ───────────────────────────────────────────────────────────────────
    # changepoint_prior_scale=0.05 — conservative for short series (n=3)
    # High values cause wild extrapolation with few points
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        interval_width=0.80,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train[["ds", "y"]])

    # ── Build all future dates explicitly — avoids freq alignment bugs ─────────
    all_years   = TRAIN_YEARS + [TEST_YEAR] + HORIZON_YEARS
    future_all  = pd.DataFrame({
        "ds": pd.to_datetime([f"{y}-01-01" for y in all_years])
    })
    forecast_all = m.predict(future_all)
    forecast_all["year"] = forecast_all["ds"].dt.year

    # ── Evaluate on test year ─────────────────────────────────────────────────
    eval_metrics = {}
    if not test.empty:
        test_row = forecast_all[forecast_all["year"] == TEST_YEAR]
        if not test_row.empty:
            y_actual = test["y"].values
            y_pred   = test_row["yhat"].clip(0, 1).values[:len(y_actual)]
            eval_metrics = {
                "mae":  mae(y_actual, y_pred),
                "rmse": rmse(y_actual, y_pred),
                "wape": wape(y_actual, y_pred),
            }

    # ── Forecast horizon ──────────────────────────────────────────────────────
    horizon_rows = []
    for yr in HORIZON_YEARS:
        row = forecast_all[forecast_all["year"] == yr]
        if not row.empty:
            horizon_rows.append({
                "forecast_year": yr,
                "yhat":          float(row["yhat"].clip(0, 1).iloc[0]),
                "yhat_lower":    float(row["yhat_lower"].clip(0, 1).iloc[0]),
                "yhat_upper":    float(row["yhat_upper"].clip(0, 1).iloc[0]),
            })

    return {
        "eval":          eval_metrics,
        "forecast":      horizon_rows,
        "model":         m,
        "train_df":      train,
        "test_df":       test,
        "full_forecast": forecast_all,
    }


# ── Run all counties ──────────────────────────────────────────────────────────
def run_all_counties(panel: pd.DataFrame):
    fips_list = panel["fips"].unique()
    log.info("Fitting Prophet for %d counties...", len(fips_list))

    eval_rows     = []
    forecast_rows = []
    models        = {}
    failed        = 0

    for i, fips in enumerate(fips_list):
        ts = panel[panel["fips"] == fips].copy()
        meta = ts.iloc[0]

        result = fit_prophet_county(ts)
        if result is None:
            failed += 1
            continue

        county_name = meta.get("county_name", fips)
        stateabbr   = meta.get("stateabbr", "")

        # Eval row
        if result["eval"]:
            eval_rows.append({
                "fips":        fips,
                "county_name": county_name,
                "stateabbr":   stateabbr,
                **result["eval"],
            })

        # Forecast rows
        for fr in result["forecast"]:
            forecast_rows.append({
                "fips":        fips,
                "county_name": county_name,
                "stateabbr":   stateabbr,
                **fr,
            })

        # Store model for highlighted counties
        if fips in HIGHLIGHT_FIPS:
            models[fips] = result

        if (i + 1) % 500 == 0:
            log.info("  Progress: %d / %d counties", i + 1, len(fips_list))

    log.info("Done. %d counties fit | %d failed (insufficient data)",
             len(fips_list) - failed, failed)

    eval_df     = pd.DataFrame(eval_rows)
    forecast_df = pd.DataFrame(forecast_rows)
    return eval_df, forecast_df, models


# ── Aggregate metrics ─────────────────────────────────────────────────────────
def summarize_eval(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()

    summary = pd.DataFrame([{
        "metric": "MAE",
        "mean":   eval_df["mae"].mean(),
        "median": eval_df["mae"].median(),
        "p25":    eval_df["mae"].quantile(0.25),
        "p75":    eval_df["mae"].quantile(0.75),
    }, {
        "metric": "RMSE",
        "mean":   eval_df["rmse"].mean(),
        "median": eval_df["rmse"].median(),
        "p25":    eval_df["rmse"].quantile(0.25),
        "p75":    eval_df["rmse"].quantile(0.75),
    }, {
        "metric": "WAPE",
        "mean":   eval_df["wape"].mean(),
        "median": eval_df["wape"].median(),
        "p25":    eval_df["wape"].quantile(0.25),
        "p75":    eval_df["wape"].quantile(0.75),
    }]).round(4)

    log.info("\n── Forecast evaluation (test year %d) ──", TEST_YEAR)
    log.info("\n%s", summary.to_string(index=False))

    # Worst counties
    log.info("\nWorst 10 counties by MAE:")
    worst = eval_df.nlargest(10, "mae")[["county_name", "stateabbr", "mae", "rmse", "wape"]]
    log.info("\n%s", worst.to_string(index=False))

    # Best counties
    log.info("\nBest 10 counties by MAE:")
    best = eval_df.nsmallest(10, "mae")[["county_name", "stateabbr", "mae", "rmse", "wape"]]
    log.info("\n%s", best.to_string(index=False))

    return summary


# ── Plot highlighted counties ─────────────────────────────────────────────────
def plot_county_forecast(fips: str, name: str, result: dict,
                          panel: pd.DataFrame) -> go.Figure:
    ts       = panel[panel["fips"] == fips].sort_values("release_year")
    forecast = result["full_forecast"]

    fig = go.Figure()

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=ts["release_year"],
        y=ts["comorbid_index"],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=8),
    ))

    # Train/test split line
    fig.add_vline(
        x=TEST_YEAR - 0.5,
        line_dash="dash",
        line_color="gray",
        line_width=1,
        annotation_text="train | test",
        annotation_position="top",
        annotation_font_size=10,
    )

    # Prophet fitted + forecast
    fig.add_trace(go.Scatter(
        x=forecast["ds"].dt.year,
        y=forecast["yhat"].clip(0, 1),
        mode="lines",
        name="Prophet forecast",
        line=dict(color="#d62728", width=2, dash="dot"),
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"].dt.year,
                     forecast["ds"].dt.year[::-1]]),
        y=pd.concat([forecast["yhat_upper"].clip(0, 1),
                     forecast["yhat_lower"].clip(0, 1)[::-1]]),
        fill="toself",
        fillcolor="rgba(214,39,40,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90% interval",
        showlegend=True,
    ))

    # Risk tier bands
    for y_val, label, color in [
        (0.75, "Critical threshold", "rgba(214,39,40,0.15)"),
        (0.50, "High threshold",     "rgba(255,127,14,0.10)"),
    ]:
        fig.add_hline(
            y=y_val,
            line_dash="dot",
            line_color=color.replace("0.15", "0.6").replace("0.10", "0.5"),
            line_width=0.8,
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=9,
        )

    eval_m = result.get("eval", {})
    eval_str = ""
    if eval_m:
        eval_str = (f"MAE={eval_m.get('mae', 0):.4f}  "
                    f"RMSE={eval_m.get('rmse', 0):.4f}  "
                    f"WAPE={eval_m.get('wape', 0):.4f}")

    fig.update_layout(
        title=dict(
            text=f"<b>{name}</b><br>"
                 f"<sup>Prophet forecast · {eval_str}</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        xaxis=dict(title="Year", dtick=1),
        yaxis=dict(title="Comorbidity Index", range=[-0.02, 1.05]),
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="rgba(245,245,245,1)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.25),
    )
    return fig


# ── Plot metric distributions ─────────────────────────────────────────────────
def plot_eval_distributions(eval_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("MAE distribution", "RMSE distribution", "WAPE distribution"),
    )

    for i, (col, label) in enumerate([("mae","MAE"), ("rmse","RMSE"), ("wape","WAPE")], 1):
        vals = eval_df[col].dropna()
        fig.add_trace(go.Histogram(
            x=vals,
            nbinsx=50,
            name=label,
            marker_color=["#1f77b4","#ff7f0e","#2ca02c"][i-1],
            showlegend=False,
        ), row=1, col=i)
        fig.add_vline(
            x=vals.median(),
            line_dash="dash",
            line_color="red",
            line_width=1.5,
            annotation_text=f"median={vals.median():.4f}",
            annotation_font_size=9,
            row=1, col=i,
        )

    fig.update_layout(
        title=dict(
            text="<b>Prophet Forecast Error Distributions</b><br>"
                 f"<sup>Test year {TEST_YEAR} · {len(eval_df):,} counties</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=380,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


# ── Plot forecast map data ────────────────────────────────────────────────────
def plot_forecast_choropleth(forecast_df: pd.DataFrame, year: int) -> go.Figure:
    yr_df = forecast_df[forecast_df["forecast_year"] == year].copy()
    yr_df["fips_5"] = yr_df["fips"].astype(str).str.zfill(5)

    def tier(v):
        if v >= 0.75: return "Critical"
        if v >= 0.50: return "High"
        if v >= 0.25: return "Moderate"
        return "Low"

    yr_df["forecast_tier"] = yr_df["yhat"].apply(tier)
    tier_counts = yr_df["forecast_tier"].value_counts()

    COLORSCALE = [
        [0.00, "#2ca02c"],
        [0.25, "#f7b731"],
        [0.50, "#ff7f0e"],
        [0.75, "#d62728"],
        [1.00, "#7b0000"],
    ]

    hover = yr_df.apply(lambda r: (
        f"<b>{r['county_name']}, {r['stateabbr']}</b><br>"
        f"Forecast {year}: {r['yhat']:.3f}<br>"
        f"Tier: {r['forecast_tier']}<br>"
        f"90% CI: [{r['yhat_lower']:.3f}, {r['yhat_upper']:.3f}]"
    ), axis=1)

    TIER_COLORS = {"Critical":"#d62728","High":"#ff7f0e","Moderate":"#f5c400","Low":"#2ca02c"}
    counts_str = "  ".join(
        f"<span style='color:{TIER_COLORS[t]}'><b>{t}</b>: {tier_counts.get(t,0)}</span>"
        for t in ["Critical","High","Moderate","Low"]
    )

    fig = go.Figure(go.Choropleth(
        geojson=(
            "https://raw.githubusercontent.com/plotly/datasets/master/"
            "geojson-counties-fips.json"
        ),
        locations=yr_df["fips_5"],
        z=yr_df["yhat"],
        zmin=0.0, zmax=1.0,
        colorscale=COLORSCALE,
        text=hover,
        hoverinfo="text",
        marker_line_width=0.2,
        marker_line_color="white",
        colorbar=dict(
            title=dict(text="Forecast<br>Index", side="right"),
            thickness=14, len=0.6,
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["0.00<br>(Low)","0.25","0.50<br>(High)","0.75","1.00<br>(Critical)"],
            tickfont=dict(size=10),
            outlinewidth=0,
        ),
        name="",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Comorbidity Index Forecast — {year}</b><br>"
                 "<sup>Prophet baseline · ComorbidAlert</sup>",
            x=0.5, xanchor="center", font=dict(size=16),
        ),
        geo=dict(
            scope="usa", projection_type="albers usa",
            showlakes=True, lakecolor="rgb(240,248,255)",
            showland=True, landcolor="rgb(240,240,240)",
            showframe=False,
        ),
        annotations=[dict(
            x=0.5, y=-0.04, xref="paper", yref="paper",
            text=counts_str, showarrow=False,
            font=dict(size=12), align="center",
        )],
        margin=dict(l=0, r=0, t=70, b=50),
        height=600,
        paper_bgcolor="white",
    )
    return fig


# ── Save outputs ──────────────────────────────────────────────────────────────
def save_outputs(eval_df, forecast_df, summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Eval CSV
    eval_path = OUTPUT_DIR / "forecast_eval.csv"
    eval_df.to_csv(eval_path, index=False)
    log.info("Eval saved → %s", eval_path)

    # Summary CSV
    summary_path = OUTPUT_DIR / "forecast_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Summary saved → %s", summary_path)

    # Forecast parquet — local + S3
    forecast_path = OUTPUT_DIR / "forecast_results.parquet"
    forecast_df.to_parquet(forecast_path, index=False)
    log.info("Forecast saved → %s", forecast_path)

    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(forecast_df), buf)
    buf.seek(0)
    boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=FORECAST_KEY, Body=buf.read())
    log.info("Forecast S3 → s3://%s/%s", S3_BUCKET, FORECAST_KEY)


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main():
    panel = load_panel()

    # Run Prophet for all counties
    eval_df, forecast_df, models = run_all_counties(panel)

    # Aggregate metrics
    summary = summarize_eval(eval_df)

    # Save data outputs
    save_outputs(eval_df, forecast_df, summary)

    # Plot error distributions
    if not eval_df.empty:
        dist_fig = plot_eval_distributions(eval_df)
        dist_path = PLOTS_DIR / "eval_distributions.html"
        dist_fig.write_html(str(dist_path), include_plotlyjs="cdn")
        log.info("Error distributions → %s", dist_path)

    # Plot forecast choropleths for each horizon year
    for yr in HORIZON_YEARS:
        fig = plot_forecast_choropleth(forecast_df, yr)
        path = PLOTS_DIR / f"forecast_map_{yr}.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
        log.info("Forecast map %d → %s", yr, path)

    # Plot individual county forecasts
    for fips, name in HIGHLIGHT_FIPS.items():
        if fips in models:
            fig = plot_county_forecast(fips, name, models[fips], panel)
            path = PLOTS_DIR / f"county_{fips}.html"
            fig.write_html(str(path), include_plotlyjs="cdn")
            log.info("County plot %s → %s", name, path)
        else:
            log.info("County %s (%s) not in panel — skipping plot", fips, name)

    log.info("\n── Week 3 forecasting complete ──")
    log.info("Open these outputs:")
    log.info("  forecasting/outputs/forecast_eval.csv")
    log.info("  forecasting/outputs/forecast_summary.csv")
    log.info("  forecasting/outputs/forecast_plots/eval_distributions.html")
    for yr in HORIZON_YEARS:
        log.info("  forecasting/outputs/forecast_plots/forecast_map_%d.html", yr)


if __name__ == "__main__":
    main()