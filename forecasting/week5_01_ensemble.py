"""
ComorbidAlert — Week 5, Step 1: Weighted Ensemble
==================================================
Exact column names confirmed from s3_inspect.py:

  comorbid_panel.parquet     → fips, release_year, comorbid_index, risk_tier, county_name, stateabbr
  forecast_results.parquet   → fips, forecast_year, yhat            (Prophet, 2025-2027)
  lgbm_forecast.parquet      → fips, forecast_year, lgbm_forecast   (LightGBM, 2025-2027)

Neither model has holdout actuals in S3. We compute per-county holdout WAPE by
treating 2024 panel values as ground truth and the 2025 first-step forecast
as the prediction (the closest proxy to a true holdout error).

Weighting approach
------------------
  For each county:
    - Prophet proxy error  = |yhat_2025 - comorbid_index_2024| / comorbid_index_2024
    - LightGBM proxy error = |lgbm_2025 - comorbid_index_2024| / comorbid_index_2024
  Inverse of these → per-county weights.

Outputs → s3://comorbid-alert-data/
  comorbid_alert/week5/ensemble_forecasts.parquet
  comorbid_alert/week5/ensemble_weights.parquet
  comorbid_alert/week5/ensemble_metrics.csv
"""

import io, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aws_session import get_s3_client

warnings.filterwarnings("ignore")

BUCKET = "comorbid-alert-data"
s3     = get_s3_client()

FORECAST_YEARS = [2025, 2026, 2027]
EPS = 1e-6

# ── helpers ───────────────────────────────────────────────────────────────────

def s3_read(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def s3_write_parquet(df, key):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.read())
    print(f"  ✓ s3://{BUCKET}/{key}  ({len(df):,} rows)")

def s3_write_csv(df, key):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.read())
    print(f"  ✓ s3://{BUCKET}/{key}  ({len(df):,} rows)")

def wape(actual, predicted):
    mask = (actual != 0) & ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return np.nan
    return np.abs(actual[mask] - predicted[mask]).sum() / np.abs(actual[mask]).sum() * 100

def mae(actual, predicted):
    return np.abs(actual - predicted).mean()

def rmse(actual, predicted):
    return np.sqrt(((actual - predicted) ** 2).mean())

# ── 1. Load data ──────────────────────────────────────────────────────────────

print("Loading panel …")
panel = s3_read("comorbid_alert/panel/comorbid_panel.parquet")
print(f"  {panel.shape}  |  years: {sorted(panel['release_year'].unique())}")

print("Loading Prophet forecasts …")
prophet = s3_read("comorbid_alert/panel/forecast_results.parquet")
print(f"  {prophet.shape}  |  years: {sorted(prophet['forecast_year'].unique())}")

print("Loading LightGBM forecasts …")
lgbm = s3_read("comorbid_alert/panel/lgbm_forecast.parquet")
print(f"  {lgbm.shape}  |  years: {sorted(lgbm['forecast_year'].unique())}")

# ── 2. 2024 actuals — ground truth baseline ───────────────────────────────────

print("\nExtracting 2024 actuals …")
panel_2024 = (
    panel[panel["release_year"] == 2024]
    [["fips", "county_name", "stateabbr", "comorbid_index", "risk_tier"]]
    .rename(columns={"comorbid_index": "actual_2024", "risk_tier": "current_tier"})
)
print(f"  Counties with 2024 data: {len(panel_2024):,}")

# ── 3. 2025 first-step forecasts ──────────────────────────────────────────────

prophet_2025 = (
    prophet[prophet["forecast_year"] == 2025][["fips", "yhat"]]
    .rename(columns={"yhat": "prophet_2025"})
)
lgbm_2025 = (
    lgbm[lgbm["forecast_year"] == 2025][["fips", "lgbm_forecast"]]
    .rename(columns={"lgbm_forecast": "lgbm_2025"})
)

# ── 4. Per-county inverse-error weights ───────────────────────────────────────

print("Computing per-county proxy WAPE …")
w = (
    panel_2024
    .merge(prophet_2025, on="fips", how="inner")
    .merge(lgbm_2025,    on="fips", how="inner")
)

w["err_prophet"] = np.abs(w["prophet_2025"] - w["actual_2024"]) / w["actual_2024"].clip(lower=EPS) * 100
w["err_lgbm"]    = np.abs(w["lgbm_2025"]    - w["actual_2024"]) / w["actual_2024"].clip(lower=EPS) * 100

w["inv_prophet"] = 1.0 / w["err_prophet"].clip(lower=EPS)
w["inv_lgbm"]    = 1.0 / w["err_lgbm"].clip(lower=EPS)
w["inv_total"]   = w["inv_prophet"] + w["inv_lgbm"]
w["w_prophet"]   = w["inv_prophet"] / w["inv_total"]
w["w_lgbm"]      = w["inv_lgbm"]    / w["inv_total"]

assert np.allclose(w["w_prophet"] + w["w_lgbm"], 1.0)

print(f"  Counties weighted     : {len(w):,}")
print(f"  Median err_prophet    : {w['err_prophet'].median():.3f}%")
print(f"  Median err_lgbm       : {w['err_lgbm'].median():.3f}%")
print(f"  Median w_prophet      : {w['w_prophet'].median():.3f}")
print(f"  Median w_lgbm         : {w['w_lgbm'].median():.3f}")
print(f"  LightGBM majority wt  : {(w['w_lgbm'] > 0.5).sum():,} counties "
      f"({(w['w_lgbm'] > 0.5).mean()*100:.1f}%)")

# ── 5. Ensemble across 2025–2027 ──────────────────────────────────────────────

print("\nBuilding full ensemble …")

prophet_all = (
    prophet[prophet["forecast_year"].isin(FORECAST_YEARS)]
    [["fips", "forecast_year", "yhat"]]
    .rename(columns={"yhat": "prophet_forecast"})
)
lgbm_all = (
    lgbm[lgbm["forecast_year"].isin(FORECAST_YEARS)]
    [["fips", "forecast_year", "lgbm_forecast"]]
)

fc = (
    prophet_all
    .merge(lgbm_all, on=["fips", "forecast_year"], how="outer")
    .merge(w[["fips", "county_name", "stateabbr", "current_tier",
              "actual_2024", "w_prophet", "w_lgbm",
              "err_prophet", "err_lgbm"]],
           on="fips", how="left")
)

fc["w_prophet"] = fc["w_prophet"].fillna(0.5)
fc["w_lgbm"]    = fc["w_lgbm"].fillna(0.5)

fc["ensemble_forecast"] = (
    fc["w_prophet"] * fc["prophet_forecast"].fillna(fc["lgbm_forecast"]) +
    fc["w_lgbm"]    * fc["lgbm_forecast"].fillna(fc["prophet_forecast"])
)

fc = fc.sort_values(["fips", "forecast_year"]).reset_index(drop=True)
print(f"  Rows: {len(fc):,}  |  Counties: {fc['fips'].nunique():,}  |  Years: {sorted(fc['forecast_year'].unique())}")

# ── 6. Evaluation — 2025 forecast vs 2024 actual ─────────────────────────────

print("\n── Evaluation: 2025 forecast vs 2024 actual (proxy holdout) ────────")
eval_df = fc[fc["forecast_year"] == 2025].dropna(subset=["actual_2024"])
a = eval_df["actual_2024"].values

results = {}
for model, col in [("Prophet",  "prophet_forecast"),
                   ("LightGBM", "lgbm_forecast"),
                   ("Ensemble", "ensemble_forecast")]:
    p = eval_df[col].values
    results[model] = {
        "MAE":      round(mae(a, p), 6),
        "RMSE":     round(rmse(a, p), 6),
        "WAPE (%)": round(wape(a, p), 4),
    }

eval_summary = pd.DataFrame(results).T
print(eval_summary.to_string())

# Ensemble county-win rate
w2 = w.copy()
ens_2025 = fc[fc["forecast_year"] == 2025].set_index("fips")["ensemble_forecast"]
w2["err_ensemble"] = (
    np.abs(ens_2025.reindex(w2["fips"]).values - w2["actual_2024"])
    / w2["actual_2024"].clip(lower=EPS) * 100
)
w2["ensemble_wins"] = w2["err_ensemble"] < w2[["err_prophet","err_lgbm"]].min(axis=1)
wins = w2["ensemble_wins"].sum()
print(f"\n  Ensemble beats BOTH: {wins:,} / {len(w2):,} counties ({wins/len(w2)*100:.1f}%)")

# ── 7. Save ───────────────────────────────────────────────────────────────────

print("\nSaving …")
s3_write_parquet(fc, "comorbid_alert/week5/ensemble_forecasts.parquet")

weights_out = w[["fips","county_name","stateabbr","current_tier",
                  "actual_2024","prophet_2025","lgbm_2025",
                  "err_prophet","err_lgbm","w_prophet","w_lgbm"]]
s3_write_parquet(weights_out, "comorbid_alert/week5/ensemble_weights.parquet")

metrics_out = eval_summary.reset_index().rename(columns={"index": "model"})
s3_write_csv(metrics_out, "comorbid_alert/week5/ensemble_metrics.csv")

print("\nWeek 5 Step 1 complete.")
print(eval_summary.to_string())