"""
data/loader.py — S3 data access layer with Streamlit caching.
All paths and column names matched to actual S3 contents.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(_ROOT), ".env"))  # comorbid_alret/.env

import io
import json
import boto3
import numpy as np
import pandas as pd
import streamlit as st

BUCKET = "comorbid-alert-data"
PREFIX = "comorbid_alert"

# ── S3 client ──────────────────────────────────────────────────────────────────
@st.cache_resource
def _s3():
    return boto3.client("s3")

def _read_parquet(key: str) -> pd.DataFrame:
    obj = _s3().get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def _read_csv(key: str) -> pd.DataFrame:
    obj = _s3().get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def _latest_run_key() -> str:
    """Read _latest.json to get the most recent raw parquet run_id."""
    try:
        obj = _s3().get_object(Bucket=BUCKET, Key=f"{PREFIX}/year=2023/_latest.json")
        meta = json.loads(obj["Body"].read())
        run_id = meta.get("run_id", "20260425T220000Z")
        return f"{PREFIX}/year=2023/run_id={run_id}/part-0000.parquet"
    except Exception:
        return f"{PREFIX}/year=2023/run_id=20260425T220000Z/part-0000.parquet"

# ── Public loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_comorbidity_index() -> pd.DataFrame:
    """
    Base comorbidity index with L1/L2/L3 breakdown + risk tiers.
    Source: raw pipeline parquet (has all three layers).
    Columns used: fips, county_name, stateabbr, comorbid_l1_clinical,
                  comorbid_l2_social, comorbid_l3_trajectory, comorbid_index, risk_tier
    """
    key = _latest_run_key()
    df = _read_parquet(key)
    df["fips"] = df["fips"].astype(str).str.zfill(5)

    # Standardise to canonical names used throughout the dashboard
    df = df.rename(columns={
        "county_name":              "county",
        "stateabbr":                "state",
        "comorbid_l1_clinical":     "l1_score",
        "comorbid_l2_social":       "l2_score",
        "comorbid_l3_trajectory":   "l3_score",
        "comorbid_index":           "comorbidity_index",
    })

    keep = [
        "fips", "county", "state",
        "l1_score", "l2_score", "l3_score",
        "comorbidity_index", "risk_tier",
        # clinical features for SHAP proxy
        "places_diabetes", "places_chd", "places_obesity",
        "places_stroke", "places_bphigh",
        # social features
        "poverty_rate_pct", "unemployment_rate_pct",
        "renter_rate_pct", "broadband_rate_pct",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].drop_duplicates("fips").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_ensemble_forecasts() -> pd.DataFrame:
    """
    Ensemble forecasts 2025-2027.
    Columns: fips, forecast_year, ensemble_forecast, prophet_forecast,
             lgbm_forecast, county_name, stateabbr, current_tier
    """
    df = _read_parquet(f"{PREFIX}/week5/ensemble_forecasts.parquet")
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df = df.rename(columns={
        "forecast_year":    "year",
        "ensemble_forecast":"predicted_ensemble",
        "county_name":      "county",
        "stateabbr":        "state",
    })
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_prophet_forecasts() -> pd.DataFrame:
    """
    Prophet forecasts with confidence intervals.
    Columns: fips, county_name, stateabbr, forecast_year, yhat, yhat_lower, yhat_upper
    """
    df = _read_parquet(f"{PREFIX}/panel/forecast_results.parquet")
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df = df.rename(columns={
        "forecast_year": "year",
        "county_name":   "county",
        "stateabbr":     "state",
    })
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_lgbm_forecasts() -> pd.DataFrame:
    """
    LightGBM forecasts.
    Columns: fips, forecast_year, lgbm_forecast, ...
    """
    df = _read_parquet(f"{PREFIX}/panel/lgbm_forecast.parquet")
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df = df.rename(columns={
        "forecast_year": "year",
        "county_name":   "county",
        "stateabbr":     "state",
    })
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_alerts() -> pd.DataFrame:
    """
    Alert log — 830 alerts.
    Columns: fips, county_name, stateabbr, alert_level, current_tier,
             tier_2027, comorbid_index_2024, ens_2025, ens_2027,
             forecast_slope, forecast_pct_chg, reason
    """
    df = _read_csv(f"{PREFIX}/week5/alert_log.csv")
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df = df.rename(columns={
        "county_name":          "county",
        "stateabbr":            "state",
        "comorbid_index_2024":  "comorbidity_index",
    })
    df["alert_level"] = df["alert_level"].str.upper()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_forecast_map(year: int) -> pd.DataFrame:
    """County-level ensemble forecast for a given year, merged with index metadata."""
    ens = load_ensemble_forecasts()
    if ens.empty:
        return pd.DataFrame()
    sub = ens[ens["year"] == year].copy()

    idx = load_comorbidity_index()
    merged = sub.merge(
        idx[["fips", "county", "state", "risk_tier", "l1_score", "l2_score", "l3_score"]],
        on="fips", how="left", suffixes=("", "_base"),
    )
    # Use forecast value as the map colour
    merged["comorbidity_index"] = merged["predicted_ensemble"]
    return merged


# ── County drill-down ──────────────────────────────────────────────────────────

def get_county_data(fips: str) -> dict:
    """Return all dashboard data for a single county."""
    idx = load_comorbidity_index()
    row = idx[idx["fips"] == fips]
    if row.empty:
        return {}
    rec = row.iloc[0].to_dict()

    # ── Ensemble forecast series ───────────────────────────────────────────────
    ens = load_ensemble_forecasts()
    if not ens.empty:
        county_ens = ens[ens["fips"] == fips].sort_values("year")
        rec["forecast_ensemble"] = (
            county_ens[["year", "predicted_ensemble"]].to_dict("records")
        )

    # ── Prophet forecast + CI ─────────────────────────────────────────────────
    prophet = load_prophet_forecasts()
    if not prophet.empty and fips in prophet["fips"].values:
        cp = prophet[prophet["fips"] == fips].sort_values("year")
        rec["forecast_prophet"] = cp[["year", "yhat"]].rename(
            columns={"yhat": "value"}
        ).to_dict("records")
        if "yhat_lower" in cp.columns and "yhat_upper" in cp.columns:
            rec["prophet_ci"] = cp[["year", "yhat_lower", "yhat_upper"]].to_dict("records")

    # ── LightGBM forecast ─────────────────────────────────────────────────────
    lgbm = load_lgbm_forecasts()
    if not lgbm.empty and fips in lgbm["fips"].values:
        cl = lgbm[lgbm["fips"] == fips].sort_values("year")
        rec["forecast_lgbm"] = cl[["year", "lgbm_forecast"]].rename(
            columns={"lgbm_forecast": "value"}
        ).to_dict("records")

    # ── SHAP proxy — use clinical feature values as importance proxy ───────────
    # Real SHAP file not in S3; derive a ranked display from feature values
    feature_display = {
        "Diabetes Prevalence":   rec.get("places_diabetes", np.nan),
        "Obesity Prevalence":    rec.get("places_obesity",  np.nan),
        "High Blood Pressure":   rec.get("places_bphigh",   np.nan),
        "Heart Disease (CHD)":   rec.get("places_chd",      np.nan),
        "Stroke Prevalence":     rec.get("places_stroke",   np.nan),
        "Poverty Rate":          rec.get("poverty_rate_pct",np.nan),
        "Unemployment Rate":     rec.get("unemployment_rate_pct", np.nan),
    }
    # Normalise to 0-1 and convert to signed SHAP-style values
    # (above national median = positive contribution to risk)
    NAT_MEDIANS = {
        "Diabetes Prevalence": 11.5, "Obesity Prevalence": 33.0,
        "High Blood Pressure": 33.0, "Heart Disease (CHD)": 6.2,
        "Stroke Prevalence": 3.5,    "Poverty Rate": 13.0,
        "Unemployment Rate": 4.5,
    }
    SCALES = {
        "Diabetes Prevalence": 8.0, "Obesity Prevalence": 10.0,
        "High Blood Pressure": 8.0, "Heart Disease (CHD)": 4.0,
        "Stroke Prevalence": 3.0,   "Poverty Rate": 10.0,
        "Unemployment Rate": 4.0,
    }
    shap_proxy = {}
    for feat, val in feature_display.items():
        if not np.isnan(float(val)) if val is not None else False:
            shap_proxy[feat] = round((float(val) - NAT_MEDIANS[feat]) / SCALES[feat], 4)
    # Top 5 by absolute value
    top5 = dict(sorted(shap_proxy.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    rec["shap_top5"] = top5

    # ── Alert ──────────────────────────────────────────────────────────────────
    alerts = load_alerts()
    county_alerts = alerts[alerts["fips"] == fips]
    if not county_alerts.empty:
        rec["alert_level"]  = county_alerts.iloc[0]["alert_level"]
        rec["alert_reason"] = county_alerts.iloc[0].get("reason", "")
    else:
        rec["alert_level"]  = None
        rec["alert_reason"] = ""

    return rec


# ── Constants used by pages ────────────────────────────────────────────────────

TIER_COLORS = {
    "Critical": "#e63946",
    "High":     "#f4a261",
    "Moderate": "#e9c46a",
    "Low":      "#52b788",
}

ALERT_COLORS = {
    "CRITICAL": "#e63946",
    "WARNING":  "#f4a261",
    "WATCH":    "#e9c46a",
}

GREAT_PLAINS_STATES = {
    "ND", "SD", "NE", "KS", "OK", "MT", "WY", "CO",
    "North Dakota", "South Dakota", "Nebraska", "Kansas",
    "Oklahoma", "Montana", "Wyoming", "Colorado",
}