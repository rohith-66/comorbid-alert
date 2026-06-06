"""
data_loader.py — All S3 reads cached via st.cache_data
"""

import io
import streamlit as st
import pandas as pd
import boto3

BUCKET = "comorbid-alert-data"
MAIN_PARQUET = "comorbid_alert/year=2023/run_id=20260423T072218Z/part-0000.parquet"


def _s3():
    return boto3.client("s3")


@st.cache_data(ttl=3600, show_spinner=False)
def load_main() -> pd.DataFrame:
    s3 = _s3()
    obj = s3.get_object(Bucket=BUCKET, Key=MAIN_PARQUET)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df = df.rename(columns={
        "comorbid_l1_clinical":    "L1_score",
        "comorbid_l2_social":      "L2_score",
        "comorbid_l3_trajectory":  "L3_score",
    })
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_forecasts() -> pd.DataFrame:
    s3 = _s3()
    obj = s3.get_object(Bucket=BUCKET, Key="comorbid_alert/week5/ensemble_forecasts.parquet")
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df = df.rename(columns={
        "prophet_forecast":  "prophet_yhat",
        "lgbm_forecast":     "lgbm_yhat",
        "ensemble_forecast": "ensemble_yhat",
        "forecast_year":     "forecast_year",
    })
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_alerts() -> pd.DataFrame:
    s3 = _s3()
    obj = s3.get_object(Bucket=BUCKET, Key="comorbid_alert/week5/alert_log.csv")
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df = df.rename(columns={
        "alert_level":    "alert_tier",
        "reason":         "alert_reason",
        "forecast_slope": "slope",
    })
    df["alert_tier"] = df["alert_tier"].str.title()  # CRITICAL → Critical
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_shap() -> pd.DataFrame:
    return pd.DataFrame()


def get_county_forecasts(fips: str, forecast_df: pd.DataFrame) -> pd.DataFrame:
    return forecast_df[forecast_df["fips"] == fips].sort_values("forecast_year")


def get_county_alerts(fips: str, alert_df: pd.DataFrame) -> pd.DataFrame:
    return alert_df[alert_df["fips"] == fips]


def get_county_shap(fips: str, shap_df: pd.DataFrame) -> pd.DataFrame:
    return shap_df[shap_df["fips"] == fips]


TIER_COLOR = {
    "Critical": "#FF4D4D",
    "High":     "#FF8C00",
    "Moderate": "#FFD700",
    "Low":      "#28E08A",
}

TIER_BADGE = {
    "Critical": "badge-critical",
    "High":     "badge-high",
    "Moderate": "badge-moderate",
    "Low":      "badge-low",
}

ALERT_COLOR = {
    "Critical": "#FF4D4D",
    "Warning":  "#FF8C00",
    "Watch":    "#FFD700",
}
