"""
Census ACS Ingestor
===================
Fetches county-level socioeconomic & demographic data from the
Census Bureau API (American Community Survey 5-Year estimates).
"""

import os
import logging
import requests
import pandas as pd
from typing import Optional, Dict

logger = logging.getLogger(__name__)

CENSUS_API_BASE = "https://api.census.gov/data/{year}/acs/acs5"

ACS_VARIABLES: Dict[str, str] = {
    # Demographics
    "B01003_001E": "total_population",
    "B01002_001E": "median_age",
    "B02001_003E": "black_alone",
    "B03003_003E": "hispanic_any_race",

    # Poverty & income
    "B19013_001E": "median_household_income",
    "B17001_002E": "below_poverty_count",
    "B17001_001E": "poverty_universe",

    # Insurance & healthcare access
    "B27001_001E": "health_insurance_universe",
    "B27001_005E": "uninsured_under18",
    "B27001_008E": "uninsured_18to34",
    "B27001_011E": "uninsured_35to64",

    # Housing
    "B25003_001E": "housing_units_total",
    "B25003_003E": "renter_occupied",
    "B25064_001E": "median_gross_rent",

    # Education
    "B15003_001E": "edu_universe",
    "B15003_017E": "high_school_diploma",
    "B15003_022E": "bachelors_degree",

    # Employment
    "B23025_003E": "employed",
    "B23025_005E": "unemployed",
    "B23025_002E": "labor_force",

    # Broadband
    "B28002_004E": "broadband_subscriptions",
    "B28002_001E": "broadband_universe",
}


def fetch_census_acs(
    year: int = 2023,
    api_key: Optional[str] = None,
) -> pd.DataFrame:

    key = api_key or os.getenv("CENSUS_API_KEY", "")
    if not key:
        logger.warning(
            "No Census API key set. Requests may be rate-limited. "
            "Register free at https://api.census.gov/data/key_signup.html"
        )

    url = CENSUS_API_BASE.format(year=year)
    var_list = ",".join(["NAME"] + list(ACS_VARIABLES.keys()))

    params = {
        "get": var_list,
        "for": "county:*",
        "in":  "state:*",
    }
    if key:
        params["key"] = key

    logger.info(f"Census ACS: fetching {year} 5-year estimates ({len(ACS_VARIABLES)} variables)")

    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    raw = resp.json()

    columns = raw[0]
    rows    = raw[1:]
    df      = pd.DataFrame(rows, columns=columns)

    df["fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    df = df.drop(columns=["state", "county", "NAME"], errors="ignore")
    df = df.rename(columns=ACS_VARIABLES)

    numeric_cols = list(ACS_VARIABLES.values())
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _derive_rates(df)

    df["source"]       = "census_acs"
    df["vintage_year"] = year

    logger.info(f"Census ACS: {len(df):,} counties")
    return df


def _derive_rates(df: pd.DataFrame) -> pd.DataFrame:
    safe_div = lambda n, d: (n / d * 100).where(d > 0)

    if {"below_poverty_count", "poverty_universe"}.issubset(df.columns):
        df["poverty_rate_pct"] = safe_div(df["below_poverty_count"], df["poverty_universe"])

    if {"unemployed", "labor_force"}.issubset(df.columns):
        df["unemployment_rate_pct"] = safe_div(df["unemployed"], df["labor_force"])

    if {"renter_occupied", "housing_units_total"}.issubset(df.columns):
        df["renter_rate_pct"] = safe_div(df["renter_occupied"], df["housing_units_total"])

    if {"broadband_subscriptions", "broadband_universe"}.issubset(df.columns):
        df["broadband_rate_pct"] = safe_div(df["broadband_subscriptions"], df["broadband_universe"])

    if {"high_school_diploma", "edu_universe"}.issubset(df.columns):
        df["hs_diploma_rate_pct"] = safe_div(df["high_school_diploma"], df["edu_universe"])

    return df