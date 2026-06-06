"""
BRFSS Ingestor
==============
BRFSS Prevalence & Trends Data — MMSA-level, long format → pivoted wide.
Dataset: https://data.cdc.gov/resource/j32a-sa6u.json
"""

import os
import logging
import requests
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

BRFSS_SOCRATA_URL = "https://data.cdc.gov/resource/j32a-sa6u.json"

# Question IDs we want → output column name
# Filter to Overall breakout only, specific response where needed
BRFSS_QUESTIONS = {
    "DRNKANY6":  "alcohol_any_30d_pct",       # Adults who had a drink in past 30 days
    "SMOKE100":  "ever_smoked_pct",            # Smoked at least 100 cigarettes
    "RFSMOK3":   "current_smoker_pct",         # Current smoker
    "EXERANY2":  "exercise_past30d_pct",       # Any exercise in past 30 days
    "SLEPTIM1":  "avg_sleep_hours",            # Average sleep hours (continuous)
    "PHYSHLTH":  "poor_phys_health_days",      # Poor physical health days
    "MENTHLTH":  "poor_mental_health_days",    # Poor mental health days
    "HLTHPLN1":  "has_health_coverage_pct",    # Has health coverage
    "MEDCOST1":  "couldnt_afford_care_pct",    # Couldn't afford to see doctor
}

# For binary Yes/No questions, we want the "Yes" response value
YES_RESPONSE_QUESTIONS = {
    "DRNKANY6", "SMOKE100", "EXERANY2", "HLTHPLN1", "MEDCOST1"
}


def fetch_brfss(
    year: int = 2023,
    app_token: Optional[str] = None,
    fallback_path: Optional[str] = None,
) -> pd.DataFrame:

    token = app_token or os.getenv("BRFSS_TOKEN", "")
    if not token:
        raise EnvironmentError("BRFSS_TOKEN not set in .env")

    headers = {"X-App-Token": token}

    question_ids = ", ".join(f"'{q}'" for q in BRFSS_QUESTIONS)

    params = {
        "$where": (
            f"year='{year}' "
            f"AND break_out='Overall' "
            f"AND questionid IN ({question_ids})"
        ),
        "$limit": 50_000,
        "$select": "locationabbr,locationdesc,questionid,response,data_value",
    }

    logger.info(f"BRFSS: fetching year={year} from Socrata SMART dataset")

    try:
        resp = requests.get(
            BRFSS_SOCRATA_URL,
            params=params,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        records = resp.json()

        if not records:
            raise ValueError(f"Empty response from BRFSS — year {year} may not be available yet")

        df = pd.DataFrame(records)

    except Exception as e:
        logger.warning(f"BRFSS Socrata fetch failed: {e}")
        if fallback_path and os.path.exists(fallback_path):
            logger.info(f"BRFSS: loading fallback file {fallback_path}")
            return _load_fallback(fallback_path)
        logger.error(
            "BRFSS: no fallback available. "
            "Download CSV from https://www.cdc.gov/brfss/smart/smart_data.htm"
        )
        return pd.DataFrame()

    df = _pivot_to_wide(df)
    df["source"] = "brfss"
    df["vintage_year"] = year

    logger.info(
        f"BRFSS: {len(df):,} MMSAs after pivot | "
        f"columns: {[c for c in df.columns if c not in ('source','vintage_year')]}"
    )
    return df


def _pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long format (one row per question+response) → wide format (one row per MMSA).
    For Yes/No questions, keep only the 'Yes' row.
    For continuous questions (sleep, health days), response IS the value.
    """
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")

    rows = []
    for (loc_abbr, loc_desc), grp in df.groupby(["locationabbr", "locationdesc"]):
        row = {"mmsa_code": loc_abbr, "mmsa_name": loc_desc}

        for qid, col_name in BRFSS_QUESTIONS.items():
            q_rows = grp[grp["questionid"] == qid]
            if q_rows.empty:
                row[col_name] = None
                continue

            if qid in YES_RESPONSE_QUESTIONS:
                yes_rows = q_rows[q_rows["response"].str.upper() == "YES"]
                row[col_name] = yes_rows["data_value"].iloc[0] if not yes_rows.empty else None
            else:
                # Continuous — take first non-null value
                vals = q_rows["data_value"].dropna()
                row[col_name] = vals.iloc[0] if not vals.empty else None

        rows.append(row)

    return pd.DataFrame(rows)


def _load_fallback(path: str) -> pd.DataFrame:
    if path.endswith(".xpt"):
        return pd.read_sas(path, format="xport", encoding="utf-8")
    return pd.read_csv(path, low_memory=False)