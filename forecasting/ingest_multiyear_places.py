"""
ingest_multiyear_places.py
==========================
Pulls CDC PLACES county data for releases 2020-2024, builds a
multi-year panel dataset, rescores comorbidity index for each year,
and writes a single versioned Parquet to S3.

Usage:
    cd ~/Documents/comorbid_alret
    source .venv/bin/activate
    python forecasting/ingest_multiyear_places.py

Output:
    s3://comorbid-alert-data/comorbid_alert/panel/comorbid_panel.parquet
    forecasting/outputs/comorbid_panel.parquet
"""

import io
import logging
import time
from pathlib import Path

import boto3
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv

load_dotenv("/Users/rohithsrinivasa/Documents/comorbid_alret/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "outputs"
S3_BUCKET  = "comorbid-alert-data"
S3_KEY     = "comorbid_alert/panel/comorbid_panel.parquet"

# ── Release registry ──────────────────────────────────────────────────────────
# fmt "wide"  → one row per county, measures as columns (GIS friendly)
# fmt "long"  → one row per county per measure, needs pivot on measureid
RELEASES = [
    {"release_year": 2020, "brfss_year": 2018, "dataset_id": "dv4u-3x3q", "fmt": "long"},
    {"release_year": 2021, "brfss_year": 2019, "dataset_id": "pqpp-u99h", "fmt": "long"},
    {"release_year": 2022, "brfss_year": 2020, "dataset_id": "duw2-7jbt", "fmt": "long"},
    {"release_year": 2023, "brfss_year": 2021, "dataset_id": "i46a-9kgh", "fmt": "wide"},
    {"release_year": 2024, "brfss_year": 2022, "dataset_id": "swc5-untb", "fmt": "long"},
]

# measureid values consistent across all long-format releases
MEASURE_IDS = {
    "DIABETES": "places_diabetes",
    "CHD":      "places_chd",
    "OBESITY":  "places_obesity",
    "STROKE":   "places_stroke",
    "BPHIGH":   "places_bphigh",
}

MEASURES = ["places_diabetes", "places_chd", "places_obesity",
            "places_stroke", "places_bphigh"]

# Valid US state FIPS (exclude territories: 60,66,69,72,78)
VALID_STATE_FIPS = {str(i).zfill(2) for i in range(1, 57)
                    if i not in [3, 7, 14, 43, 52]}


# ── Fetch helpers ─────────────────────────────────────────────────────────────
def paginate(url: str, params: dict, max_retries: int = 3) -> list:
    rows = []
    p = dict(params)
    p["$offset"] = 0

    for attempt in range(max_retries):
        try:
            while True:
                resp = requests.get(url, params=p, timeout=90)
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                rows.extend(batch)
                p["$offset"] += len(batch)
                if len(batch) < p.get("$limit", 50000):
                    break
            return rows
        except Exception as e:
            log.warning("Attempt %d failed: %s", attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                p["$offset"] = 0
                rows = []
            else:
                raise
    return rows


def fetch_long(release: dict) -> pd.DataFrame:
    """Fetch long-format release and pivot to wide."""
    dataset_id   = release["dataset_id"]
    release_year = release["release_year"]

    url = f"https://data.cdc.gov/resource/{dataset_id}.json"

    # Fetch only the measures we need + crude prevalence type
    measure_filter = " OR ".join(f"measureid='{m}'" for m in MEASURE_IDS)
    params = {
        "$limit":  50000,
        "$where":  f"({measure_filter}) AND data_value_type='Crude prevalence'",
        "$select": "locationid,locationname,stateabbr,measureid,data_value",
    }

    log.info("  Fetching long-format (measures only)...")
    try:
        rows = paginate(url, params)
    except Exception as e:
        log.error("  Failed: %s", e)
        return pd.DataFrame()

    if not rows:
        log.warning("  No rows returned")
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    raw.columns = raw.columns.str.lower()
    log.info("  Raw long: %d rows", len(raw))

    # Check for locationid (FIPS for county releases)
    fips_col = None
    for candidate in ["locationid", "countyfips", "locationname"]:
        if candidate in raw.columns:
            fips_col = candidate
            break

    if fips_col is None:
        log.error("  No FIPS-like column. Columns: %s", list(raw.columns))
        return pd.DataFrame()

    raw["fips"] = raw[fips_col].astype(str).str.zfill(5)
    raw["data_value"] = pd.to_numeric(raw["data_value"], errors="coerce")
    raw["measureid"] = raw["measureid"].str.upper()

    # Filter to county-level FIPS (5 digits, valid state prefix)
    raw = raw[raw["fips"].str.len() == 5]
    raw = raw[raw["fips"].str[:2].isin(VALID_STATE_FIPS)]

    # Filter to measures we want
    raw = raw[raw["measureid"].isin(MEASURE_IDS)]

    if raw.empty:
        log.warning("  No valid rows after filtering")
        return pd.DataFrame()

    # Pivot: fips × measureid → wide
    pivot = raw.pivot_table(
        index=["fips", "stateabbr"],
        columns="measureid",
        values="data_value",
        aggfunc="mean",
    ).reset_index()

    # Also grab county name
    name_map = raw.drop_duplicates("fips").set_index("fips")["locationname"]
    pivot["county_name"] = pivot["fips"].map(name_map)

    # Rename measureid columns to our standard names
    pivot = pivot.rename(columns={k: v for k, v in MEASURE_IDS.items()
                                   if k in pivot.columns})
    pivot.columns.name = None

    log.info("  Pivoted: %d counties", len(pivot))
    return pivot


def fetch_wide(release: dict) -> pd.DataFrame:
    """Fetch wide-format (GIS friendly) release — already one row per county."""
    dataset_id   = release["dataset_id"]
    release_year = release["release_year"]

    url = f"https://data.cdc.gov/resource/{dataset_id}.json"
    params = {"$limit": 50000}

    log.info("  Fetching wide-format...")
    try:
        rows = paginate(url, params)
    except Exception as e:
        log.error("  Failed: %s", e)
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    raw.columns = raw.columns.str.lower()
    log.info("  Raw wide: %d rows, %d cols", len(raw), len(raw.columns))

    # Find FIPS column
    fips_col = next((c for c in ["countyfips", "locationid", "fips"] if c in raw.columns), None)
    if fips_col is None:
        log.error("  No FIPS column. Columns: %s", list(raw.columns[:15]))
        return pd.DataFrame()

    raw["fips"] = raw[fips_col].astype(str).str.zfill(5)
    raw = raw[raw["fips"].str[:2].isin(VALID_STATE_FIPS)]

    # Map wide column names to our standard names
    wide_col_map = {
        "diabetes_crudeprev": "places_diabetes",
        "chd_crudeprev":      "places_chd",
        "obesity_crudeprev":  "places_obesity",
        "stroke_crudeprev":   "places_stroke",
        "bphigh_crudeprev":   "places_bphigh",
        "countyname":         "county_name",
        "locationname":       "county_name",
    }
    raw = raw.rename(columns={k: v for k, v in wide_col_map.items() if k in raw.columns})

    for col in MEASURES:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    log.info("  Wide: %d counties", len(raw))
    return raw


# ── Fetch one release ─────────────────────────────────────────────────────────
def fetch_release(release: dict) -> pd.DataFrame:
    release_year = release["release_year"]
    log.info("Fetching release %d (dataset: %s, fmt: %s)...",
             release_year, release["dataset_id"], release["fmt"])

    if release["fmt"] == "long":
        df = fetch_long(release)
    else:
        df = fetch_wide(release)

    if df.empty:
        return df

    # Ensure required columns exist
    df["release_year"] = release_year
    df["brfss_year"]   = release["brfss_year"]

    # Keep only what we need
    keep = ["fips", "county_name", "stateabbr", "release_year", "brfss_year"] + MEASURES
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Drop rows missing both key clinical measures
    df = df.dropna(subset=["places_diabetes", "places_chd"], how="all")

    log.info("Release %d ✅ — %d counties", release_year, len(df))
    return df


# ── Scoring ───────────────────────────────────────────────────────────────────
def minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def score_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw = (
        df["places_diabetes"].fillna(df["places_diabetes"].median())
        + df["places_chd"].fillna(df["places_chd"].median())
    )
    df["comorbid_l1_clinical"] = minmax(raw)
    df["comorbid_index"]       = df["comorbid_l1_clinical"]

    def tier(v):
        if v >= 0.75: return "Critical"
        if v >= 0.50: return "High"
        if v >= 0.25: return "Moderate"
        return "Low"

    df["risk_tier"] = df["comorbid_index"].apply(tier)
    return df


# ── Build panel ───────────────────────────────────────────────────────────────
def build_panel(frames: list) -> pd.DataFrame:
    panel = pd.concat(frames, ignore_index=True)
    county_counts = panel.groupby("fips")["release_year"].nunique()
    n_years = panel["release_year"].nunique()
    complete_fips = county_counts[county_counts == n_years].index
    dropped = len(panel["fips"].unique()) - len(complete_fips)

    if dropped > 0:
        log.warning("%d counties dropped (not in all %d years)", dropped, n_years)

    panel = panel[panel["fips"].isin(complete_fips)].copy()
    panel = panel.sort_values(["fips", "release_year"]).reset_index(drop=True)

    log.info("Panel: %d rows | %d counties × %d years",
             len(panel), len(complete_fips), n_years)
    log.info("Years: %s", sorted(panel["release_year"].unique().tolist()))
    return panel


# ── Save ──────────────────────────────────────────────────────────────────────
def save_panel(panel: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = OUTPUT_DIR / "comorbid_panel.parquet"
    panel.to_parquet(local_path, index=False)
    log.info("Local → %s", local_path)

    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(panel), buf)
    buf.seek(0)
    boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=buf.read())
    log.info("S3    → s3://%s/%s", S3_BUCKET, S3_KEY)


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main():
    frames = []

    for release in RELEASES:
        df = fetch_release(release)
        if df.empty:
            log.error("Skipping release %d", release["release_year"])
            continue
        df = score_year(df)
        frames.append(df)
        time.sleep(2)

    if not frames:
        log.error("No data fetched — exiting")
        return

    panel = build_panel(frames)

    # Sanity check — show a known Critical county across years
    for check_fips in ["46113", "28027", "01085"]:
        sample = panel[panel["fips"] == check_fips]
        if not sample.empty:
            log.info("\nSanity check FIPS %s:\n%s", check_fips,
                     sample[["fips", "release_year", "places_diabetes",
                              "places_chd", "comorbid_index"]].to_string(index=False))
            break

    save_panel(panel)
    log.info("\nDone. Panel ready for Prophet. Next: forecasting/forecast_prophet.py")


if __name__ == "__main__":
    main()