import os
import time
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

DATASET_IDS = {
    2023: "swc5-untb",
    2022: "duw2-7jbt",
    2021: "cwsq-ngmh",
}

BASE_URL = "https://data.cdc.gov/resource/{dataset_id}.json"

TARGET_MEASURES = [
    "DIABETES", "BPHIGH", "OBESITY", "CASTHMA",
    "CHD", "STROKE", "COPD", "KIDNEY", "DEPRESSION", "SLEEP",
]

def fetch_cdc_places(year=2023, app_token=None, page_size=50000):
    dataset_id = DATASET_IDS.get(year)
    if not dataset_id:
        raise ValueError(f"Unsupported year {year}")

    token = app_token or os.getenv("CDC_PLACES_TOKEN", "")
    url = BASE_URL.format(dataset_id=dataset_id)
    measure_filter = ", ".join(f"'{m}'" for m in TARGET_MEASURES)
    headers = {"X-App-Token": token} if token else {}
    all_records = []
    offset = 0

    logger.info(f"CDC PLACES: fetching year={year}")

    while True:
        query = (
            f"?$where=measureid IN ({measure_filter})"
            f"&$limit={page_size}&$offset={offset}"
        )
        resp = requests.get(url + query, headers=headers, timeout=60)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_records.extend(batch)
        offset += page_size
        if len(batch) < page_size:
            break
        time.sleep(0.25)

    if not all_records:
        logger.warning("CDC PLACES returned 0 records")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"locationid": "fips", "locationname": "county_name"})
    df["fips"] = df["fips"].astype(str).str.zfill(5)

    for col in ["data_value", "low_confidence_limit", "high_confidence_limit", "totalpopulation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source"] = "cdc_places"
    df["vintage_year"] = year
    logger.info(f"CDC PLACES: {len(df):,} rows, {df['fips'].nunique():,} counties")
    return df
