"""
FIPS Join Transform
===================
Joins CDC PLACES, BRFSS, and Census ACS datasets on 5-digit FIPS county codes.
"""
import os
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

MMSA_FIPS_CROSSWALK_URL = (
    "https://www2.census.gov/programs-surveys/metro-micro/"
    "geographies/reference-files/2023/delineation-files/list1_2023.xlsx"
)


def join_on_fips(
    places_df: pd.DataFrame,
    brfss_df:  pd.DataFrame,
    census_df: pd.DataFrame,
    crosswalk_path: Optional[str] = None,
) -> pd.DataFrame:

    logger.info("Join: starting FIPS assembly")

    places_wide = _pivot_places(places_df)
    logger.info(f"Join: PLACES pivoted → {places_wide.shape}")

    brfss_county = _brfss_to_county(brfss_df, crosswalk_path)
    logger.info(f"Join: BRFSS county-level → {brfss_county.shape}")

    logger.info(f"Join: Census spine → {census_df.shape}")
    spine = census_df.copy()
    spine["fips"] = spine["fips"].astype(str).str.zfill(5)

    combined = spine.merge(places_wide, on="fips", how="left", suffixes=("", "_places"))

    if not brfss_county.empty:
        combined = combined.merge(brfss_county, on="fips", how="left", suffixes=("", "_brfss"))
    else:
        logger.warning("Join: BRFSS county data empty — BRFSS columns will be absent")

    combined["fips"]        = combined["fips"].astype(str).str.zfill(5)
    combined["state_fips"]  = combined["fips"].str[:2]
    combined["county_fips"] = combined["fips"].str[2:]

    dup_cols = [c for c in combined.columns if c.endswith(("_places", "_brfss"))]
    combined = combined.drop(columns=dup_cols, errors="ignore")

    _log_coverage(combined)
    logger.info(f"Join: final table {combined.shape}")

    return combined


def _pivot_places(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "measureid" not in df.columns:
        return pd.DataFrame(columns=["fips"])

    pivot = df.pivot_table(
        index="fips",
        columns="measureid",
        values="data_value",
        aggfunc="mean",
    ).reset_index()

    pivot.columns.name = None
    pivot.columns = ["fips"] + [
        f"places_{c.lower()}" for c in pivot.columns if c != "fips"
    ]

    meta_cols = ["fips", "county_name", "stateabbr", "statedesc", "totalpopulation"]
    meta = df[meta_cols].drop_duplicates("fips").copy()
    pivot = pivot.merge(meta, on="fips", how="left")

    return pivot


def _brfss_to_county(df: pd.DataFrame, crosswalk_path: Optional[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["fips"])

    crosswalk = _load_mmsa_crosswalk(crosswalk_path)
    if crosswalk.empty:
        logger.warning("BRFSS: MMSA→FIPS crosswalk unavailable; skipping BRFSS join")
        return pd.DataFrame(columns=["fips"])

    if "mmsa_code" not in df.columns:
        logger.warning("BRFSS: 'mmsa_code' column missing; cannot join")
        return pd.DataFrame(columns=["fips"])

    df["mmsa_code"]        = df["mmsa_code"].astype(str)
    crosswalk["mmsa_code"] = crosswalk["mmsa_code"].astype(str)

    merged = df.merge(crosswalk, on="mmsa_code", how="inner")
    merged = merged.rename(columns={"county_fips_full": "fips"})

    numeric_cols = merged.select_dtypes(include="number").columns.tolist()
    brfss_county = merged.groupby("fips")[numeric_cols].mean().reset_index()

    logger.info(f"BRFSS: mapped to {brfss_county['fips'].nunique():,} counties via crosswalk")
    return brfss_county

def _load_mmsa_crosswalk(path: Optional[str]) -> pd.DataFrame:
    import io, requests

    # ── Option 1: local file ──────────────────────────────────────────────────
    if path and os.path.exists(path):
        try:
            ext = os.path.splitext(path)[1].lower()
            engine = "openpyxl" if ext == ".xlsx" else "xlrd"
            raw = open(path, "rb").read()
            xls = pd.read_excel(io.BytesIO(raw), sheet_name=0, header=2, dtype=str, engine=engine)
            return _parse_crosswalk_excel(xls)
        except Exception as e:
            logger.warning(f"Local crosswalk load failed: {e} — trying remote")

    # ── Option 2: Census Bureau XLSX with browser User-Agent ─────────────────
    try:
        logger.info("BRFSS: downloading MMSA→FIPS crosswalk from Census Bureau")
        url = (
            "https://www2.census.gov/programs-surveys/metro-micro/"
            "geographies/reference-files/2023/delineation-files/list1_2023.xlsx"
        )
        r = requests.get(url, timeout=60, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        r.raise_for_status()

        if b"DOCTYPE" in r.content[:100]:
            raise ValueError("Got HTML instead of XLSX")

        # Cache it locally for future runs
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(r.content)
            logger.info(f"BRFSS: crosswalk cached to {path}")

        xls = pd.read_excel(io.BytesIO(r.content), sheet_name=0, header=2, dtype=str, engine="openpyxl")
        return _parse_crosswalk_excel(xls)

    except Exception as e:
        logger.warning(f"MMSA crosswalk load failed: {e}")
        return pd.DataFrame()


def _parse_crosswalk_excel(xls: pd.DataFrame) -> pd.DataFrame:
    """Parse Census delineation file with exact column names from list1_2023.xlsx."""
    col_map = {
        "CBSA Code":        "mmsa_code",
        "FIPS State Code":  "state_fips",
        "FIPS County Code": "county_fips",
    }
    xls = xls.rename(columns=col_map)

    required = {"mmsa_code", "state_fips", "county_fips"}
    if not required.issubset(xls.columns):
        raise ValueError(f"Crosswalk missing columns: {required - set(xls.columns)}")

    xls["county_fips_full"] = xls["state_fips"].str.zfill(2) + xls["county_fips"].str.zfill(3)
    result = xls[["mmsa_code", "county_fips_full"]].dropna()
    logger.info(f"BRFSS: crosswalk parsed — {len(result):,} CBSA→county mappings")
    return result


def _log_coverage(df: pd.DataFrame) -> dict:
    coverage = {}
    for prefix in ["places_", "brfss_", "census_", "poverty_", "unemployment_"]:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            pct_complete = (df[cols].notna().mean() * 100).mean()
            logger.info(f"  Coverage [{prefix.rstrip('_')}]: {pct_complete:.1f}%")
            coverage[prefix] = pct_complete
    return coverage