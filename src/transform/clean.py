"""
Clean & Validate Transform
===========================
Source-aware cleaning rules applied before the FIPS join.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

NULL_DROP_THRESHOLD = 0.85
FIPS_RE = r"^\d{5}$"
EXCLUDE_STATE_FIPS = {"72", "78", "66", "60", "69"}


def clean_and_validate(df: pd.DataFrame, source: str) -> pd.DataFrame:

    if df.empty:
        logger.warning(f"clean_and_validate: {source} DataFrame is empty, skipping")
        return df

    original_rows = len(df)

    df = _strip_strings(df)
    df = _drop_high_null_cols(df, source)

    if "fips" in df.columns:
        df = _validate_fips(df, source)

    if source == "cdc_places":
        df = _clean_places(df)
    elif source == "brfss":
        df = _clean_brfss(df)
    elif source == "census_acs":
        df = _clean_census(df)

    dropped = original_rows - len(df)
    logger.info(f"  [{source}] cleaned: {len(df):,} rows kept, {dropped:,} dropped")
    return df


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


def _drop_high_null_cols(df: pd.DataFrame, source: str) -> pd.DataFrame:
    null_rate = df.isnull().mean()
    drop_cols = null_rate[null_rate > NULL_DROP_THRESHOLD].index.tolist()
    if drop_cols:
        logger.debug(f"  [{source}] dropping high-null columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    return df


def _validate_fips(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df["fips"] = df["fips"].astype(str).str.zfill(5)

    valid_format = df["fips"].str.match(FIPS_RE)
    n_invalid = (~valid_format).sum()
    if n_invalid:
        logger.warning(f"  [{source}] dropping {n_invalid} rows with invalid FIPS format")
    df = df[valid_format].copy()

    state_prefix = df["fips"].str[:2]
    excluded = state_prefix.isin(EXCLUDE_STATE_FIPS)
    if excluded.sum():
        logger.debug(f"  [{source}] excluding {excluded.sum()} territory rows")
    df = df[~excluded].copy()

    return df


def _clean_places(df: pd.DataFrame) -> pd.DataFrame:
    if "data_value" in df.columns:
        df = df[df["data_value"].between(0, 100) | df["data_value"].isna()]

    if "measureid" in df.columns:
        df = df[df["measureid"].notna()]

    return df


def _clean_brfss(df: pd.DataFrame) -> pd.DataFrame:
    sentinel_cols = df.select_dtypes(include="number").columns
    for col in sentinel_cols:
        col_max = df[col].max(skipna=True)
        if col_max in (99.0, 999.0, 9999.0):
            df[col] = df[col].replace({99.0: None, 999.0: None, 9999.0: None})

    return df


def _clean_census(df: pd.DataFrame) -> pd.DataFrame:
    SENTINEL_VALUES = {-666666666, -999999999, -888888888}
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        df[col] = df[col].apply(
            lambda x: None if (pd.notna(x) and x in SENTINEL_VALUES) else x
        )

    if "total_population" in df.columns:
        df = df[df["total_population"] > 0]

    return df