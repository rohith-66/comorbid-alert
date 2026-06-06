"""
S3 Versioned Parquet Writer
============================
Writes the combined county DataFrame to S3 as partitioned, versioned Parquet.

S3 path convention:
    s3://{bucket}/comorbid_alert/
        year={year}/
            run_id={run_id}/
                part-0000.parquet
                _metadata.json
"""

import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

def _get_bucket() -> str:
    return os.getenv("COMORBID_S3_BUCKET", "comorbid-alert-data")

def _get_prefix() -> str:
    return os.getenv("COMORBID_S3_PREFIX", "comorbid_alert")

PARQUET_COMPRESSION = "snappy"
ROW_GROUP_SIZE      = 50_000


def write_versioned_parquet(
    df: pd.DataFrame,
    year: int,
    run_id: str,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    partition_by_state: bool = False,
) -> str:

    bucket = bucket or _get_bucket()
    prefix = prefix or _get_prefix()

    s3 = _get_s3_client()

    dataset_prefix = f"{prefix}/year={year}/run_id={run_id}"
    s3_root        = f"s3://{bucket}/{dataset_prefix}"

    logger.info(f"S3 writer: target={s3_root} | rows={len(df):,} | compression={PARQUET_COMPRESSION}")

    table = _df_to_arrow(df)

    if partition_by_state and "state_fips" in df.columns:
        _write_partitioned(s3, table, bucket, dataset_prefix, df)
    else:
        _write_single(s3, table, bucket, dataset_prefix)

    metadata = _build_metadata(df, year, run_id, s3_root)
    _write_json(s3, bucket, f"{dataset_prefix}/_metadata.json", metadata)

    latest_key = f"{prefix}/year={year}/_latest.json"
    _write_json(s3, bucket, latest_key, {"run_id": run_id, "s3_root": s3_root})
    logger.info(f"S3 writer: updated latest pointer → s3://{bucket}/{latest_key}")

    return s3_root


def read_latest_parquet(
    year: int,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:

    bucket = bucket or _get_bucket()
    prefix = prefix or _get_prefix()    
    s3     = _get_s3_client()

    latest_key = f"{prefix}/year={year}/_latest.json"
    obj        = s3.get_object(Bucket=bucket, Key=latest_key)
    latest     = json.loads(obj["Body"].read())
    s3_root    = latest["s3_root"]

    logger.info(f"Loading latest dataset from {s3_root}")
    return pd.read_parquet(s3_root, storage_options=_storage_options())


def _get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def _storage_options() -> dict:
    return {
        "key":    os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "token":  os.getenv("AWS_SESSION_TOKEN"),
    }


def _df_to_arrow(df: pd.DataFrame) -> pa.Table:
    df = df.copy()
    if "fips" in df.columns:
        df["fips"] = df["fips"].astype(str)
    return pa.Table.from_pandas(df, preserve_index=False)


def _write_single(s3, table: pa.Table, bucket: str, prefix: str):
    buf = io.BytesIO()
    pq.write_table(
        table,
        buf,
        compression=PARQUET_COMPRESSION,
        row_group_size=ROW_GROUP_SIZE,
    )
    buf.seek(0)
    key = f"{prefix}/part-0000.parquet"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    size_mb = len(buf.getvalue()) / 1_048_576
    logger.info(f"  Wrote s3://{bucket}/{key} ({size_mb:.2f} MB)")


def _write_partitioned(s3, table: pa.Table, bucket: str, prefix: str, df: pd.DataFrame):
    for state_fips, group in df.groupby("state_fips"):
        part_table = _df_to_arrow(group)
        buf = io.BytesIO()
        pq.write_table(part_table, buf, compression=PARQUET_COMPRESSION)
        buf.seek(0)
        key = f"{prefix}/state_fips={state_fips}/part-0000.parquet"
        s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
        logger.debug(f"  Wrote partition state_fips={state_fips}")


def _build_metadata(df: pd.DataFrame, year: int, run_id: str, s3_root: str) -> dict:
    return {
        "run_id":         run_id,
        "vintage_year":   year,
        "s3_root":        s3_root,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count":      len(df),
        "county_count":   int(df["fips"].nunique()) if "fips" in df.columns else None,
        "columns":        df.columns.tolist(),
        "sources":        list(df["source"].unique()) if "source" in df.columns else [],
        "null_rates":     {
            col: round(float(df[col].isnull().mean()), 4)
            for col in df.columns
            if df[col].isnull().any()
        },
    }


def _write_json(s3, bucket: str, key: str, data: dict):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2).encode(),
        ContentType="application/json",
    )