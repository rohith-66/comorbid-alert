"""
ComorbidAlert — Data Pipeline
==============================
Pulls CDC PLACES API, BRFSS, and Census ACS data,
joins on FIPS county codes, stores versioned Parquet on S3.

Usage:
    python pipeline.py [--year 2023] [--dry-run]
"""
import os
import argparse
import logging
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # must be before any src imports

from src.ingest.cdc_places import fetch_cdc_places
from src.ingest.brfss import fetch_brfss
from src.ingest.census import fetch_census_acs
from src.transform.join import join_on_fips
from src.transform.clean import clean_and_validate
from src.transform.score import build_comorbid_index
from src.storage.s3_writer import write_versioned_parquet
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="ComorbidAlert data pipeline")
    parser.add_argument("--year", type=int, default=2023, help="Data vintage year")
    parser.add_argument("--dry-run", action="store_true", help="Skip S3 write, print summary only")
    return parser.parse_args()


def run_pipeline(year: int, dry_run: bool = False):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info(f"=== ComorbidAlert pipeline start | run_id={run_id} | year={year} ===")

    # ── 1. Ingest ─────────────────────────────────────────────────────────────
    logger.info("Step 1/4 — Ingesting sources...")
    places_df = fetch_cdc_places(year=year)
    brfss_df  = fetch_brfss(year=year)
    census_df = fetch_census_acs(year=year)

    logger.info(f"  CDC PLACES : {len(places_df):,} rows")
    logger.info(f"  BRFSS      : {len(brfss_df):,} rows")
    logger.info(f"  Census ACS : {len(census_df):,} rows")

    # ── 2. Clean & validate ───────────────────────────────────────────────────
    logger.info("Step 2/4 — Cleaning and validating...")
    places_df = clean_and_validate(places_df, source="cdc_places")
    brfss_df  = clean_and_validate(brfss_df,  source="brfss")
    census_df = clean_and_validate(census_df, source="census_acs")

    # ── 3. Join on FIPS ───────────────────────────────────────────────────────
    logger.info("Step 3/4 — Joining on FIPS county codes...")
    combined_df = join_on_fips(
        places_df, brfss_df, census_df,
        crosswalk_path=os.getenv("MMSA_CROSSWALK_PATH"),
    )
    logger.info(f"  Combined   : {len(combined_df):,} rows | {combined_df['fips'].nunique():,} counties")

    # ── 3b. Comorbidity scoring ────────────────────────────────────────────────
    logger.info("Step 3b/4 — Building comorbidity index...")
    combined_df = build_comorbid_index(combined_df)
    logger.info(f"  Risk tiers : {combined_df['risk_tier'].value_counts().to_dict()}")

    if dry_run:
        logger.info("[DRY RUN] Skipping S3 write. Sample output:")
        score_cols = ["fips", "county_name", "comorbid_l1_clinical",
                      "comorbid_l2_social", "comorbid_l3_trajectory",
                      "comorbid_index", "risk_tier"]
        print(combined_df[[c for c in score_cols if c in combined_df.columns]].head(10).to_string())
        return combined_df

    # ── 4. Write versioned Parquet to S3 ──────────────────────────────────────
    logger.info("Step 4/4 — Writing versioned Parquet to S3...")
    s3_path = write_versioned_parquet(combined_df, year=year, run_id=run_id)
    logger.info(f"  Written to : {s3_path}")
    logger.info(f"=== Pipeline complete | run_id={run_id} ===")

    return combined_df


if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(year=args.year, dry_run=args.dry_run)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)