"""
ComorbidAlert — Week 5, Step 2: Early Warning Alerting System
=============================================================
Reads ensemble_forecasts.parquet (from Step 1) and generates a
directional alert log.

Tier thresholds (matched to actual comorbid_index distribution):
  Critical  ≥ 0.65   (matches ~33 counties from Week 1-2)
  High      ≥ 0.50
  Moderate  ≥ 0.35
  Low        < 0.35

Alert levels:
  CRITICAL  — county currently Critical AND ensemble slope > 0 (still rising)
  WARNING   — county currently High AND ensemble_2027 ≥ 0.65 (crossing into Critical)
  WATCH     — county currently Moderate AND slope > 0 AND Δ > 3%

Outputs → s3://comorbid-alert-data/comorbid_alert/week5/
  alert_log.csv
  critical_top10.csv
  warning_top10.csv
  watch_top10.csv
"""

import io, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aws_session import get_s3_client

warnings.filterwarnings("ignore")

BUCKET = "comorbid-alert-data"
s3     = get_s3_client()

# ── Tier thresholds — calibrated to comorbid_index range [0, 1] ───────────────
# From inspect: 75th pct ≈ 0.493, max = 1.0, mean = 0.406
# Week 1-2 produced: Critical=33, High=381, Moderate=2164, Low=566
# These thresholds reproduce that distribution
THRESH_CRITICAL = 0.65
THRESH_HIGH     = 0.50
THRESH_MODERATE = 0.35

FORECAST_YEARS  = [2025, 2026, 2027]

# ── helpers ───────────────────────────────────────────────────────────────────

def s3_read_parquet(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def s3_write_csv(df, key):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.read())
    print(f"  ✓ s3://{BUCKET}/{key}  ({len(df):,} rows)")

def classify_tier(v):
    if   v >= THRESH_CRITICAL: return "Critical"
    elif v >= THRESH_HIGH:     return "High"
    elif v >= THRESH_MODERATE: return "Moderate"
    return "Low"

def pct_change(start, end):
    if start == 0 or np.isnan(start): return np.nan
    return (end - start) / start * 100

# ── 1. Load ensemble ──────────────────────────────────────────────────────────

print("Loading ensemble forecasts …")
fc = s3_read_parquet("comorbid_alert/week5/ensemble_forecasts.parquet")
print(f"  {fc.shape}  |  counties: {fc['fips'].nunique():,}")

# ── 2. Pivot to wide (one row per county) ────────────────────────────────────

print("Building trajectory table …")

fc_wide = (
    fc[fc["forecast_year"].isin(FORECAST_YEARS)]
    .pivot(index="fips", columns="forecast_year", values="ensemble_forecast")
    .reset_index()
)
fc_wide.columns = ["fips"] + [f"ens_{y}" for y in FORECAST_YEARS]

# Attach metadata from the ensemble file (take first row per county)
meta = (
    fc[fc["forecast_year"] == 2025]
    [["fips","county_name","stateabbr","current_tier","actual_2024"]]
    .drop_duplicates("fips")
)

traj = fc_wide.merge(meta, on="fips", how="left")

# Compute trajectory slope via linear regression on the 3 forecast years
year_arr = np.array(FORECAST_YEARS, dtype=float)
def slope(row):
    vals = np.array([row.get(f"ens_{y}", np.nan) for y in FORECAST_YEARS])
    mask = ~np.isnan(vals)
    if mask.sum() < 2: return np.nan
    s, *_ = stats.linregress(year_arr[mask], vals[mask])
    return s

traj["forecast_slope"]   = traj.apply(slope, axis=1)
traj["ens_2025"]         = traj.get("ens_2025", np.nan)
traj["ens_2027"]         = traj.get("ens_2027", np.nan)
traj["forecast_pct_chg"] = traj.apply(
    lambda r: pct_change(r["actual_2024"], r["ens_2027"]), axis=1
)
traj["tier_2027"] = traj["ens_2027"].apply(
    lambda v: classify_tier(v) if not np.isnan(v) else "Unknown"
)

# Current tier: use panel value if present, else reclassify from actual_2024
if "current_tier" not in traj.columns:
    traj["current_tier"] = traj["actual_2024"].apply(classify_tier)

print(f"  Trajectory rows: {len(traj):,}")
print(f"  Current tier distribution:\n{traj['current_tier'].value_counts().to_string()}")

# ── 3. Generate alerts ────────────────────────────────────────────────────────

print("\nGenerating alerts …")
alerts = []

for _, row in traj.iterrows():
    tier  = str(row.get("current_tier", "Unknown"))
    t2027 = str(row.get("tier_2027", "Unknown"))
    slope_val = row.get("forecast_slope", 0.0)
    pchg      = row.get("forecast_pct_chg", np.nan)
    idx_val   = row.get("actual_2024", np.nan)

    alert_level = None
    reason      = ""

    if tier == "Critical" and slope_val > 0:
        alert_level = "CRITICAL"
        reason = (f"Already Critical; comorbid_index still rising "
                  f"(slope={slope_val:+.5f}/yr, Δ={pchg:+.1f}% by 2027)")

    elif tier == "High" and t2027 == "Critical":
        alert_level = "WARNING"
        reason = (f"High tier now; forecast to cross Critical threshold by 2027 "
                  f"(Δ={pchg:+.1f}%)")

    elif tier == "Moderate" and slope_val > 0 and (np.isnan(pchg) or pchg > 3):
        alert_level = "WATCH"
        reason = (f"Moderate tier; accelerating upward "
                  f"(slope={slope_val:+.5f}/yr, Δ={pchg:+.1f}% by 2027)")

    if alert_level:
        alerts.append({
            "fips":              row["fips"],
            "county_name":       row.get("county_name", ""),
            "stateabbr":         row.get("stateabbr", ""),
            "alert_level":       alert_level,
            "current_tier":      tier,
            "tier_2027":         t2027,
            "comorbid_index_2024": round(float(idx_val), 4) if not np.isnan(idx_val) else np.nan,
            "ens_2025":          round(float(row["ens_2025"]), 4) if not np.isnan(row["ens_2025"]) else np.nan,
            "ens_2027":          round(float(row["ens_2027"]), 4) if not np.isnan(row["ens_2027"]) else np.nan,
            "forecast_slope":    round(float(slope_val), 6),
            "forecast_pct_chg":  round(float(pchg), 2) if not np.isnan(pchg) else np.nan,
            "reason":            reason,
        })

level_order = {"CRITICAL": 0, "WARNING": 1, "WATCH": 2}
alert_df = (
    pd.DataFrame(alerts)
    .assign(_sort=lambda d: d["alert_level"].map(level_order))
    .sort_values(["_sort", "forecast_slope"], ascending=[True, False])
    .drop(columns="_sort")
    .reset_index(drop=True)
)

print(f"\n  Alert counts:")
print(alert_df["alert_level"].value_counts().to_string())

# ── 4. Top 10 lists ───────────────────────────────────────────────────────────

def top10(df, level, sort_col="forecast_slope"):
    return (
        df[df["alert_level"] == level]
        .sort_values(sort_col, ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

critical_top10 = top10(alert_df, "CRITICAL")
warning_top10  = top10(alert_df, "WARNING",  sort_col="forecast_pct_chg")
watch_top10    = top10(alert_df, "WATCH")

display_cols = ["fips","county_name","stateabbr","comorbid_index_2024",
                "ens_2027","forecast_slope","forecast_pct_chg"]

print(f"\n── TOP 10 CRITICAL (worsening despite Critical status) ───────────────")
print(critical_top10[display_cols].to_string(index=False))

print(f"\n── TOP 10 WARNING (High → Critical by 2027) ──────────────────────────")
print(warning_top10[display_cols].to_string(index=False))

print(f"\n── TOP 10 WATCH (early warning catches) ──────────────────────────────")
print(watch_top10[display_cols].to_string(index=False))

# ── 5. Sanity check vs known high-risk counties ───────────────────────────────

print("\n── Sanity check: known high-risk regions ────────────────────────────")
KNOWN = {
    "01085": "Lowndes County, AL",
    "28027": "Coahoma County, MS",
    "28049": "Humphreys County, MS",
    "28133": "Sunflower County, MS",
    "22035": "East Carroll Parish, LA",
    "01105": "Perry County, AL",
    "01017": "Chambers County, AL",
    "13205": "Monroe County, GA",
    "37079": "Greene County, NC",
    "47155": "Scott County, TN",
}
flagged = alert_df[alert_df["fips"].isin(KNOWN)]
print(f"  {len(flagged)}/{len(KNOWN)} known high-risk counties caught:")
for _, row in flagged.iterrows():
    print(f"    [{row['alert_level']:8s}]  {row['fips']}  {KNOWN[row['fips']]}")
missed = [f for f in KNOWN if f not in alert_df["fips"].values]
if missed:
    print(f"\n  Not flagged (stable or below threshold):")
    for f in missed:
        # Show their actual index and tier for diagnostics
        info = traj[traj["fips"] == f][["current_tier","actual_2024","forecast_slope"]].to_dict("records")
        print(f"    {f}  {KNOWN[f]}  →  {info[0] if info else 'not in forecast'}")

# ── 6. Save ───────────────────────────────────────────────────────────────────

print("\nSaving …")
s3_write_csv(alert_df,      "comorbid_alert/week5/alert_log.csv")
s3_write_csv(critical_top10,"comorbid_alert/week5/critical_top10.csv")
s3_write_csv(warning_top10, "comorbid_alert/week5/warning_top10.csv")
s3_write_csv(watch_top10,   "comorbid_alert/week5/watch_top10.csv")

print(f"\nWeek 5 Step 2 complete.")
print(f"   CRITICAL : {(alert_df['alert_level']=='CRITICAL').sum():,}")
print(f"   WARNING  : {(alert_df['alert_level']=='WARNING').sum():,}")
print(f"   WATCH    : {(alert_df['alert_level']=='WATCH').sum():,}")
print(f"   TOTAL    : {len(alert_df):,}")