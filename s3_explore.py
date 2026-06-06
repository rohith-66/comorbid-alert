"""
s3_inspect.py — inspect the three key parquet files before running Week 5
"""
import sys, io
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from aws_session import get_s3_client

s3     = get_s3_client()
BUCKET = "comorbid-alert-data"

KEYS = [
    "comorbid_alert/panel/comorbid_panel.parquet",
    "comorbid_alert/panel/forecast_results.parquet",
    "comorbid_alert/panel/lgbm_forecast.parquet",
]

for key in KEYS:
    print(f"\n{'='*70}")
    print(f"  {key}")
    print(f"{'='*70}")
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df  = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    print(f"  Shape  : {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Dtypes :\n{df.dtypes.to_string()}")
    print(f"\n  Sample (first 3 rows):")
    print(df.head(3).to_string())
    print(f"\n  Numeric summary:")
    print(df.describe(include='number').to_string())
    if 'year' in df.columns:
        print(f"\n  Years present: {sorted(df['year'].unique())}")