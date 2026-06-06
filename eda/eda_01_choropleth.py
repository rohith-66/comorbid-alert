"""
eda_01_choropleth.py
====================
Week 3 EDA — US county choropleth map of comorbidity index.

Usage:
    cd ~/Documents/comorbid_alret
    source .venv/bin/activate
    python eda/eda_01_choropleth.py

Output:
    eda/outputs/choropleth_comorbid_index.html   ← open in browser
    eda/outputs/choropleth_comorbid_index.png    ← for Medium article / README

Dependencies (add to requirements.txt if missing):
    plotly>=5.18  geopandas>=0.14  kaleido>=0.2  boto3  pandas  pyarrow
"""

import io
import logging
import os
import sys
from pathlib import Path

import boto3
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv("/Users/rohithsrinivasa/Documents/comorbid_alret/.env")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
S3_BUCKET = "comorbid-alert-data"
S3_KEY = (
    "comorbid_alert/year=2023/run_id=20260425T220000Z/part-0000.parquet"
)
OUTPUT_DIR = Path(__file__).parent / "outputs"

TIER_COLORS = {
    "Critical": "#d62728",
    "High":     "#ff7f0e",
    "Moderate": "#f7b731",
    "Low":      "#2ca02c",
}
COLORSCALE = [
    [0.00, "#2ca02c"],
    [0.25, "#f7b731"],
    [0.50, "#ff7f0e"],
    [0.75, "#d62728"],
    [1.00, "#7b0000"],
]


# ── Data loading ─────────────────────────────────────────────────────────────
def load_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    log.info("Loading parquet from s3://%s/%s", bucket, key)
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        df = pq.read_table(buf).to_pandas()
        log.info("  Loaded %d rows, %d columns", len(df), df.shape[1])
        return df
    except ClientError as e:
        log.error("S3 error: %s", e)
        sys.exit(1)


def prep_fips(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure fips_code is a zero-padded 5-char string."""
    df = df.copy()
    col = next(
        (c for c in df.columns if "fips" in c.lower()), None
    )
    if col is None:
        raise ValueError("No FIPS column found. Columns: " + str(list(df.columns)))
    df["fips_5"] = df[col].astype(str).str.zfill(5)
    return df


# ── Map building ──────────────────────────────────────────────────────────────
def build_choropleth(df: pd.DataFrame) -> go.Figure:
    log.info("Building choropleth for %d counties", len(df))

    # Hover text — rich tooltip
    def hover(row):
        lines = [
            f"<b>{row.get('county_name', row['fips_5'])}</b>",
            f"FIPS: {row['fips_5']}",
            f"Tier: <b>{row.get('risk_tier', '?')}</b>",
            f"Comorbidity index: {row.get('comorbid_index', 0):.3f}",
            "─────────────",
            f"L1 clinical:    {row.get('l1_clinical_score', 0):.3f}",
            f"L2 social:      {row.get('l2_social_score', 0):.3f}",
            f"L3 trajectory:  {row.get('l3_trajectory_score', 0):.3f}",
        ]
        state = row.get("state_abbr") or row.get("state_name", "")
        if state:
            lines.insert(1, str(state))
        return "<br>".join(lines)

    df["hover_text"] = df.apply(hover, axis=1)

    fig = go.Figure(
        go.Choropleth(
            geojson=(
                "https://raw.githubusercontent.com/plotly/datasets/master/"
                "geojson-counties-fips.json"
            ),
            locations=df["fips_5"],
            z=df["comorbid_index"],
            zmin=0.0,
            zmax=1.0,
            colorscale=COLORSCALE,
            text=df["hover_text"],
            hoverinfo="text",
            marker_line_width=0.2,
            marker_line_color="white",
            colorbar=dict(
                title=dict(text="Comorbidity<br>Index", side="right"),
                thickness=14,
                len=0.6,
                tickvals=[0, 0.25, 0.50, 0.75, 1.0],
                ticktext=["0.00<br>(Low)", "0.25", "0.50<br>(High)", "0.75", "1.00<br>(Critical)"],
                tickfont=dict(size=10),
                outlinewidth=0,
            ),
            name="",
        )
    )

    # ── Annotation: tier counts ──────────────────────────────────────────────
    tier_counts = df["risk_tier"].value_counts() if "risk_tier" in df.columns else {}
    counts_str = "  ".join(
        f"<span style='color:{TIER_COLORS[t]}'><b>{t}</b>: {tier_counts.get(t, 0)}</span>"
        for t in ["Critical", "High", "Moderate", "Low"]
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>US County Diabetes-Cardiac Comorbidity Index</b><br>"
                "<sup>2023 · CDC PLACES + Census ACS · ComorbidAlert</sup>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showlakes=True,
            lakecolor="rgb(240,248,255)",
            showland=True,
            landcolor="rgb(240,240,240)",
            showframe=False,
            coastlinewidth=0.5,
        ),
        annotations=[
            dict(
                x=0.5, y=-0.04,
                xref="paper", yref="paper",
                text=counts_str,
                showarrow=False,
                font=dict(size=12),
                align="center",
            )
        ],
        margin=dict(l=0, r=0, t=70, b=50),
        height=600,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_parquet_from_s3(S3_BUCKET, S3_KEY)
    df = prep_fips(df)

    # Validate required columns
    required = ["comorbid_index"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error("Missing required columns: %s", missing)
        log.error("Available: %s", list(df.columns))
        sys.exit(1)

    fig = build_choropleth(df)

    # ── Save HTML (interactive, shareable) ───────────────────────────────────
    html_path = OUTPUT_DIR / "choropleth_comorbid_index.html"
    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        full_html=True,
        config=dict(displayModeBar=True, scrollZoom=False),
    )
    log.info("HTML saved → %s", html_path)

    # ── Save PNG (static, for README / Medium) ───────────────────────────────
    try:
        png_path = OUTPUT_DIR / "choropleth_comorbid_index.png"
        fig.write_image(str(png_path), width=1400, height=700, scale=2)
        log.info("PNG saved  → %s", png_path)
    except Exception as e:
        log.warning("PNG export failed (install kaleido): %s", e)
        log.warning("  pip install kaleido")

    log.info("Done. Open %s in your browser.", html_path)


if __name__ == "__main__":
    main()