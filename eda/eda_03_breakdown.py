"""
eda_03_breakdown.py
===================
Week 3 EDA — L1/L2 layer breakdown for Critical vs Low counties.

Visualizes *what's driving* the comorbidity index at the extremes:
  - Critical counties: high L1 clinical + high L2 social, or one dominant?
  - Low counties: low across the board, or low L2 masking moderate L1?

Usage:
    cd ~/Documents/comorbid_alret
    source .venv/bin/activate
    python eda/eda_03_breakdown.py

Output:
    eda/outputs/breakdown_l1l2.html   ← main breakdown chart
"""

import io
import logging
from pathlib import Path

import boto3
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

load_dotenv("/Users/rohithsrinivasa/Documents/comorbid_alret/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

S3_BUCKET  = "comorbid-alert-data"
S3_KEY     = "comorbid_alert/year=2023/run_id=20260425T220000Z/part-0000.parquet"
OUTPUT_DIR = Path(__file__).parent / "outputs"

TIER_COLORS = {
    "Critical": "#d62728",
    "High":     "#ff7f0e",
    "Moderate": "#f5c400",
    "Low":      "#2ca02c",
}


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    log.info("Loading parquet from S3...")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    df = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    df["fips_5"] = df["fips"].astype(str).str.zfill(5)
    df["label"] = df["county_name"] + ", " + df["stateabbr"]
    log.info("Loaded %d rows", len(df))
    return df


# ── Chart 1: Scatter — L1 vs L2 colored by tier ──────────────────────────────
def plot_l1_vs_l2_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for tier in ["Low", "Moderate", "High", "Critical"]:
        sub = df[df["risk_tier"] == tier]
        fig.add_trace(go.Scatter(
            x=sub["comorbid_l1_clinical"],
            y=sub["comorbid_l2_social"],
            mode="markers",
            name=f"{tier} ({len(sub):,})",
            marker=dict(
                color=TIER_COLORS[tier],
                size=5 if tier not in ("Critical", "Low") else 9,
                opacity=0.5 if tier not in ("Critical", "Low") else 0.9,
                line=dict(width=0.5, color="white") if tier in ("Critical", "Low") else dict(width=0),
            ),
            text=sub["label"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "L1 clinical: %{x:.3f}<br>"
                "L2 social: %{y:.3f}<br>"
                "<extra></extra>"
            ),
        ))

    # Tier boundary lines
    for val, label in [(0.25, "Low→Moderate"), (0.50, "Moderate→High"), (0.75, "High→Critical")]:
        fig.add_vline(x=val, line_dash="dot", line_color="gray", line_width=0.8,
                      annotation_text=label, annotation_position="top",
                      annotation_font_size=9)

    fig.update_layout(
        title=dict(
            text="<b>L1 Clinical vs L2 Social — by Risk Tier</b><br>"
                 "<sup>Each point = one county · Critical and Low counties enlarged</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        xaxis=dict(title="L1 Clinical Burden (normalized 0–1)", range=[-0.02, 1.02]),
        yaxis=dict(title="L2 Social Vulnerability (normalized 0–1)", range=[-0.02, 1.02]),
        height=480,
        paper_bgcolor="white",
        plot_bgcolor="rgba(245,245,245,1)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.3),
    )
    return fig


# ── Chart 2: Grouped bar — mean L1/L2 + raw indicators by tier ───────────────
def plot_layer_bars(df: pd.DataFrame) -> go.Figure:
    tiers = ["Critical", "High", "Moderate", "Low"]

    # Layer scores
    l1_means = [df[df["risk_tier"] == t]["comorbid_l1_clinical"].mean() for t in tiers]
    l2_means = [df[df["risk_tier"] == t]["comorbid_l2_social"].mean() for t in tiers]
    index_means = [df[df["risk_tier"] == t]["comorbid_index"].mean() for t in tiers]

    # Raw indicators
    raw_features = {
        "Diabetes %":    "places_diabetes",
        "CHD %":         "places_chd",
        "Obesity %":     "places_obesity",
        "Poverty %":     "poverty_rate_pct",
    }

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Mean layer scores by risk tier",
            "Diabetes & CHD prevalence by tier",
            "Obesity prevalence by tier",
            "Poverty rate by tier",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    colors = [TIER_COLORS[t] for t in tiers]

    # ── Top left: layer scores ────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        name="L1 Clinical", x=tiers, y=l1_means,
        marker_color=colors, opacity=1.0,
        text=[f"{v:.3f}" for v in l1_means],
        textposition="outside", textfont_size=10,
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        name="L2 Social", x=tiers, y=l2_means,
        marker_color=colors, opacity=0.5,
        text=[f"{v:.3f}" for v in l2_means],
        textposition="outside", textfont_size=10,
        showlegend=False,
    ), row=1, col=1)

    # ── Top right: diabetes + CHD ─────────────────────────────────────────────
    for feat, col_key, opacity in [
        ("Diabetes %", "places_diabetes", 1.0),
        ("CHD %",      "places_chd",      0.5),
    ]:
        vals = [df[df["risk_tier"] == t][col_key].mean() for t in tiers]
        fig.add_trace(go.Bar(
            name=feat, x=tiers, y=vals,
            marker_color=colors, opacity=opacity,
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside", textfont_size=10,
            showlegend=False,
        ), row=1, col=2)

    # ── Bottom left: obesity ──────────────────────────────────────────────────
    obesity_vals = [df[df["risk_tier"] == t]["places_obesity"].mean() for t in tiers]
    fig.add_trace(go.Bar(
        x=tiers, y=obesity_vals,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in obesity_vals],
        textposition="outside", textfont_size=10,
        showlegend=False,
    ), row=2, col=1)

    # ── Bottom right: poverty ─────────────────────────────────────────────────
    poverty_vals = [df[df["risk_tier"] == t]["poverty_rate_pct"].mean() for t in tiers]
    fig.add_trace(go.Bar(
        x=tiers, y=poverty_vals,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in poverty_vals],
        textposition="outside", textfont_size=10,
        showlegend=False,
    ), row=2, col=2)

    fig.update_layout(
        title=dict(
            text="<b>Layer Breakdown — Critical vs Low Counties</b><br>"
                 "<sup>L1 = clinical burden · L2 = social vulnerability · ComorbidAlert</sup>",
            x=0.5, xanchor="center", font=dict(size=15),
        ),
        barmode="group",
        height=580,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.update_yaxes(range=[0, None])
    return fig


# ── Chart 3: Strip/box — index distribution by tier ──────────────────────────
def plot_index_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    tiers = ["Critical", "High", "Moderate", "Low"]
    for tier in tiers:
        sub = df[df["risk_tier"] == tier]
        fig.add_trace(go.Box(
            y=sub["comorbid_index"],
            name=f"{tier}<br>n={len(sub):,}",
            marker_color=TIER_COLORS[tier],
            boxpoints="outliers",
            line_width=1.5,
            fillcolor=TIER_COLORS[tier],
            opacity=0.7,
        ))

    # Tier boundary lines
    for val in [0.25, 0.50, 0.75]:
        fig.add_hline(y=val, line_dash="dot", line_color="gray", line_width=0.8)

    fig.update_layout(
        title=dict(
            text="<b>Comorbidity Index Distribution by Tier</b><br>"
                 "<sup>Dots = outliers · Dashed lines = tier boundaries</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        yaxis=dict(title="Comorbidity Index", range=[-0.02, 1.02]),
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="rgba(245,245,245,1)",
        showlegend=False,
    )
    return fig


# ── Chart 4: Top/bottom 10 counties ──────────────────────────────────────────
def plot_top_bottom(df: pd.DataFrame) -> go.Figure:
    top10 = df.nlargest(10, "comorbid_index")[["label", "comorbid_index", "comorbid_l1_clinical", "comorbid_l2_social"]].iloc[::-1]
    bot10 = df.nsmallest(10, "comorbid_index")[["label", "comorbid_index", "comorbid_l1_clinical", "comorbid_l2_social"]]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top 10 highest-risk counties", "Top 10 lowest-risk counties"),
        horizontal_spacing=0.18,
    )

    for trace_df, col, color_l1, color_l2 in [
        (top10, 1, "#d62728", "#ff9999"),
        (bot10, 2, "#2ca02c", "#99dd99"),
    ]:
        fig.add_trace(go.Bar(
            y=trace_df["label"],
            x=trace_df["comorbid_l1_clinical"],
            name="L1 Clinical",
            orientation="h",
            marker_color=color_l1,
            showlegend=(col == 1),
        ), row=1, col=col)

        fig.add_trace(go.Bar(
            y=trace_df["label"],
            x=trace_df["comorbid_l2_social"],
            name="L2 Social",
            orientation="h",
            marker_color=color_l2,
            showlegend=(col == 1),
        ), row=1, col=col)

    fig.update_layout(
        title=dict(
            text="<b>L1 + L2 Breakdown — Highest vs Lowest Risk Counties</b><br>"
                 "<sup>Stacked bars show contribution of each layer</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        barmode="stack",
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.35),
    )
    fig.update_xaxes(title_text="Score", row=1, col=1)
    fig.update_xaxes(title_text="Score", row=1, col=2)
    return fig


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    # Print summary stats for Critical vs Low
    for tier in ["Critical", "Low"]:
        sub = df[df["risk_tier"] == tier]
        log.info(
            "\n%s counties (n=%d):\n"
            "  comorbid_index : mean=%.3f  std=%.3f\n"
            "  L1 clinical    : mean=%.3f  std=%.3f\n"
            "  L2 social      : mean=%.3f  std=%.3f\n"
            "  diabetes       : mean=%.1f%%\n"
            "  CHD            : mean=%.1f%%\n"
            "  poverty        : mean=%.1f%%",
            tier, len(sub),
            sub["comorbid_index"].mean(), sub["comorbid_index"].std(),
            sub["comorbid_l1_clinical"].mean(), sub["comorbid_l1_clinical"].std(),
            sub["comorbid_l2_social"].mean(), sub["comorbid_l2_social"].std(),
            sub["places_diabetes"].mean(),
            sub["places_chd"].mean(),
            sub["poverty_rate_pct"].mean(),
        )

    # Top 10 Critical counties
    log.info("\nTop 10 Critical counties:")
    top = df[df["risk_tier"] == "Critical"].nlargest(10, "comorbid_index")[
        ["label", "comorbid_index", "comorbid_l1_clinical", "comorbid_l2_social"]
    ]
    log.info("\n%s", top.to_string(index=False))

    # Build and save charts
    charts = [
        ("breakdown_scatter.html",      plot_l1_vs_l2_scatter(df)),
        ("breakdown_layer_bars.html",   plot_layer_bars(df)),
        ("breakdown_distribution.html", plot_index_distribution(df)),
        ("breakdown_top_bottom.html",   plot_top_bottom(df)),
    ]

    for filename, fig in charts:
        path = OUTPUT_DIR / filename
        fig.write_html(str(path), include_plotlyjs="cdn")
        log.info("Saved → %s", path)

    log.info("\nDone. Open these in your browser:")
    for filename, _ in charts:
        log.info("  eda/outputs/%s", filename)


if __name__ == "__main__":
    main()