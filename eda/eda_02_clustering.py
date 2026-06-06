"""
eda_02_clustering.py
====================
Week 3 EDA — KMeans county clustering on raw feature space.

Clusters counties by *why* they're high risk, not just *that* they are.
Features: clinical burden, social vulnerability, specific CDC PLACES measures,
          poverty, uninsured rate, obesity.

Usage:
    cd ~/Documents/comorbid_alret
    source .venv/bin/activate
    python eda/eda_02_clustering.py

Output:
    eda/outputs/cluster_map.html          ← choropleth colored by cluster
    eda/outputs/cluster_profiles.html     ← radar/bar chart of cluster profiles
    eda/outputs/cluster_summary.csv       ← cluster means for each feature

Dependencies:
    pip install plotly scikit-learn pandas pyarrow boto3 python-dotenv
"""

import io
import logging
import os
from pathlib import Path

import boto3
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dotenv import load_dotenv

load_dotenv("/Users/rohithsrinivasa/Documents/comorbid_alret/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET  = "comorbid-alert-data"
S3_KEY     = "comorbid_alert/year=2023/run_id=20260425T220000Z/part-0000.parquet"
OUTPUT_DIR = Path(__file__).parent / "outputs"
N_CLUSTERS = 5
RANDOM_STATE = 42

# Features to cluster on — raw indicators, not derived scores
CLUSTER_FEATURES = [
    "places_diabetes",       # diabetes prevalence
    "places_chd",            # coronary heart disease
    "places_obesity",        # obesity prevalence
    "places_stroke",         # stroke prevalence
    "places_bphigh",         # high blood pressure
    "poverty_rate_pct",      # poverty rate
    "unemployment_rate_pct", # unemployment rate
    "hs_diploma_rate_pct",   # education (inverse proxy)
]

# Human-readable cluster names — assigned after inspecting profiles
CLUSTER_NAMES = {
    0: "Suburban Moderate",
    1: "Deep South High Burden",
    2: "Mid-Burden Low Education",
    3: "Healthy Baseline",
    4: "Moderate — Structurally Vulnerable",
}

# Colors for each cluster
CLUSTER_COLORS = {
    0: "#f5a623",  # amber — moderate
    1: "#d62728",  # red — high burden
    2: "#ff7f0e",  # orange — elevated
    3: "#2ca02c",  # green — healthy
    4: "#e377c2",  # pink — vulnerable
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    log.info("Loading parquet from S3...")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    df = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    log.info("  Loaded %d rows, %d columns", len(df), df.shape[1])
    df["fips_5"] = df["fips"].astype(str).str.zfill(5)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame):
    available = [f for f in CLUSTER_FEATURES if f in df.columns]
    missing = [f for f in CLUSTER_FEATURES if f not in df.columns]
    if missing:
        log.warning("Missing features (will skip): %s", missing)

    X_raw = df[available].copy()

    # Impute with median — don't drop counties with partial data
    for col in available:
        median = X_raw[col].median()
        n_missing = X_raw[col].isna().sum()
        if n_missing > 0:
            log.info("  Imputing %d missing values in '%s' with median %.2f",
                     n_missing, col, median)
        X_raw[col] = X_raw[col].fillna(median)

    log.info("Feature matrix: %d counties × %d features", *X_raw.shape)
    return X_raw, available


# ── Clustering ────────────────────────────────────────────────────────────────
def run_kmeans(X_raw: pd.DataFrame, n_clusters: int = N_CLUSTERS):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Elbow check — log inertia for k=2..8
    log.info("Elbow inertias:")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        log.info("  k=%d → inertia=%.1f", k, km.inertia_)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X_scaled)
    log.info("KMeans k=%d | inertia=%.1f", n_clusters, km.inertia_)

    return labels, X_scaled, scaler


# ── Cluster profiles ──────────────────────────────────────────────────────────
def build_profiles(df: pd.DataFrame, features: list) -> pd.DataFrame:
    profile = df.groupby("cluster")[features + ["comorbid_index", "risk_tier"]].agg(
        {**{f: "mean" for f in features}, "comorbid_index": "mean",
         "risk_tier": lambda x: x.value_counts().index[0]}
    ).round(3)
    profile["n_counties"] = df.groupby("cluster").size()

    # Sort by mean comorbid_index descending
    profile = profile.sort_values("comorbid_index", ascending=False)
    log.info("Cluster profiles:\n%s", profile[["n_counties", "comorbid_index", "risk_tier"]])
    return profile


# ── Visualizations ────────────────────────────────────────────────────────────
def plot_cluster_map(df: pd.DataFrame) -> go.Figure:
    log.info("Building cluster choropleth...")

    df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)
    df["cluster_color_val"] = df["cluster"].astype(float)

    hover_text = df.apply(lambda r: (
        f"<b>{r.get('county_name', r['fips_5'])}</b><br>"
        f"{r.get('stateabbr', '')}<br>"
        f"Cluster: <b>{CLUSTER_NAMES[r['cluster']]}</b><br>"
        f"Risk tier: {r['risk_tier']}<br>"
        f"Comorbidity index: {r['comorbid_index']:.3f}<br>"
        f"──────────<br>"
        f"Diabetes: {r.get('places_diabetes', 'N/A')}%<br>"
        f"CHD: {r.get('places_chd', 'N/A')}%<br>"
        f"Obesity: {r.get('places_obesity', 'N/A')}%<br>"
        f"Poverty: {r.get('poverty_rate_pct', 'N/A'):.1f}%"
    ), axis=1)

    colorscale = [[i / (N_CLUSTERS - 1), CLUSTER_COLORS[i]]
                  for i in range(N_CLUSTERS)]

    fig = go.Figure(go.Choropleth(
        geojson=(
            "https://raw.githubusercontent.com/plotly/datasets/master/"
            "geojson-counties-fips.json"
        ),
        locations=df["fips_5"],
        z=df["cluster_color_val"],
        colorscale=colorscale,
        zmin=0,
        zmax=N_CLUSTERS - 1,
        text=hover_text,
        hoverinfo="text",
        marker_line_width=0.2,
        marker_line_color="white",
        showscale=False,
        name="",
    ))

    # Custom legend as annotations
    legend_items = "  ".join(
        f"<span style='color:{CLUSTER_COLORS[i]}'><b>{CLUSTER_NAMES[i]}</b></span>"
        for i in range(N_CLUSTERS)
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>US County Clustering — Structural Health Profiles</b><br>"
                "<sup>KMeans k=5 · CDC PLACES + Census ACS · ComorbidAlert</sup>"
            ),
            x=0.5, xanchor="center", font=dict(size=16),
        ),
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showlakes=True,
            lakecolor="rgb(240,248,255)",
            showland=True,
            landcolor="rgb(240,240,240)",
            showframe=False,
        ),
        annotations=[dict(
            x=0.5, y=-0.04,
            xref="paper", yref="paper",
            text=legend_items,
            showarrow=False,
            font=dict(size=12),
            align="center",
        )],
        margin=dict(l=0, r=0, t=70, b=50),
        height=600,
        paper_bgcolor="white",
    )
    return fig


def plot_cluster_profiles(df: pd.DataFrame, features: list) -> go.Figure:
    log.info("Building cluster profile chart...")

    # Normalize feature means to 0-1 for radar readability
    profile_means = df.groupby("cluster")[features].mean()
    profile_norm = (profile_means - profile_means.min()) / \
                   (profile_means.max() - profile_means.min())

    # Clean feature labels
    labels = [f.replace("places_", "").replace("_pct", " %")
               .replace("_rate", " rate").replace("_", " ").title()
              for f in features]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Feature profiles by cluster (normalized)",
                        "Mean comorbidity index by cluster"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        column_widths=[0.65, 0.35],
    )

    # Grouped bar — feature profiles
    for cluster_id in sorted(df["cluster"].unique()):
        vals = profile_norm.loc[cluster_id].tolist()
        fig.add_trace(go.Bar(
            name=CLUSTER_NAMES[cluster_id],
            x=labels,
            y=vals,
            marker_color=CLUSTER_COLORS[cluster_id],
            legendgroup=str(cluster_id),
            showlegend=True,
        ), row=1, col=1)

    # Mean comorbidity index per cluster
    mean_index = df.groupby("cluster")["comorbid_index"].mean().sort_values(ascending=True)
    n_counties = df.groupby("cluster").size()

    fig.add_trace(go.Bar(
        x=mean_index.values,
        y=[f"{CLUSTER_NAMES[i]}<br>n={n_counties[i]:,}" for i in mean_index.index],
        orientation="h",
        marker_color=[CLUSTER_COLORS[i] for i in mean_index.index],
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text="<b>Cluster Health Profiles</b><br>"
                 "<sup>ComorbidAlert Week 3 EDA</sup>",
            x=0.5, xanchor="center", font=dict(size=15),
        ),
        barmode="group",
        height=480,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.3),
    )
    fig.update_xaxes(tickangle=-35, row=1, col=1)
    fig.update_xaxes(title_text="Mean comorbidity index", row=1, col=2)

    return fig


def plot_pca_scatter(df: pd.DataFrame, X_scaled: np.ndarray) -> go.Figure:
    log.info("Building PCA scatter...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    var1, var2 = pca.explained_variance_ratio_ * 100

    fig = go.Figure()
    for cluster_id in sorted(df["cluster"].unique()):
        mask = df["cluster"] == cluster_id
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            name=CLUSTER_NAMES[cluster_id],
            marker=dict(
                color=CLUSTER_COLORS[cluster_id],
                size=4,
                opacity=0.6,
            ),
            text=df[mask].apply(
                lambda r: f"{r.get('county_name','?')}, {r.get('stateabbr','')}<br>"
                          f"Index: {r['comorbid_index']:.3f}", axis=1
            ),
            hoverinfo="text",
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>PCA — County Cluster Separation</b><br>"
                 f"<sup>PC1 {var1:.1f}% variance · PC2 {var2:.1f}% variance</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        xaxis_title=f"PC1 ({var1:.1f}% variance)",
        yaxis_title=f"PC2 ({var2:.1f}% variance)",
        height=450,
        paper_bgcolor="white",
        plot_bgcolor="rgba(245,245,245,1)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.3),
    )
    return fig


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_raw, features = build_feature_matrix(df)
    labels, X_scaled, scaler = run_kmeans(X_raw)

    df["cluster"] = labels
    profile = build_profiles(df, features)

    # Save cluster summary CSV
    csv_path = OUTPUT_DIR / "cluster_summary.csv"
    profile.to_csv(csv_path)
    log.info("Cluster summary saved → %s", csv_path)

    # Print top counties per cluster for insight
    log.info("\n── Top 3 Critical counties per cluster ──")
    critical = df[df["risk_tier"] == "Critical"]
    for cid in sorted(df["cluster"].unique()):
        subset = critical[critical["cluster"] == cid]
        if len(subset):
            names = subset.nlargest(3, "comorbid_index")[["county_name", "stateabbr", "comorbid_index"]]
            log.info("Cluster %d: %s", cid, names.to_dict("records"))

    # Plots
    fig_map = plot_cluster_map(df)
    fig_profiles = plot_cluster_profiles(df, features)
    fig_pca = plot_pca_scatter(df, X_scaled)

    map_path = OUTPUT_DIR / "cluster_map.html"
    fig_map.write_html(str(map_path), include_plotlyjs="cdn")
    log.info("Cluster map saved → %s", map_path)

    profiles_path = OUTPUT_DIR / "cluster_profiles.html"
    fig_profiles.write_html(str(profiles_path), include_plotlyjs="cdn")
    log.info("Cluster profiles saved → %s", profiles_path)

    pca_path = OUTPUT_DIR / "cluster_pca.html"
    fig_pca.write_html(str(pca_path), include_plotlyjs="cdn")
    log.info("PCA scatter saved → %s", pca_path)

    log.info("\nDone. Open these in your browser:")
    log.info("  %s", map_path)
    log.info("  %s", profiles_path)
    log.info("  %s", pca_path)


if __name__ == "__main__":
    main()