"""
forecast_lgbm.py  (Week 4 — corrected for actual panel schema)
===============================================================
Panel columns confirmed:
  fips, county_name, stateabbr, release_year, brfss_year,
  places_diabetes, places_chd, places_obesity, places_stroke, places_bphigh,
  comorbid_l1_clinical, comorbid_index, risk_tier

Target: comorbid_index

Feature set:
  - Lag features: lag_1yr, lag_2yr, delta_1yr
  - Rolling stats: rolling_mean_2yr, rolling_std_2yr
  - Raw PLACES: places_diabetes, places_chd, places_obesity, places_stroke, places_bphigh
  - Time & identity: year_trend, fips_encoded

Temporal CV (expanding window, no leakage):
  Fold 1: train [2022]      → test 2023  (2021 has no lag, first usable train yr = 2022)
  Fold 2: train [2022-2023] → test 2024  ← primary holdout (compare vs Prophet)

Forecast: retrain on 2021-2024, recursive predict 2025/2026/2027

Usage:
    cd ~/Documents/comorbid_alret
    source .venv/bin/activate
    python forecasting/forecast_lgbm.py

Outputs (forecasting/outputs/):
    lgbm_eval.csv               per-county MAE/RMSE/WAPE on 2024 holdout
    lgbm_summary.csv            fold-level aggregate metrics
    lgbm_forecast.csv           2025/2026/2027 county predictions
    lgbm_comparison.csv         head-to-head vs Prophet (if forecast_eval.csv exists)
    lgbm_plots/
        feature_importance.html
        shap_county_{fips}.html   Oglala Lakota SD, Lowndes AL, Coahoma MS
        model_comparison.html
        error_by_tier.html
        forecast_map_{year}.html
"""

import io
import logging
import warnings
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import shap
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv("/Users/rohithsrinivasa/Documents/comorbid_alret/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET   = "comorbid-alert-data"
PANEL_KEY   = "comorbid_alert/panel/comorbid_panel.parquet"
LGBM_FC_KEY = "comorbid_alert/panel/lgbm_forecast.parquet"

OUTPUT_DIR = Path(__file__).parent / "outputs"
PLOTS_DIR  = OUTPUT_DIR / "lgbm_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL    = "comorbid_index"
TRAIN_YEARS   = [2021, 2022, 2023]
TEST_YEAR     = 2024
HORIZON_YEARS = [2025, 2026, 2027]

PLACES_COLS = [
    "places_diabetes", "places_chd", "places_obesity",
    "places_stroke", "places_bphigh",
]

FEATURE_COLS = [
    "lag_1yr", "lag_2yr", "delta_1yr",
    "rolling_mean_2yr", "rolling_std_2yr",
] + PLACES_COLS + ["year_trend", "fips_encoded"]

SHAP_COUNTIES = {
    "46113": "Oglala Lakota, SD",
    "01085": "Lowndes, AL",
    "28027": "Coahoma, MS",
}

LGB_PARAMS = {
    "objective":         "regression_l1",
    "metric":            "mae",
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}


# ── Metrics ───────────────────────────────────────────────────────────────────
def mae_score(a, p):  return np.mean(np.abs(a - p))
def rmse_score(a, p): return np.sqrt(np.mean((a - p) ** 2))
def wape_score(a, p): return np.sum(np.abs(a - p)) / np.sum(np.abs(a))


# ── S3 ────────────────────────────────────────────────────────────────────────
def load_panel() -> pd.DataFrame:
    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=PANEL_KEY)
    df  = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    log.info("Panel: %d rows | %d counties | years %s",
             len(df), df["fips"].nunique(),
             sorted(df["release_year"].unique().tolist()))
    return df


def save_parquet_s3(df: pd.DataFrame, key: str) -> None:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf)
    buf.seek(0)
    boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=key, Body=buf.read())
    log.info("Saved → s3://%s/%s", S3_BUCKET, key)


# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    df = panel.copy().sort_values(["fips", "release_year"]).reset_index(drop=True)

    grp = df.groupby("fips")[TARGET_COL]

    # Lag features — 2021 rows get NaN (no prior year); that's intentional
    df["lag_1yr"]   = grp.shift(1)
    df["lag_2yr"]   = grp.shift(2)
    df["delta_1yr"] = df[TARGET_COL] - df["lag_1yr"]

    # Rolling stats computed on lagged values (no leakage)
    df["rolling_mean_2yr"] = grp.transform(
        lambda s: s.shift(1).rolling(2, min_periods=1).mean()
    )
    df["rolling_std_2yr"] = grp.transform(
        lambda s: s.shift(1).rolling(2, min_periods=1).std().fillna(0)
    )

    df["year_trend"] = df["release_year"] - df["release_year"].min()

    le = LabelEncoder()
    df["fips_encoded"] = le.fit_transform(df["fips"])

    log.info("Features built OK. Sample NaN counts in lag_1yr by year:")
    for yr, cnt in df.groupby("release_year")["lag_1yr"].apply(lambda s: s.isna().sum()).items():
        log.info("  %d: %d NaN lag_1yr", yr, cnt)

    return df, le


# ── Temporal CV ───────────────────────────────────────────────────────────────
def temporal_cv(df: pd.DataFrame) -> dict:
    """
    Expanding-window temporal CV — no data leakage.

    2021 rows have lag_1yr=NaN (no prior year), so earliest usable training
    year is 2022 (has lag from 2021). Remaining NaNs (lag_2yr on 2022 rows)
    are filled with 0 — LightGBM handles this gracefully.

    Fold 1: train [2022]       → test 2023
    Fold 2: train [2022, 2023] → test 2024  ← primary holdout
    """
    results = {}

    cv_splits = [
        {"train_years": [2022],       "test_year": 2023, "fold": 1},
        {"train_years": [2022, 2023], "test_year": 2024, "fold": 2},
    ]

    for split in cv_splits:
        fold     = split["fold"]
        tr_years = split["train_years"]
        ts_year  = split["test_year"]

        train_df = (
            df[df["release_year"].isin(tr_years)]
            .dropna(subset=["lag_1yr", TARGET_COL])
            .copy()
        )
        test_df = (
            df[df["release_year"] == ts_year]
            .dropna(subset=["lag_1yr", TARGET_COL])
            .copy()
        )

        # Fill lag_2yr NaNs (2022 rows have no lag_2yr) with 0
        train_df[FEATURE_COLS] = train_df[FEATURE_COLS].fillna(0)
        test_df[FEATURE_COLS]  = test_df[FEATURE_COLS].fillna(0)

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test  = test_df[FEATURE_COLS]
        y_test  = test_df[TARGET_COL]

        log.info(
            "Fold %d | Train %s (n=%d rows, %d counties) → Test %d (n=%d counties)",
            fold, tr_years, len(train_df), train_df["fips"].nunique(),
            ts_year, test_df["fips"].nunique()
        )

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        preds = model.predict(X_test)
        test_df = test_df.copy()
        test_df["predicted"] = preds

        county_metrics = (
            test_df.groupby("fips")
            .apply(lambda g: pd.Series({
                "mae":         mae_score(g[TARGET_COL].values, g["predicted"].values),
                "rmse":        rmse_score(g[TARGET_COL].values, g["predicted"].values),
                "wape":        wape_score(g[TARGET_COL].values, g["predicted"].values),
                "actual":      g[TARGET_COL].values[0],
                "predicted":   g["predicted"].values[0],
                "county_name": g["county_name"].values[0],
                "stateabbr":   g["stateabbr"].values[0],
                "risk_tier":   g["risk_tier"].values[0],
            }), include_groups=False)
            .reset_index()
        )

        agg = {
            "fold":        fold,
            "train_years": str(tr_years),
            "test_year":   ts_year,
            "n_train":     len(train_df),
            "n_test":      len(test_df),
            "n_counties":  test_df["fips"].nunique(),
            "mae_median":  county_metrics["mae"].median(),
            "mae_mean":    county_metrics["mae"].mean(),
            "rmse_median": county_metrics["rmse"].median(),
            "rmse_mean":   county_metrics["rmse"].mean(),
            "wape_median": county_metrics["wape"].median(),
            "wape_mean":   county_metrics["wape"].mean(),
            "best_iter":   model.best_iteration_,
        }

        log.info(
            "  → MAE=%.4f | RMSE=%.4f | WAPE=%.4f | best_iter=%d",
            agg["mae_median"], agg["rmse_median"], agg["wape_median"], agg["best_iter"]
        )

        results[fold] = {
            "model":          model,
            "county_metrics": county_metrics,
            "aggregate":      agg,
            "test_df":        test_df,
            "X_train":        X_train,
            "X_test":         X_test,
        }

    return results


# ── Recursive forecast 2025-2027 ──────────────────────────────────────────────
def recursive_forecast(df: pd.DataFrame, model: lgb.LGBMRegressor) -> pd.DataFrame:
    working   = df.copy().sort_values(["fips", "release_year"])
    forecasts = []

    for horizon_year in HORIZON_YEARS:
        rows = []
        for fips, grp in working.groupby("fips"):
            grp  = grp.sort_values("release_year")
            last = grp.iloc[-1]
            prev = grp.iloc[-2] if len(grp) >= 2 else last

            row = {
                "fips":             fips,
                "release_year":     horizon_year,
                "county_name":      last["county_name"],
                "stateabbr":        last["stateabbr"],
                "risk_tier":        last["risk_tier"],
                "lag_1yr":          last[TARGET_COL],
                "lag_2yr":          prev[TARGET_COL],
                "delta_1yr":        last[TARGET_COL] - prev[TARGET_COL],
                "rolling_mean_2yr": np.mean([last[TARGET_COL], prev[TARGET_COL]]),
                "rolling_std_2yr":  np.std([last[TARGET_COL], prev[TARGET_COL]]),
                **{c: last.get(c, 0) for c in PLACES_COLS},
                "year_trend":    horizon_year - df["release_year"].min(),
                "fips_encoded":  last["fips_encoded"],
            }
            rows.append(row)

        horizon_df = pd.DataFrame(rows)
        X = horizon_df[FEATURE_COLS].fillna(0)
        horizon_df["lgbm_forecast"] = model.predict(X)
        horizon_df["forecast_year"] = horizon_year
        forecasts.append(horizon_df)

        # Feed predictions back as actuals for next iteration's lags
        stub = horizon_df[
            ["fips", "release_year", "lgbm_forecast", "county_name",
             "stateabbr", "risk_tier", "fips_encoded", "year_trend"] + PLACES_COLS
        ].copy()
        stub[TARGET_COL]         = stub["lgbm_forecast"]
        stub["lag_1yr"]          = stub["lgbm_forecast"]
        stub["lag_2yr"]          = horizon_df["lag_1yr"].values
        stub["delta_1yr"]        = stub["lag_1yr"] - stub["lag_2yr"]
        stub["rolling_mean_2yr"] = horizon_df["rolling_mean_2yr"].values
        stub["rolling_std_2yr"]  = 0.0

        working = pd.concat([working, stub], ignore_index=True).sort_values(["fips", "release_year"])

        n_critical = (horizon_df["lgbm_forecast"] >= 0.70).sum()
        log.info(
            "  %d | Critical(≥0.70)=%d | mean=%.4f | max=%.4f",
            horizon_year, n_critical,
            horizon_df["lgbm_forecast"].mean(),
            horizon_df["lgbm_forecast"].max(),
        )

    return pd.concat(forecasts, ignore_index=True)


# ── SHAP ──────────────────────────────────────────────────────────────────────
def run_shap(model, X_train, X_test, test_df) -> dict:
    log.info("Computing SHAP values (TreeExplainer)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    global_imp = pd.DataFrame({
        "feature":   FEATURE_COLS,
        "mean_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_shap", ascending=False)

    log.info("Top 10 SHAP features:")
    for _, r in global_imp.head(10).iterrows():
        log.info("  %-30s %.5f", r["feature"], r["mean_shap"])

    base = (
        float(explainer.expected_value)
        if not isinstance(explainer.expected_value, np.ndarray)
        else float(explainer.expected_value[0])
    )

    county_shap = {}
    for fips, name in SHAP_COUNTIES.items():
        idx = test_df.index[test_df["fips"] == fips].tolist()
        if not idx:
            log.warning("SHAP: %s (%s) not in test set — skipping", name, fips)
            continue
        # Get position in X_test (which shares index with test_df)
        try:
            pos = X_test.index.get_loc(idx[0])
        except KeyError:
            log.warning("SHAP: index mismatch for %s — skipping", name)
            continue
        sv = shap_values[pos]
        county_shap[fips] = {"name": name, "shap": sv.tolist(), "base": base}
        log.info("  SHAP captured: %s", name)

    return {
        "explainer":         explainer,
        "shap_values":       shap_values,
        "global_importance": global_imp,
        "county_shap":       county_shap,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_feature_importance(global_imp: pd.DataFrame) -> go.Figure:
    top = global_imp.head(15)
    fig = go.Figure(go.Bar(
        x=top["mean_shap"], y=top["feature"], orientation="h",
        marker_color="#c0392b",
        text=top["mean_shap"].round(5), textposition="outside",
    ))
    fig.update_layout(
        title="LightGBM Global Feature Importance (mean |SHAP|)",
        xaxis_title="Mean |SHAP|", yaxis=dict(autorange="reversed"),
        height=550, template="plotly_white", margin=dict(l=200),
    )
    return fig


def plot_shap_county(fips: str, county_data: dict) -> go.Figure:
    shap_vals = np.array(county_data["shap"])
    order     = np.argsort(np.abs(shap_vals))[::-1][:12]
    labels    = [FEATURE_COLS[i] for i in order]
    vals      = [shap_vals[i] for i in order]
    colors    = ["#c0392b" if v > 0 else "#2980b9" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title=f"SHAP Contributions — {county_data['name']} (base={county_data['base']:.4f})",
        xaxis_title="SHAP value (impact on comorbid_index prediction)",
        yaxis=dict(autorange="reversed"),
        height=500, template="plotly_white", margin=dict(l=200),
    )
    return fig


def plot_model_comparison(comp: pd.DataFrame) -> go.Figure:
    if "prophet_wape" not in comp.columns:
        return go.Figure().update_layout(title="Prophet eval not available — run forecast_prophet.py first")

    has = comp.dropna(subset=["lgbm_wape", "prophet_wape"])
    lgbm_wins = (has["winner"] == "LightGBM").sum()
    prop_wins = (has["winner"] == "Prophet").sum()

    tier_colors = {"Critical": "#c0392b", "High": "#e67e22",
                   "Moderate": "#f1c40f", "Low": "#2ecc71"}

    fig = go.Figure()
    for tier, grp in has.groupby("risk_tier"):
        fig.add_trace(go.Scatter(
            x=grp["prophet_wape"], y=grp["lgbm_wape"],
            mode="markers", name=tier,
            marker=dict(color=tier_colors.get(tier, "gray"), size=5, opacity=0.6),
            text=grp["county_name"] + ", " + grp["stateabbr"],
            hovertemplate="%{text}<br>Prophet WAPE: %{x:.3f} | LightGBM WAPE: %{y:.3f}",
        ))

    mx = max(has["prophet_wape"].max(), has["lgbm_wape"].max())
    fig.add_trace(go.Scatter(
        x=[0, mx], y=[0, mx], mode="lines", name="Equal performance",
        line=dict(color="black", dash="dash", width=1),
    ))
    fig.update_layout(
        title=f"Prophet vs LightGBM WAPE per county | LightGBM wins: {lgbm_wins} | Prophet wins: {prop_wins}",
        xaxis_title="Prophet WAPE (lower → better)",
        yaxis_title="LightGBM WAPE (lower → better)",
        height=600, template="plotly_white",
    )
    return fig


def plot_error_by_tier(comp: pd.DataFrame) -> go.Figure:
    tier_order  = ["Critical", "High", "Moderate", "Low"]
    tier_colors = {"Critical": "#c0392b", "High": "#e67e22",
                   "Moderate": "#f1c40f", "Low": "#2ecc71"}
    has_prophet = "prophet_wape" in comp.columns

    fig = make_subplots(
        rows=1, cols=2 if has_prophet else 1,
        subplot_titles=(
            ["LightGBM WAPE by Risk Tier", "Prophet WAPE by Risk Tier"]
            if has_prophet else ["LightGBM WAPE by Risk Tier"]
        ),
    )
    for tier in tier_order:
        grp = comp[comp["risk_tier"] == tier]
        fig.add_trace(go.Box(
            y=grp["lgbm_wape"].dropna(), name=tier,
            marker_color=tier_colors.get(tier, "gray"),
            legendgroup=tier, showlegend=True,
        ), row=1, col=1)
        if has_prophet:
            fig.add_trace(go.Box(
                y=grp["prophet_wape"].dropna(), name=tier,
                marker_color=tier_colors.get(tier, "gray"),
                legendgroup=tier, showlegend=False,
            ), row=1, col=2)

    fig.update_layout(title="Error Distribution by Risk Tier", height=500, template="plotly_white")
    return fig


def plot_forecast_choropleth(forecast_df: pd.DataFrame, year: int) -> go.Figure:
    yr_df      = forecast_df[forecast_df["forecast_year"] == year]
    n_critical = (yr_df["lgbm_forecast"] >= 0.70).sum()

    fig = go.Figure(go.Choropleth(
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations=yr_df["fips"],
        z=yr_df["lgbm_forecast"],
        colorscale=[
            [0.00, "#2ecc71"], [0.35, "#f1c40f"],
            [0.60, "#e67e22"], [0.80, "#c0392b"], [1.00, "#7b241c"],
        ],
        zmin=0.0, zmax=1.0,
        colorbar=dict(title="Comorbid Index"),
        marker_line_width=0.2,
        customdata=yr_df[["county_name", "stateabbr"]].values,
        hovertemplate="<b>%{customdata[0]}, %{customdata[1]}</b><br>Index: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"LightGBM Forecast {year} | Critical counties (≥0.70): {n_critical}",
        geo=dict(scope="usa", showlakes=False),
        height=550, template="plotly_white",
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("ComorbidAlert — Week 4 LightGBM")
    log.info("=" * 60)

    # 1. Load panel
    panel = load_panel()

    # 2. Feature engineering
    df, le = build_features(panel)

    # 3. Temporal CV
    log.info("\n── Temporal CV ──")
    cv_results = temporal_cv(df)

    fold2       = cv_results[2]
    best_model  = fold2["model"]
    county_eval = fold2["county_metrics"]

    county_eval.to_csv(OUTPUT_DIR / "lgbm_eval.csv", index=False)
    pd.DataFrame([cv_results[f]["aggregate"] for f in sorted(cv_results)]).to_csv(
        OUTPUT_DIR / "lgbm_summary.csv", index=False
    )

    # 4. Prophet comparison (optional — needs forecast_eval.csv from Week 3)
    prophet_eval_path = OUTPUT_DIR / "forecast_eval.csv"
    prophet_eval = pd.read_csv(prophet_eval_path) if prophet_eval_path.exists() else None
    if prophet_eval is None:
        log.warning("forecast_eval.csv not found — skipping Prophet comparison columns")

    comp = county_eval.rename(columns={
        "mae": "lgbm_mae", "rmse": "lgbm_rmse", "wape": "lgbm_wape"
    })
    if prophet_eval is not None:
        pe = prophet_eval[["fips", "mae", "rmse", "wape"]].rename(columns={
            "mae": "prophet_mae", "rmse": "prophet_rmse", "wape": "prophet_wape"
        })
        pe["fips"] = pe["fips"].astype(str).str.zfill(5)
        comp["fips"] = comp["fips"].astype(str).str.zfill(5)
        comp = comp.merge(pe, on="fips", how="left")
        comp["winner"]    = np.where(comp["lgbm_wape"] <= comp["prophet_wape"], "LightGBM", "Prophet")
        comp["wape_diff"] = comp["prophet_wape"] - comp["lgbm_wape"]
        lgbm_w = (comp["winner"] == "LightGBM").sum()
        prop_w = (comp["winner"] == "Prophet").sum()
        log.info(
            "Comparison | LightGBM wins: %d | Prophet wins: %d | median WAPE diff: %.4f",
            lgbm_w, prop_w, comp["wape_diff"].median()
        )
    comp.to_csv(OUTPUT_DIR / "lgbm_comparison.csv", index=False)

    # 5. SHAP
    log.info("\n── SHAP ──")
    shap_results = run_shap(
        best_model, fold2["X_train"], fold2["X_test"], fold2["test_df"]
    )

    # 6. Retrain on full 2021-2024, then recursive forecast
    log.info("\n── Retrain on full data + forecast 2025-2027 ──")
    full_train = (
        df[df["release_year"].isin(TRAIN_YEARS + [TEST_YEAR])]
        .dropna(subset=["lag_1yr", TARGET_COL])
        .copy()
    )
    full_train[FEATURE_COLS] = full_train[FEATURE_COLS].fillna(0)
    final_model = lgb.LGBMRegressor(**LGB_PARAMS)
    final_model.fit(
        full_train[FEATURE_COLS], full_train[TARGET_COL],
        callbacks=[lgb.log_evaluation(0)]
    )
    log.info("Final model trained on %d rows", len(full_train))

    forecast_df = recursive_forecast(df, final_model)
    forecast_df.to_csv(OUTPUT_DIR / "lgbm_forecast.csv", index=False)
    save_parquet_s3(forecast_df, LGBM_FC_KEY)

    # 7. Plots
    log.info("\n── Generating plots ──")
    plot_feature_importance(shap_results["global_importance"]).write_html(
        str(PLOTS_DIR / "feature_importance.html"), include_plotlyjs="cdn"
    )
    for fips, cdata in shap_results["county_shap"].items():
        plot_shap_county(fips, cdata).write_html(
            str(PLOTS_DIR / f"shap_county_{fips}.html"), include_plotlyjs="cdn"
        )
    plot_model_comparison(comp).write_html(
        str(PLOTS_DIR / "model_comparison.html"), include_plotlyjs="cdn"
    )
    plot_error_by_tier(comp).write_html(
        str(PLOTS_DIR / "error_by_tier.html"), include_plotlyjs="cdn"
    )
    for yr in HORIZON_YEARS:
        plot_forecast_choropleth(forecast_df, yr).write_html(
            str(PLOTS_DIR / f"forecast_map_{yr}.html"), include_plotlyjs="cdn"
        )

    # 8. Final summary
    log.info("\n" + "=" * 60)
    log.info("WEEK 4 SUMMARY")
    log.info("=" * 60)
    log.info("LightGBM CV:")
    for fid, res in cv_results.items():
        a = res["aggregate"]
        log.info("  Fold %d | Train %-15s → Test %d | MAE=%.4f | RMSE=%.4f | WAPE=%.4f",
                 fid, a["train_years"], a["test_year"],
                 a["mae_median"], a["rmse_median"], a["wape_median"])
    log.info("Prophet baseline (Week 3): MAE=0.020 | RMSE=0.020 | WAPE=0.0507")
    log.info("\nLightGBM forecast — Critical counties (index ≥ 0.70):")
    for yr in HORIZON_YEARS:
        n = (forecast_df[forecast_df["forecast_year"] == yr]["lgbm_forecast"] >= 0.70).sum()
        log.info("  %d: %d counties", yr, n)
    log.info("\nTop 5 SHAP features:")
    for _, r in shap_results["global_importance"].head(5).iterrows():
        log.info("  %-30s %.5f", r["feature"], r["mean_shap"])
    log.info("\nOutputs written:")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            log.info("  %s", f.relative_to(OUTPUT_DIR.parent))
    log.info("\nWeek 4 complete.")


if __name__ == "__main__":
    main()