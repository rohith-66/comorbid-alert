"""
Comorbidity Scoring Model
=========================
3-layer composite risk index for diabetes-cardiac comorbidity.

Layer 1 — Clinical Burden   : diabetes + CHD prevalence (CDC PLACES)
Layer 2 — Social Vulnerability: poverty + uninsured rate + obesity (Census/PLACES)
Layer 3 — Trajectory Signal  : YoY delta on Layer 1 (requires prior year data)

ComorbidIndex = 0.5 * L1 + 0.3 * L2 + 0.2 * L3
  — when L3 unavailable: 0.625 * L1 + 0.375 * L2 (weight redistributed proportionally)
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

# Layer weights — must sum to 1.0
W1_CLINICAL   = 0.5
W2_SOCIAL     = 0.3
W3_TRAJECTORY = 0.2

RISK_TIERS = [
    (0.75, "Critical"),
    (0.50, "High"),
    (0.25, "Moderate"),
    (0.00, "Low"),
]


def build_comorbid_index(
    df: pd.DataFrame,
    prior_df: Optional[pd.DataFrame] = None,
    w1: float = W1_CLINICAL,
    w2: float = W2_SOCIAL,
    w3: float = W3_TRAJECTORY,
) -> pd.DataFrame:
    """
    Add comorbidity score columns to the combined county DataFrame.

    Args:
        df:       Current year combined DataFrame (output of join_on_fips)
        prior_df: Prior year combined DataFrame for trajectory calculation (optional)
        w1, w2, w3: Layer weights (must sum to 1.0)

    Returns:
        df with added columns:
            comorbid_l1_clinical, comorbid_l2_social, comorbid_l3_trajectory,
            comorbid_index, risk_tier
    """
    assert abs(w1 + w2 + w3 - 1.0) < 1e-6, "Weights must sum to 1.0"

    out = df.copy()

    # ── Layer 1: Clinical Burden ──────────────────────────────────────────────
    out = _build_layer1(out)

    # ── Layer 2: Social Vulnerability ────────────────────────────────────────
    out = _build_layer2(out)

    # ── Layer 3: Trajectory Signal ───────────────────────────────────────────
    out = _build_layer3(out, prior_df)

    # ── Final Index ───────────────────────────────────────────────────────────
    # If L3 is unavailable (no prior year data), drop it and redistribute its
    # weight proportionally to L1 and L2 to preserve their ratio.
    # L1:L2 ratio = 0.5:0.3 = 5:3, so:
    #   eff_w1 = 0.5 / (0.5 + 0.3) = 0.625
    #   eff_w2 = 0.3 / (0.5 + 0.3) = 0.375
    l3_available = prior_df is not None and not prior_df.empty
    if not l3_available:
        eff_w1 = w1 / (w1 + w2)   # 0.625
        eff_w2 = w2 / (w1 + w2)   # 0.375
        eff_w3 = 0.0
        logger.warning(
            "L3 trajectory unavailable — weights redistributed: "
            "L1=%.3f  L2=%.3f  L3=0.000", eff_w1, eff_w2
        )
    else:
        eff_w1, eff_w2, eff_w3 = w1, w2, w3
        logger.info(
            "L3 trajectory available — using full weights: "
            "L1=%.3f  L2=%.3f  L3=%.3f", eff_w1, eff_w2, eff_w3
        )

    out["comorbid_index"] = (
        eff_w1 * out["comorbid_l1_clinical"].fillna(0)
        + eff_w2 * out["comorbid_l2_social"].fillna(0)
        + eff_w3 * out["comorbid_l3_trajectory"].fillna(0)
    )


    # ── Risk Tier ─────────────────────────────────────────────────────────────
    out["risk_tier"] = out["comorbid_index"].apply(_assign_tier)

    _log_summary(out)
    return out


def _build_layer1(df: pd.DataFrame) -> pd.DataFrame:
    """Clinical burden: diabetes + CHD prevalence, normalized 0-1."""
    if "places_diabetes" not in df.columns or "places_chd" not in df.columns:
        logger.warning("Score L1: missing places_diabetes or places_chd — layer will be 0")
        df["comorbid_l1_clinical"] = 0.0
        return df

    raw = (
        df["places_diabetes"].fillna(df["places_diabetes"].median())
        + df["places_chd"].fillna(df["places_chd"].median())
    )

    df["comorbid_l1_clinical"] = _minmax(raw)
    logger.info(
        "Score L1 clinical: mean=%.3f | std=%.3f",
        df["comorbid_l1_clinical"].mean(),
        df["comorbid_l1_clinical"].std(),
    )
    return df


def _build_layer2(df: pd.DataFrame) -> pd.DataFrame:
    """Social vulnerability: poverty + uninsured rate + obesity, normalized 0-1."""
    available = []

    if "poverty_rate_pct" in df.columns:
        available.append(
            _minmax(df["poverty_rate_pct"].fillna(df["poverty_rate_pct"].median()))
        )
    else:
        logger.warning("Score L2: poverty_rate_pct missing")

    # Uninsured rate — use pre-computed column or derive from Census columns
    if "uninsured_rate_pct" in df.columns:
        available.append(
            _minmax(df["uninsured_rate_pct"].fillna(df["uninsured_rate_pct"].median()))
        )
    elif all(
        c in df.columns
        for c in ["uninsured_18to34", "uninsured_35to64", "health_insurance_universe"]
    ):
        total_uninsured = df["uninsured_18to34"] + df["uninsured_35to64"]
        uninsured_rate = (
            total_uninsured / df["health_insurance_universe"].replace(0, np.nan)
        ) * 100
        available.append(_minmax(uninsured_rate.fillna(uninsured_rate.median())))
        logger.info("Score L2: derived uninsured_rate from Census columns")
    else:
        logger.warning("Score L2: uninsured rate unavailable")

    if "places_obesity" in df.columns:
        available.append(
            _minmax(df["places_obesity"].fillna(df["places_obesity"].median()))
        )
    else:
        logger.warning("Score L2: places_obesity missing")

    if not available:
        logger.warning("Score L2: no social vulnerability inputs — layer will be 0")
        df["comorbid_l2_social"] = 0.0
        return df

    # Equal-weight average across available indicators
    df["comorbid_l2_social"] = np.mean(np.stack(available, axis=1), axis=1)
    logger.info(
        "Score L2 social: mean=%.3f | inputs=%d/3",
        df["comorbid_l2_social"].mean(),
        len(available),
    )
    return df


def _build_layer3(
    df: pd.DataFrame, prior_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Trajectory: YoY delta on Layer 1 clinical score.
    Normalized to [0, 1] where 1 = most worsening, 0 = most improving.
    Falls back to neutral 0.5 when no prior year data is available —
    but the caller (build_comorbid_index) will zero out this layer's
    weight in that case, so the 0.5 value never affects the final index.
    """
    if prior_df is None or prior_df.empty:
        logger.info(
            "Score L3: no prior year data — trajectory set to neutral (0.5). "
            "L3 weight will be redistributed to L1+L2."
        )
        df["comorbid_l3_trajectory"] = 0.5
        return df

    # Build L1 for prior year
    prior_scored = _build_layer1(prior_df.copy())

    fips_col = "fips" if "fips" in df.columns else "fips_5"
    prior_fips_col = "fips" if "fips" in prior_df.columns else "fips_5"

    prior_l1 = (
        prior_scored.set_index(prior_fips_col)["comorbid_l1_clinical"]
        .rename("prior_l1")
    )
    current = df.set_index(fips_col)

    delta = (current["comorbid_l1_clinical"] - prior_l1).reset_index()
    delta.columns = [fips_col, "l3_delta"]

    df = df.merge(delta, on=fips_col, how="left")

    # Normalize delta: positive (worsening) → higher score
    raw_delta = df["l3_delta"].fillna(0)
    df["comorbid_l3_trajectory"] = _minmax(raw_delta)
    df = df.drop(columns=["l3_delta"])

    n_worsening = (raw_delta > 0).sum()
    logger.info(
        "Score L3 trajectory: %d counties worsening YoY | mean delta=%.4f",
        n_worsening,
        raw_delta.mean(),
    )
    return df


def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]. Returns 0.5 if all values are equal."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _assign_tier(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    for threshold, tier in RISK_TIERS:
        if score >= threshold:
            return tier
    return "Low"


def _log_summary(df: pd.DataFrame) -> None:
    tier_counts = df["risk_tier"].value_counts().to_dict()
    logger.info(
        "Score summary | index mean=%.3f | tiers: %s",
        df["comorbid_index"].mean(),
        tier_counts,
    )