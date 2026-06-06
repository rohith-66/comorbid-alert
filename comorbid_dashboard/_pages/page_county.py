"""
Page 2 — County Drill-Down
Full profile: tier badge · L1/L2/L3 · forecast chart · alerts · SHAP
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_loader import (
    load_main, load_forecasts, load_alerts, load_shap,
    TIER_COLOR, TIER_BADGE, ALERT_COLOR
)


def render():
    st.markdown('<div class="page-title">🔍 County Risk Profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Full comorbidity analysis for a single county — '
        'risk layers, forecast trajectories, alerts, and model explanations.</div>',
        unsafe_allow_html=True
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading data…"):
        main_df     = load_main()
        forecast_df = load_forecasts()
        alert_df    = load_alerts()
        shap_df     = load_shap()

    # ── County selector ───────────────────────────────────────────────────────
    # Build display names
    main_df["display_name"] = (
        main_df["county_name"].fillna("") + ", " + main_df["stateabbr"].fillna("")
        + "  [" + main_df["risk_tier"].fillna("") + "]"
    )
    # Sort: Critical first, then alpha
    tier_order = {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3}
    main_df["_tier_ord"] = main_df["risk_tier"].map(tier_order).fillna(9)
    sorted_df = main_df.sort_values(["_tier_ord", "county_name"])
    options = sorted_df["display_name"].tolist()
    fips_map = dict(zip(sorted_df["display_name"], sorted_df["fips"]))

    # Pre-select from session state (clicked from map or alerts page)
    default_idx = 0
    if st.session_state.get("selected_fips"):
        sel_fips = st.session_state["selected_fips"]
        match = sorted_df[sorted_df["fips"] == sel_fips]
        if not match.empty:
            default_idx = options.index(match.iloc[0]["display_name"])

    search_col, _ = st.columns([2, 3])
    with search_col:
        chosen = st.selectbox(
            "Select county",
            options,
            index=default_idx,
            label_visibility="collapsed",
        )

    fips = fips_map[chosen]
    st.session_state["selected_fips"] = fips

    county_row   = main_df[main_df["fips"] == fips].iloc[0]
    county_fc    = forecast_df[forecast_df["fips"] == fips].sort_values("forecast_year")
    county_alerts = alert_df[alert_df["fips"] == fips]
    county_shap  = shap_df[shap_df["fips"] == fips] if "fips" in shap_df.columns else pd.DataFrame()

    tier   = str(county_row.get("risk_tier", "Unknown"))
    badge  = TIER_BADGE.get(tier, "badge-moderate")
    t_color = TIER_COLOR.get(tier, "#9BB0CC")
    index  = float(county_row.get("comorbid_index", 0))

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # ── Header card ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, rgba(17,29,53,0.95), rgba(22,32,64,0.95));
                border:1px solid rgba(0,180,180,0.2); border-radius:12px;
                padding:1.5rem 2rem; margin-bottom:1.5rem;
                border-left:4px solid {t_color};">
        <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
            <div>
                <div style="font-size:1.4rem; font-weight:700; color:#E8F0FE; margin-bottom:4px;">
                    {county_row.get('county_name','')}
                </div>
                <div style="font-size:0.9rem; color:#9BB0CC; margin-bottom:10px;">
                    {county_row.get('stateabbr','')} · FIPS {fips}
                </div>
                <span class="badge {badge}">{tier}</span>
            </div>
            <div style="margin-left:auto; text-align:right;">
                <div style="font-size:0.7rem; color:#9BB0CC; text-transform:uppercase; letter-spacing:0.1em;">
                    Comorbidity Index
                </div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:2.2rem;
                            font-weight:700; color:{t_color};">
                    {index:.4f}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── L1 / L2 / L3 + Alert status ──────────────────────────────────────────
    layer_col, alert_col = st.columns([3, 2])

    with layer_col:
        st.markdown('<div class="section-header">COMORBIDITY LAYER BREAKDOWN</div>', unsafe_allow_html=True)

        l1 = float(county_row.get("L1_score", county_row.get("l1_score", 0)) or 0)
        l2 = float(county_row.get("L2_score", county_row.get("l2_score", 0)) or 0)
        l3 = float(county_row.get("L3_score", county_row.get("l3_score", 0)) or 0)
        total = l1 + l2 + l3 or 1

        for name, val, color, desc in [
            ("L1 — Clinical Burden",    l1, "#FF4D4D",
             "Prevalence of diabetes, heart disease, obesity, hypertension"),
            ("L2 — Social Vulnerability", l2, "#FF8C00",
             "Poverty, insurance gaps, healthcare access, food insecurity"),
            ("L3 — Trajectory",         l3, "#00B4B4",
             "Rate of change, emerging trend vs baseline"),
        ]:
            pct = val / total * 100
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="font-size:0.8rem; font-weight:500; color:#E8F0FE;">{name}</span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:{color};">
                        {val:.4f}
                    </span>
                </div>
                <div style="background:rgba(255,255,255,0.06); border-radius:3px; height:8px; overflow:hidden;">
                    <div style="width:{pct:.1f}%; background:{color}; height:100%; border-radius:3px;
                                transition:width 0.4s ease;"></div>
                </div>
                <div style="font-size:0.7rem; color:#9BB0CC; margin-top:3px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with alert_col:
        st.markdown('<div class="section-header">ALERT STATUS</div>', unsafe_allow_html=True)
        if county_alerts.empty:
            st.markdown("""
            <div style="background:rgba(40,224,138,0.08); border:1px solid rgba(40,224,138,0.2);
                        border-radius:8px; padding:1rem; color:#28E08A; font-size:0.82rem;">
                ✓ No active alerts<br>
                <span style="color:#9BB0CC; font-size:0.75rem;">
                This county is not on an escalating trajectory.
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            for _, alert in county_alerts.iterrows():
                tier_a  = str(alert.get("alert_tier", ""))
                reason  = str(alert.get("alert_reason", "No detail available"))
                slope   = alert.get("slope", None)
                a_color = ALERT_COLOR.get(tier_a, "#9BB0CC")
                slope_str = f"Slope: {slope:+.4f}" if slope is not None and pd.notna(slope) else ""
                st.markdown(f"""
                <div style="border:1px solid {a_color}40; border-left:4px solid {a_color};
                            border-radius:0 8px 8px 0; padding:12px 14px; margin-bottom:10px;
                            background:rgba(17,29,53,0.7);">
                    <div style="color:{a_color}; font-size:0.75rem; font-weight:700;
                                text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">
                        ⚠ {tier_a} ALERT
                    </div>
                    <div style="font-size:0.8rem; color:#C5D5E8; line-height:1.5; margin-bottom:6px;">
                        {reason}
                    </div>
                    {"<div style='font-family:IBM Plex Mono,monospace; font-size:0.7rem; color:#9BB0CC;'>" + slope_str + "</div>" if slope_str else ""}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)

    # ── Forecast chart ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">FORECAST TRAJECTORY — ALL MODELS</div>', unsafe_allow_html=True)

    if not county_fc.empty:
        fig = go.Figure()

        years = county_fc["forecast_year"].tolist()

        # Anchor 2023 actual
        actual_val = index

        # Prophet
        if "prophet_yhat" in county_fc.columns:
            p_y = county_fc["prophet_yhat"].tolist()
            all_y_p = [actual_val] + p_y
            all_x_p = [2023] + years
            if "prophet_lower" in county_fc.columns and "prophet_upper" in county_fc.columns:
                fig.add_trace(go.Scatter(
                    x=list(county_fc["forecast_year"]) + list(county_fc["forecast_year"][::-1]),
                    y=list(county_fc["prophet_upper"]) + list(county_fc["prophet_lower"][::-1]),
                    fill="toself", fillcolor="rgba(155,176,204,0.08)",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))
            fig.add_trace(go.Scatter(
                x=all_x_p, y=all_y_p,
                mode="lines+markers", name="Prophet",
                line=dict(color="#9BB0CC", width=1.8, dash="dot"),
                marker=dict(size=6, color="#9BB0CC"),
            ))

        # LightGBM
        if "lgbm_yhat" in county_fc.columns:
            lgbm_y = county_fc["lgbm_yhat"].tolist()
            all_y_l = [actual_val] + lgbm_y
            all_x_l = [2023] + years
            fig.add_trace(go.Scatter(
                x=all_x_l, y=all_y_l,
                mode="lines+markers", name="LightGBM",
                line=dict(color="#FF8C00", width=1.8, dash="dash"),
                marker=dict(size=6, color="#FF8C00"),
            ))

        # Ensemble (primary)
        if "ensemble_yhat" in county_fc.columns:
            ens_y = county_fc["ensemble_yhat"].tolist()
            all_y_e = [actual_val] + ens_y
            all_x_e = [2023] + years
            if "ensemble_upper" in county_fc.columns and "ensemble_lower" in county_fc.columns:
                fig.add_trace(go.Scatter(
                    x=list(county_fc["forecast_year"]) + list(county_fc["forecast_year"][::-1]),
                    y=list(county_fc["ensemble_upper"]) + list(county_fc["ensemble_lower"][::-1]),
                    fill="toself", fillcolor="rgba(0,180,180,0.12)",
                    line=dict(width=0), name="Ensemble CI", hoverinfo="skip",
                ))
            fig.add_trace(go.Scatter(
                x=all_x_e, y=all_y_e,
                mode="lines+markers", name="Ensemble",
                line=dict(color="#00B4B4", width=2.5),
                marker=dict(size=8, color="#00B4B4", symbol="diamond"),
            ))

        # 2023 baseline dot
        fig.add_trace(go.Scatter(
            x=[2023], y=[actual_val],
            mode="markers", name="2023 Actual",
            marker=dict(size=10, color="#E8F0FE", symbol="circle",
                        line=dict(width=2, color="#00B4B4")),
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                tickfont=dict(color="#9BB0CC", size=10),
                gridcolor="rgba(255,255,255,0.04)",
                tickvals=[2023, 2025, 2026, 2027],
            ),
            yaxis=dict(
                tickfont=dict(color="#9BB0CC", size=10),
                gridcolor="rgba(255,255,255,0.04)",
                title=dict(text="Comorbidity Index", font=dict(color="#9BB0CC", size=10)),
            ),
            legend=dict(
                bgcolor="rgba(11,20,38,0.7)", bordercolor="rgba(0,180,180,0.2)",
                borderwidth=1, font=dict(color="#9BB0CC", size=11),
                orientation="h", x=0, y=-0.15,
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"fc_full_{fips}")
    else:
        st.info("No forecast data available for this county.")

    # ── SHAP top 5 ────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">TOP 5 RISK DRIVERS — SHAP EXPLANATION</div>', unsafe_allow_html=True)

    if not county_shap.empty:
        row_s = county_shap.iloc[0]
        shap_data = {}
        for i in range(1, 6):
            fn = row_s.get(f"feature{i}", row_s.get(f"feat{i}"))
            fv = row_s.get(f"value{i}", row_s.get(f"shap{i}"))
            if fn and fv is not None:
                shap_data[str(fn)] = float(fv)
        if not shap_data and "feature_name" in county_shap.columns:
            for _, sr in county_shap.head(5).iterrows():
                shap_data[str(sr["feature_name"])] = float(sr.get("shap_value", 0))

        if shap_data:
            feats  = list(shap_data.keys())
            vals   = [shap_data[f] for f in feats]
            colors = ["#FF4D4D" if v > 0 else "#28E08A" for v in vals]

            FEATURE_LABELS = {
                "places_diabetes":         "Diabetes prevalence (CDC PLACES)",
                "places_chd":              "Coronary heart disease prevalence",
                "places_obesity":          "Adult obesity rate",
                "places_hypertension":     "Hypertension prevalence",
                "places_bphigh":           "High blood pressure rate",
                "acs_poverty_rate":        "Poverty rate (Census ACS)",
                "acs_uninsured":           "Population without insurance",
                "acs_median_income":       "Median household income",
                "L1_score":                "L1 Clinical burden score",
                "L2_score":                "L2 Social vulnerability score",
                "L3_score":                "L3 Trajectory score",
                "comorbid_index_lag1":     "Comorbidity index — 1 year lag",
                "comorbid_index_lag2":     "Comorbidity index — 2 year lag",
            }
            display_feats = [FEATURE_LABELS.get(f, f.replace("_", " ").title()) for f in feats]

            shap_col, desc_col = st.columns([3, 2])
            with shap_col:
                fig_shap = go.Figure(go.Bar(
                    x=vals[::-1],
                    y=display_feats[::-1],
                    orientation="h",
                    marker_color=colors[::-1],
                    text=[f"{v:+.4f}" for v in vals[::-1]],
                    textposition="outside",
                    textfont=dict(color="#9BB0CC", size=10),
                ))
                fig_shap.update_layout(
                    height=250,
                    margin=dict(l=0, r=80, t=5, b=5),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        showgrid=False, showticklabels=False,
                        zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
                    ),
                    yaxis=dict(tickfont=dict(color="#C5D5E8", size=10)),
                    showlegend=False,
                )
                st.plotly_chart(fig_shap, use_container_width=True,
                                config={"displayModeBar": False}, key=f"shap_full_{fips}")

            with desc_col:
                st.markdown("""
                <div style="padding:1rem; background:rgba(17,29,53,0.8);
                            border:1px solid rgba(0,180,180,0.15); border-radius:8px;
                            font-size:0.78rem; color:#9BB0CC; line-height:1.7;">
                    <div style="color:#00B4B4; font-weight:600; margin-bottom:8px;">
                        How to read this
                    </div>
                    <span style="color:#FF4D4D;">■ Red bars</span> push the risk index
                    <strong>higher</strong> — these features increase this county's score
                    above the average.<br><br>
                    <span style="color:#28E08A;">■ Green bars</span> push the risk index
                    <strong>lower</strong> — these features are protective relative to
                    the average county.<br><br>
                    Bar length = magnitude of effect. The most important
                    driver is at the top.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No SHAP data available for this county.")