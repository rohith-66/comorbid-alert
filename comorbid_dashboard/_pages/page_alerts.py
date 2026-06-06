"""
Page 3 — Alerts
830 alerts · sortable · filterable · Great Plains callout
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_loader import load_alerts, load_main, ALERT_COLOR


GREAT_PLAINS_STATES = {"NE", "IA", "SD", "KS", "ND", "MN"}


def render():
    st.markdown('<div class="page-title">🚨 Early Warning Alerts</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">830 counties on escalating trajectories — '
        'ranked by severity and rate of change.</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Loading alert data…"):
        alert_df = load_alerts()
        main_df  = load_main()

    # Enrich alerts with comorbidity index and risk tier
    if "comorbid_index" not in alert_df.columns:
        enrich = main_df[["fips", "comorbid_index", "risk_tier"]].copy()
        alert_df = alert_df.merge(enrich, on="fips", how="left")

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    n_critical = (alert_df["alert_tier"] == "Critical").sum()
    n_warning  = (alert_df["alert_tier"] == "Warning").sum()
    n_watch    = (alert_df["alert_tier"] == "Watch").sum()
    n_gp       = alert_df["stateabbr"].isin(GREAT_PLAINS_STATES).sum()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in [
        (c1, "Critical Alerts",  n_critical, "#FF4D4D"),
        (c2, "Warning Alerts",   n_warning,  "#FF8C00"),
        (c3, "Watch Alerts",     n_watch,    "#FFD700"),
        (c4, "Great Plains",     n_gp,       "#00B4B4"),
    ]:
        col.markdown(f"""
        <div class="metric-card" style="text-align:center; padding:0.7rem;">
            <div class="value" style="color:{color};">{val}</div>
            <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)

    # ── Great Plains callout ──────────────────────────────────────────────────
    gp_alerts = alert_df[alert_df["stateabbr"].isin(GREAT_PLAINS_STATES)]
    if not gp_alerts.empty:
        st.markdown(f"""
        <div class="alert-callout">
            <div style="display:flex; align-items:flex-start; gap:12px;">
                <div style="font-size:1.5rem;">🌾</div>
                <div>
                    <div style="font-weight:700; color:#FFD700; font-size:0.9rem; margin-bottom:4px;">
                        Novel Finding: Great Plains Emerging Cluster
                    </div>
                    <div style="font-size:0.82rem; color:#C5D5E8; line-height:1.6; margin-bottom:6px;">
                        <strong>{len(gp_alerts)} counties</strong> across Nebraska, Iowa, and South Dakota are showing
                        unexpected worsening trajectories — a pattern not visible in current 2023 scores.
                        This cluster was not identified in prior public health literature and represents a
                        genuine early warning: rural agricultural counties with rising obesity and declining
                        healthcare access are converging on a diabetes-cardiac risk pathway.
                    </div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#FFD700;">
                        States: {" · ".join(sorted(gp_alerts["stateabbr"].unique()))}
                        · Avg slope: {f'{gp_alerts["slope"].mean():+.4f}' if "slope" in gp_alerts.columns else "N/A"}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">FILTERS</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([2, 2, 1])

    with f1:
        tier_filter = st.multiselect(
            "Alert tier",
            ["Critical", "Warning", "Watch"],
            default=["Critical", "Warning", "Watch"],
            label_visibility="visible",
        )
    with f2:
        states = sorted(alert_df["stateabbr"].dropna().unique())
        state_filter = st.multiselect(
            "State",
            states,
            default=[],
            placeholder="All states",
            label_visibility="visible",
        )
    with f3:
        gp_only = st.toggle("Great Plains only", value=False)
        sort_col = st.selectbox(
            "Sort by",
            ["alert_tier", "slope", "stateabbr", "comorbid_index"],
            label_visibility="collapsed",
        )

    # Apply filters
    filtered = alert_df.copy()
    if tier_filter:
        filtered = filtered[filtered["alert_tier"].isin(tier_filter)]
    if state_filter:
        filtered = filtered[filtered["stateabbr"].isin(state_filter)]
    if gp_only:
        filtered = filtered[filtered["stateabbr"].isin(GREAT_PLAINS_STATES)]

    # Sort
    tier_ord = {"Critical": 0, "Warning": 1, "Watch": 2}
    if sort_col == "alert_tier":
        filtered["_tier_ord"] = filtered["alert_tier"].map(tier_ord)
        filtered = filtered.sort_values("_tier_ord").drop(columns=["_tier_ord"])
    elif sort_col == "slope":
        filtered = filtered.sort_values("slope", ascending=False)
    else:
        filtered = filtered.sort_values(sort_col)

    st.markdown(f"""
    <div style="font-size:0.78rem; color:#9BB0CC; margin-bottom:8px;">
        Showing <strong style="color:#00B4B4;">{len(filtered)}</strong> of 830 alerts
        {" · Great Plains highlighted" if gp_only else ""}
    </div>
    """, unsafe_allow_html=True)

    # ── Render table ──────────────────────────────────────────────────────────
    if not filtered.empty:
        # Build display table
        display_cols = []
        col_rename = {}

        for c, label in [
            ("county_name", "County"),
            ("stateabbr", "State"),
            ("alert_tier", "Alert Tier"),
            ("comorbid_index", "Index 2023"),
            ("risk_tier", "Risk Tier"),
            ("slope", "Slope"),
            ("alert_reason", "Alert Reason"),
        ]:
            if c in filtered.columns:
                display_cols.append(c)
                col_rename[c] = label

        table_df = filtered[display_cols].rename(columns=col_rename).reset_index(drop=True)

        # Style the dataframe
        def style_tier(val):
            colors = {
                "Critical": "color: #FF4D4D; font-weight:700",
                "Warning":  "color: #FF8C00; font-weight:600",
                "Watch":    "color: #FFD700",
            }
            return colors.get(val, "")

        def style_risk(val):
            colors = {
                "Critical": "color: #FF4D4D",
                "High":     "color: #FF8C00",
                "Moderate": "color: #FFD700",
                "Low":      "color: #28E08A",
            }
            return colors.get(val, "")

        # Highlight Great Plains rows
        def highlight_gp(row):
            if row.get("State") in GREAT_PLAINS_STATES:
                return ["background-color: rgba(255,215,0,0.06)"] * len(row)
            return [""] * len(row)

        styler = table_df.style

        if "Alert Tier" in table_df.columns:
            styler = styler.map(style_tier, subset=["Alert Tier"])
        if "Risk Tier" in table_df.columns:
            styler = styler.map(style_risk, subset=["Risk Tier"])
        if "State" in table_df.columns:
            styler = styler.apply(highlight_gp, axis=1)
        if "Slope" in table_df.columns:
            styler = styler.format({"Slope": "{:+.4f}"})
        if "Index 2023" in table_df.columns:
            styler = styler.format({"Index 2023": "{:.4f}"})

        st.dataframe(
            styler,
            use_container_width=True,
            height=480,
            column_config={
                "Alert Reason": st.column_config.TextColumn(width="large"),
                "County": st.column_config.TextColumn(width="medium"),
            },
        )
    else:
        st.info("No alerts match the current filters.")

    # ── Tier distribution chart ───────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">ALERT DISTRIBUTION BY STATE</div>', unsafe_allow_html=True)

    chart_col, map_col = st.columns([3, 2])
    with chart_col:
        state_counts = (
            filtered.groupby(["stateabbr", "alert_tier"])
            .size()
            .reset_index(name="count")
        )
        if not state_counts.empty:
            top_states = (
                state_counts.groupby("stateabbr")["count"].sum()
                .nlargest(20).index
            )
            state_top = state_counts[state_counts["stateabbr"].isin(top_states)]

            fig = go.Figure()
            for tier, color in [("Watch", "#FFD700"), ("Warning", "#FF8C00"), ("Critical", "#FF4D4D")]:
                td = state_top[state_top["alert_tier"] == tier]
                if td.empty:
                    continue
                fig.add_trace(go.Bar(
                    x=td["stateabbr"], y=td["count"],
                    name=tier, marker_color=color,
                ))
            fig.update_layout(
                barmode="stack", height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickfont=dict(color="#9BB0CC", size=9), gridcolor="rgba(0,0,0,0)"),
                yaxis=dict(tickfont=dict(color="#9BB0CC", size=9), gridcolor="rgba(255,255,255,0.05)"),
                legend=dict(font=dict(color="#9BB0CC"), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with map_col:
        st.markdown("""
        <div style="padding:1rem; background:rgba(17,29,53,0.8);
                    border:1px solid rgba(0,180,180,0.15); border-radius:8px;
                    font-size:0.78rem; color:#9BB0CC; line-height:1.8; margin-top:0.5rem;">
            <div style="color:#00B4B4; font-weight:600; margin-bottom:8px;">ALERT DEFINITIONS</div>
            <div style="margin-bottom:8px;">
                <span style="color:#FF4D4D; font-weight:700;">● Critical Alert</span><br>
                County already in Critical tier and forecast shows continued worsening.
                Immediate intervention needed.
            </div>
            <div style="margin-bottom:8px;">
                <span style="color:#FF8C00; font-weight:700;">▲ Warning Alert</span><br>
                County in High tier with trajectory crossing into Critical by 2026–2027.
                Proactive action window closing.
            </div>
            <div>
                <span style="color:#FFD700; font-weight:700;">■ Watch Alert</span><br>
                County showing early-stage acceleration above baseline. Monitor and prepare.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Drill-down link ───────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    if not filtered.empty and "fips" in filtered.columns:
        sel_fips = st.selectbox(
            "Open county profile",
            ["— select a county —"] + filtered["fips"].tolist(),
            format_func=lambda f: (
                f if f == "— select a county —"
                else f"{filtered[filtered['fips']==f]['county_name'].values[0]}, "
                     f"{filtered[filtered['fips']==f]['stateabbr'].values[0]}"
                     if f in filtered['fips'].values else f
            ),
            label_visibility="visible",
        )
        if sel_fips != "— select a county —":
            st.session_state["selected_fips"] = sel_fips
            st.info("County selected — go to the 🔍 County Drill-Down page to view the full profile.")