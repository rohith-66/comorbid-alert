"""
Page 1 — Main Map
US choropleth colored by comorbidity index
Toggle: alert overlays · forecast year
Click county → drill-down panel
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import urllib.request

from data_loader import (
    load_main, load_forecasts, load_alerts,
    TIER_COLOR, ALERT_COLOR
)

GEOJSON_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/"
    "geojson-counties-fips.json"
)


@st.cache_data(ttl=86400, show_spinner=False)
def load_geojson():
    with urllib.request.urlopen(GEOJSON_URL) as resp:
        return json.load(resp)


def render():
    st.markdown('<div class="page-title">🗺️ Comorbidity Risk Map</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">US counties colored by 2023 comorbidity index. '
        'Toggle alerts and forecast years. Click any county to drill down.</div>',
        unsafe_allow_html=True
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading county data…"):
        main_df    = load_main()
        forecast_df = load_forecasts()
        alert_df   = load_alerts()
        counties   = load_geojson()

    # ── Top KPI strip ─────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    kpi_style = "text-align:center; padding:0.6rem 0.5rem;"
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="{kpi_style}">
            <div class="value">3,144</div>
            <div class="label">Counties Tracked</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="{kpi_style}">
            <div class="value" style="color:#FF4D4D;">33</div>
            <div class="label">Critical Counties</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="{kpi_style}">
            <div class="value" style="color:#FF8C00;">381</div>
            <div class="label">High Risk</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="{kpi_style}">
            <div class="value" style="color:#FFD700;">830</div>
            <div class="label">Active Alerts</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card" style="{kpi_style}">
            <div class="value">0.46%</div>
            <div class="label">Ensemble WAPE</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])
    with ctrl_col1:
        map_mode = st.radio(
            "Map data",
            ["Current (2023 Index)", "Forecast 2025", "Forecast 2026", "Forecast 2027"],
            horizontal=True,
            label_visibility="collapsed"
        )
    with ctrl_col2:
        show_alerts = st.toggle("Show alert markers", value=True)
        alert_filter = st.multiselect(
            "Alert tiers",
            ["Critical", "Warning", "Watch"],
            default=["Critical", "Warning", "Watch"],
            label_visibility="collapsed"
        ) if show_alerts else []
    with ctrl_col3:
        color_scale = st.selectbox(
            "Color scale",
            ["YlOrRd", "RdYlGn_r", "Plasma"],
            label_visibility="collapsed"
        )

    # ── Build choropleth data ─────────────────────────────────────────────────
    if map_mode == "Current (2023 Index)":
        plot_df = main_df[["fips", "county_name", "stateabbr", "comorbid_index", "risk_tier"]].copy()
        plot_df["z"] = plot_df["comorbid_index"]
        z_label = "Comorbidity Index (2023)"
        z_min, z_max = plot_df["z"].quantile(0.02), plot_df["z"].quantile(0.98)
    else:
        year = int(map_mode.split()[-1])
        fc = forecast_df[forecast_df["forecast_year"] == year].copy()
        plot_df = fc[["fips", "county_name", "stateabbr", "ensemble_yhat"]].copy()
        plot_df["z"] = plot_df["ensemble_yhat"]
        plot_df["risk_tier"] = "—"
        z_label = f"Forecast Index ({year})"
        z_min, z_max = plot_df["z"].quantile(0.02), plot_df["z"].quantile(0.98)

    plot_df["hover"] = (
        "<b>" + plot_df["county_name"] + ", " + plot_df["stateabbr"] + "</b><br>"
        + "Index: " + plot_df["z"].round(4).astype(str) + "<br>"
        + "Risk Tier: " + plot_df.get("risk_tier", "—").astype(str)
    )

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        geojson=counties,
        locations=plot_df["fips"],
        z=plot_df["z"],
        text=plot_df["hover"],
        hoverinfo="text",
        colorscale=color_scale,
        zmin=z_min,
        zmax=z_max,
        marker_line_width=0.2,
        marker_line_color="rgba(255,255,255,0.1)",
        colorbar=dict(
            title=dict(text=z_label, font=dict(color="#9BB0CC", size=11)),
            tickfont=dict(color="#9BB0CC", size=10),
            bgcolor="rgba(11,20,38,0.8)",
            bordercolor="rgba(0,180,180,0.2)",
            borderwidth=1,
            len=0.7,
            thickness=14,
            x=1.01,
        ),
        customdata=plot_df["fips"],
    ))

    # Alert overlays
    if show_alerts and alert_filter:
        alert_sub = alert_df[alert_df["alert_tier"].isin(alert_filter)].copy()
        # Join lat/lon from main_df if available, else use centroid approximation
        if "latitude" in main_df.columns and "longitude" in main_df.columns:
            geo = main_df[["fips", "latitude", "longitude"]]
            alert_sub = alert_sub.merge(geo, on="fips", how="left")
        else:
            # Approximate county centroids from FIPS (state-level fallback)
            alert_sub["latitude"] = None
            alert_sub["longitude"] = None

        for tier in ["Critical", "Warning", "Watch"]:
            tier_data = alert_sub[alert_sub["alert_tier"] == tier]
            if tier_data.empty or tier_data["latitude"].isna().all():
                continue
            tier_data = tier_data.dropna(subset=["latitude", "longitude"])
            color = ALERT_COLOR[tier]
            symbol = "circle" if tier == "Critical" else "triangle-up" if tier == "Warning" else "square"
            size   = 10 if tier == "Critical" else 7 if tier == "Warning" else 5

            fig.add_trace(go.Scattergeo(
                lat=tier_data["latitude"],
                lon=tier_data["longitude"],
                mode="markers",
                marker=dict(
                    size=size,
                    color=color,
                    symbol=symbol,
                    opacity=0.85,
                    line=dict(width=0.5, color="white"),
                ),
                text=(
                    "<b>" + tier + " Alert</b><br>"
                    + tier_data["county_name"] + ", " + tier_data["stateabbr"] + "<br>"
                    + tier_data.get("alert_reason", "").fillna("").astype(str)
                ),
                hoverinfo="text",
                name=f"{tier} Alert",
                showlegend=True,
            ))

    fig.update_geos(
        scope="usa",
        showland=True, landcolor="rgba(11,20,38,1)",
        showlakes=True, lakecolor="rgba(0,70,100,0.4)",
        showcoastlines=True, coastlinecolor="rgba(0,180,180,0.15)",
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            bgcolor="rgba(11,20,38,0.85)",
            bordercolor="rgba(0,180,180,0.2)",
            borderwidth=1,
            font=dict(color="#9BB0CC", size=11),
            x=0.01, y=0.01,
        ),
        height=560,
    )

    # ── Render map + drill-down ───────────────────────────────────────────────
    map_col, drill_col = st.columns([3, 1.1])

    with map_col:
        selected = st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False},
            key="main_map",
            on_select="rerun",
        )

    # ── Drill-down panel ──────────────────────────────────────────────────────
    with drill_col:
        clicked_fips = None

        # Extract FIPS from plotly selection
        if selected and selected.get("selection") and selected["selection"].get("points"):
            pt = selected["selection"]["points"][0]
            clicked_fips = pt.get("customdata")

        # Also support session-state selection from Alerts page
        if not clicked_fips and st.session_state.get("selected_fips"):
            clicked_fips = st.session_state["selected_fips"]

        if clicked_fips:
            _render_drill_panel(clicked_fips, main_df, forecast_df, alert_df)
        else:
            st.markdown("""
            <div style="
                border:1px dashed rgba(0,180,180,0.25);
                border-radius:10px;
                padding:2rem 1rem;
                text-align:center;
                color:#9BB0CC;
                font-size:0.82rem;
                line-height:1.8;
                margin-top:1rem;
            ">
                <div style="font-size:1.5rem;margin-bottom:0.5rem;">🗺️</div>
                Click any county on the map<br>to open its risk profile
            </div>
            """, unsafe_allow_html=True)

        # Map legend
        st.markdown("""
        <div style="margin-top:1rem; font-size:0.72rem; color:#9BB0CC;">
        <div class="section-header">RISK TIERS</div>
        <div style="display:flex;flex-direction:column;gap:4px;">
            <div><span style="color:#FF4D4D;">●</span> Critical — top 1% risk</div>
            <div><span style="color:#FF8C00;">●</span> High — top 13%</div>
            <div><span style="color:#FFD700;">●</span> Moderate — middle</div>
            <div><span style="color:#28E08A;">●</span> Low — bottom 18%</div>
        </div>
        <div class="section-header" style="margin-top:12px;">ALERT MARKERS</div>
        <div style="display:flex;flex-direction:column;gap:4px;">
            <div><span style="color:#FF4D4D;">●</span> Critical alert (29)</div>
            <div><span style="color:#FF8C00;">▲</span> Warning alert (129)</div>
            <div><span style="color:#FFD700;">■</span> Watch alert (672)</div>
        </div>
        </div>
        """, unsafe_allow_html=True)


def _render_drill_panel(fips, main_df, forecast_df, alert_df):
    """Compact drill-down panel shown beside the map."""
    from data_loader import load_shap, TIER_BADGE
    import plotly.graph_objects as go

    shap_df = load_shap()

    county_row = main_df[main_df["fips"] == fips]
    if county_row.empty:
        st.warning(f"No data for FIPS {fips}")
        return

    row = county_row.iloc[0]
    county_alerts = alert_df[alert_df["fips"] == fips]
    county_fc = forecast_df[forecast_df["fips"] == fips].sort_values("forecast_year")
    county_shap = shap_df[shap_df["fips"] == fips]

    tier = str(row.get("risk_tier", "Unknown"))
    badge_cls = TIER_BADGE.get(tier, "badge-moderate")

    # Header
    st.markdown(f"""
    <div style="border:1px solid rgba(0,180,180,0.2); border-radius:10px; padding:1rem; background:rgba(17,29,53,0.9);">
        <div style="font-weight:600; font-size:0.95rem; margin-bottom:2px;">{row.get('county_name','')}</div>
        <div style="font-size:0.78rem; color:#9BB0CC; margin-bottom:8px;">{row.get('stateabbr','')}</div>
        <span class="badge {badge_cls}">{tier}</span>
        <span style="margin-left:8px; font-family:'IBM Plex Mono',monospace; font-size:0.82rem; color:#00B4B4;">
            {float(row.get('comorbid_index', 0)):.4f}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # L1 / L2 / L3 breakdown
    l1 = float(row.get("L1_score", row.get("l1_score", 0)) or 0)
    l2 = float(row.get("L2_score", row.get("l2_score", 0)) or 0)
    l3 = float(row.get("L3_score", row.get("l3_score", 0)) or 0)

    if l1 + l2 + l3 > 0:
        fig_layers = go.Figure(go.Bar(
            x=[l1, l2, l3],
            y=["L1 Clinical", "L2 Social", "L3 Trajectory"],
            orientation="h",
            marker_color=["#FF4D4D", "#FF8C00", "#00B4B4"],
            text=[f"{v:.3f}" for v in [l1, l2, l3]],
            textposition="outside",
            textfont=dict(color="#9BB0CC", size=9),
        ))
        fig_layers.update_layout(
            height=120, margin=dict(l=0, r=40, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(tickfont=dict(color="#9BB0CC", size=9)),
            showlegend=False,
        )
        st.markdown('<div class="section-header" style="margin-top:10px;">LAYER BREAKDOWN</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_layers, use_container_width=True, config={"displayModeBar": False}, key=f"layers_{fips}")

    # Forecast chart
    if not county_fc.empty:
        fig_fc = go.Figure()
        # Ensemble with CI
        if "ensemble_yhat" in county_fc.columns:
            fig_fc.add_trace(go.Scatter(
                x=county_fc["forecast_year"], y=county_fc["ensemble_yhat"],
                mode="lines+markers", name="Ensemble",
                line=dict(color="#00B4B4", width=2),
                marker=dict(size=5),
            ))
        if "ensemble_upper" in county_fc.columns and "ensemble_lower" in county_fc.columns:
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([county_fc["forecast_year"], county_fc["forecast_year"][::-1]]),
                y=pd.concat([county_fc["ensemble_upper"], county_fc["ensemble_lower"][::-1]]),
                fill="toself", fillcolor="rgba(0,180,180,0.12)",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
        if "prophet_yhat" in county_fc.columns:
            fig_fc.add_trace(go.Scatter(
                x=county_fc["forecast_year"], y=county_fc["prophet_yhat"],
                mode="lines", name="Prophet",
                line=dict(color="#9BB0CC", width=1.2, dash="dot"),
            ))
        if "lgbm_yhat" in county_fc.columns:
            fig_fc.add_trace(go.Scatter(
                x=county_fc["forecast_year"], y=county_fc["lgbm_yhat"],
                mode="lines", name="LightGBM",
                line=dict(color="#FF8C00", width=1.2, dash="dash"),
            ))
        fig_fc.update_layout(
            height=160, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickfont=dict(color="#9BB0CC", size=8), gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(tickfont=dict(color="#9BB0CC", size=8), gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(font=dict(color="#9BB0CC", size=8), bgcolor="rgba(0,0,0,0)"),
        )
        st.markdown('<div class="section-header">FORECAST 2025–2027</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False}, key=f"fc_{fips}")

    # Alert status
    if not county_alerts.empty:
        for _, alert in county_alerts.iterrows():
            tier_a = str(alert.get("alert_tier", ""))
            color_map = {"Critical": "#FF4D4D", "Warning": "#FF8C00", "Watch": "#FFD700"}
            color = color_map.get(tier_a, "#9BB0CC")
            reason = str(alert.get("alert_reason", "No detail available"))
            st.markdown(f"""
            <div style="border-left:3px solid {color}; padding:6px 10px; background:rgba(17,29,53,0.7);
                        border-radius:0 6px 6px 0; margin-bottom:6px;">
                <div style="color:{color}; font-size:0.72rem; font-weight:600; text-transform:uppercase;">{tier_a} ALERT</div>
                <div style="font-size:0.75rem; color:#C5D5E8; margin-top:2px; line-height:1.4;">{reason}</div>
            </div>
            """, unsafe_allow_html=True)

    # SHAP top 5
    if not county_shap.empty:
        # Handle wide or long format
        row_shap = county_shap.iloc[0]
        shap_data = {}
        # Wide: feature1, value1, feature2, value2, …
        for i in range(1, 6):
            fn = row_shap.get(f"feature{i}", row_shap.get(f"feat{i}"))
            fv = row_shap.get(f"value{i}", row_shap.get(f"shap{i}"))
            if fn and fv is not None:
                shap_data[str(fn)] = float(fv)
        # Long format: feature_name, shap_value
        if not shap_data and "feature_name" in county_shap.columns:
            for _, sr in county_shap.head(5).iterrows():
                shap_data[str(sr["feature_name"])] = float(sr.get("shap_value", 0))

        if shap_data:
            feats = list(shap_data.keys())[::-1]
            vals  = [shap_data[f] for f in feats]
            colors = ["#FF4D4D" if v > 0 else "#28E08A" for v in vals]

            fig_shap = go.Figure(go.Bar(
                x=vals, y=feats, orientation="h",
                marker_color=colors,
                text=[f"{v:+.4f}" for v in vals],
                textposition="outside",
                textfont=dict(color="#9BB0CC", size=8),
            ))
            fig_shap.update_layout(
                height=160, margin=dict(l=0, r=50, t=8, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=True,
                           zerolinecolor="rgba(255,255,255,0.1)"),
                yaxis=dict(tickfont=dict(color="#9BB0CC", size=8)),
                showlegend=False,
            )
            st.markdown('<div class="section-header">SHAP TOP 5 DRIVERS</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False}, key=f"shap_{fips}")

    # Link to full drill-down
    if st.button("Open Full Profile →", key=f"open_profile_{fips}"):
        st.session_state["selected_fips"] = fips
        st.session_state["nav_to"] = "drilldown"
        st.rerun()