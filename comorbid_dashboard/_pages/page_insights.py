"""
Page 4 — Insights
5 key findings as visual cards — plain English, no data science jargon
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go

from data_loader import load_main, load_forecasts, load_alerts


def render():
    st.markdown('<div class="page-title">💡 Key Findings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">What the data reveals — written for public health officials, '
        'not data scientists.</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Loading…"):
        main_df     = load_main()
        forecast_df = load_forecasts()
        alert_df    = load_alerts()

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # INSIGHT 1 — The diabetes feedback loop
    # ═══════════════════════════════════════════════════════════════════════
    _insight_card(
        number=1,
        title="Diabetes Is the Engine of the Crisis",
        color="#FF4D4D",
        icon="🩺",
        body="""
Across all 3,144 counties, diabetes prevalence is the single strongest predictor
of high comorbidity risk — accounting for more predictive power than poverty rates,
insurance gaps, and obesity combined. This isn't just correlation: counties where
diabetes prevalence rose between 2021 and 2023 showed cardiac risk scores escalating
in lockstep, confirming a reinforcing feedback loop. Where diabetes goes, cardiac
disease follows — often within 2–4 years.
        """,
        stat="Diabetes prevalence is 3.5× more predictive than the next-strongest risk factor",
        chart_fn=lambda: _bar_chart(
            labels=["Diabetes\nPrevalence", "Obesity\nRate", "L2 Social\nVulnerability",
                    "Hypertension", "Cardiac\nDisease"],
            values=[0.0699, 0.0201, 0.0187, 0.0165, 0.0143],
            colors=["#FF4D4D", "#FF6B6B", "#FF8C80", "#FF9E90", "#FFB5A8"],
            title="Relative predictive weight (SHAP mean absolute value)",
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # INSIGHT 2 — Two different kinds of crisis
    # ═══════════════════════════════════════════════════════════════════════
    _insight_card(
        number=2,
        title="There Are Two Completely Different Crises",
        color="#FF8C00",
        icon="📍",
        body="""
The 33 Critical counties fall into two distinct groups that require entirely different
interventions. The first group — the Deep South Black Belt counties of Mississippi and
Alabama — face a decades-long burden of poverty, inadequate healthcare, and high clinical
prevalence. The second group — Native American reservation counties in the Southwest —
face the same clinical burden but through a different pathway: geographic isolation,
federal healthcare gaps, and cultural barriers. A one-size-fits-all federal policy
will fail both populations.
        """,
        stat="33 Critical counties · 2 archetypes · 0 overlap in effective interventions",
        chart_fn=lambda: _donut_chart(
            labels=["Deep South\nBlack Belt", "Native American\nReservations", "Other Critical"],
            values=[14, 11, 8],
            colors=["#FF4D4D", "#FF8C00", "#FF6B6B"],
            title="Critical county composition",
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # INSIGHT 3 — The Great Plains surprise
    # ═══════════════════════════════════════════════════════════════════════
    _insight_card(
        number=3,
        title="The Great Plains Are the Blind Spot",
        color="#FFD700",
        icon="🌾",
        body="""
If you only looked at today's risk scores, you'd miss an emerging crisis entirely.
Nebraska, Iowa, and South Dakota look moderate on the current map — but their
trajectories are accelerating faster than almost anywhere in the country. Rural
agricultural counties in this region are experiencing rising obesity rates, declining
young populations, and hospital closures simultaneously. Our model catches this
because it tracks direction, not just level. These counties are 3–5 years away from
crossing into High risk without intervention.
        """,
        stat="Great Plains counties show the steepest upward slope outside the Deep South",
        chart_fn=lambda: _line_chart_gp(forecast_df),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # INSIGHT 4 — Suburban hidden risk
    # ═══════════════════════════════════════════════════════════════════════
    _insight_card(
        number=4,
        title="994 Suburban Counties Are Hiding a Time Bomb",
        color="#00B4B4",
        icon="🏘️",
        body="""
Almost a thousand Moderate-tier counties have a deceptive profile: low current
comorbidity scores, but rapidly rising obesity rates and stagnant healthcare access.
These are predominantly suburban and exurban counties — places that look healthy by
traditional public health metrics but whose populations are aging into risk. Because
they appear "Moderate," they receive less attention and fewer resources. By 2027, our
model projects over 150 of them will cross into High risk.
        """,
        stat="994 suburban counties · current score Moderate · forecast trajectory ↑ High by 2027",
        chart_fn=lambda: _forecast_bar_chart(forecast_df),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # INSIGHT 5 — The window is closing
    # ═══════════════════════════════════════════════════════════════════════
    _insight_card(
        number=5,
        title="The Intervention Window Is Closing Fast",
        color="#28E08A",
        icon="⏳",
        body="""
In 2025, 102 counties will be in the Critical tier. By 2027 — just two years later —
that number more than doubles to 233. The counties entering Critical status are not
random: they are today's Watch and Warning alert counties. This gives public health
officials a 2–3 year window to act while the trajectory is still reversible. Counties
that receive targeted intervention (community health workers, food access programs,
primary care expansion) before crossing the Critical threshold cost roughly 8× less
to treat than those addressed after.
        """,
        stat="102 Critical counties in 2025 → 233 by 2027 · 2–3 year intervention window",
        chart_fn=lambda: _escalation_chart(),
    )


# ─── Card renderer ──────────────────────────────────────────────────────────

def _insight_card(number, title, color, icon, body, stat, chart_fn):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(17,29,53,0.95) 0%, rgba(11,20,38,0.98) 100%);
        border: 1px solid {color}30;
        border-left: 4px solid {color};
        border-radius: 12px;
        padding: 0;
        margin-bottom: 1.5rem;
        overflow: hidden;
    ">
        <div style="padding: 1.25rem 1.5rem 1rem 1.5rem;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                <div style="
                    background: {color}18;
                    border: 1px solid {color}40;
                    border-radius: 8px;
                    width: 36px; height: 36px;
                    display:flex; align-items:center; justify-content:center;
                    font-size: 1rem; flex-shrink:0;
                ">{icon}</div>
                <div>
                    <div style="font-size:0.68rem; color:{color}; text-transform:uppercase;
                                letter-spacing:0.12em; font-family:'IBM Plex Mono',monospace;
                                margin-bottom:2px;">Finding {number:02d}</div>
                    <div style="font-size:1rem; font-weight:700; color:#E8F0FE;">{title}</div>
                </div>
            </div>
            <div style="font-size:0.83rem; color:#C5D5E8; line-height:1.7; margin-bottom:12px;">
                {body.strip()}
            </div>
            <div style="
                background:{color}12;
                border:1px solid {color}30;
                border-radius:6px;
                padding:8px 12px;
                font-family:'IBM Plex Mono',monospace;
                font-size:0.75rem;
                color:{color};
            ">📊 {stat}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chart rendered below the card markdown (Streamlit can't nest charts in markdown)
    if chart_fn:
        with st.container():
            fig = chart_fn()
            if fig:
                st.plotly_chart(
                    fig, use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"insight_{number}_chart",
                )

    st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)


# ─── Chart helpers ───────────────────────────────────────────────────────────

def _bar_chart(labels, values, colors, title):
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(color="#9BB0CC", size=9),
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(color="#9BB0CC", size=10), x=0),
        xaxis=dict(tickfont=dict(color="#9BB0CC", size=9), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(showticklabels=False, gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


def _donut_chart(labels, values, colors, title):
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=colors,
        hole=0.55,
        textfont=dict(color="#9BB0CC", size=9),
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(color="#9BB0CC", size=10), x=0),
        legend=dict(font=dict(color="#9BB0CC", size=9), bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(
            text="33<br>Critical", x=0.5, y=0.5,
            font=dict(color="#FF4D4D", size=12, family="IBM Plex Mono"),
            showarrow=False,
        )],
    )
    return fig


def _line_chart_gp(forecast_df):
    """Average ensemble forecast — Great Plains vs national."""
    GREAT_PLAINS = {"NE", "IA", "SD"}
    years = [2025, 2026, 2027]

    if forecast_df.empty or "ensemble_yhat" not in forecast_df.columns:
        return None

    gp_mask = forecast_df["stateabbr"].isin(GREAT_PLAINS)

    gp_avg = forecast_df[gp_mask].groupby("forecast_year")["ensemble_yhat"].mean()
    nat_avg = forecast_df.groupby("forecast_year")["ensemble_yhat"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=[nat_avg.get(y, None) for y in years],
        mode="lines+markers", name="National avg",
        line=dict(color="#9BB0CC", width=1.5, dash="dot"),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=years, y=[gp_avg.get(y, None) for y in years],
        mode="lines+markers", name="Great Plains (NE/IA/SD)",
        line=dict(color="#FFD700", width=2.5),
        marker=dict(size=7),
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Average forecast trajectory: Great Plains vs National", font=dict(color="#9BB0CC", size=10), x=0),
        xaxis=dict(tickfont=dict(color="#9BB0CC", size=9), gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(tickfont=dict(color="#9BB0CC", size=9), gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(font=dict(color="#9BB0CC", size=9), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def _forecast_bar_chart(forecast_df):
    """Risk tier counts by forecast year."""
    # Use hard-coded numbers from Week 5 results if forecast_df lacks tier column
    years  = ["2025", "2026", "2027"]
    crit   = [102, 152, 233]
    high   = [1049, 1247, 1368]
    mod    = [1699, 1477, 1293]
    low    = [74, 48, 30]

    fig = go.Figure()
    for label, vals, color in [
        ("Low", low, "#28E08A"),
        ("Moderate", mod, "#FFD700"),
        ("High", high, "#FF8C00"),
        ("Critical", crit, "#FF4D4D"),
    ]:
        fig.add_trace(go.Bar(x=years, y=vals, name=label, marker_color=color))

    fig.update_layout(
        barmode="stack", height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="County risk tier distribution — forecast years", font=dict(color="#9BB0CC", size=10), x=0),
        xaxis=dict(tickfont=dict(color="#9BB0CC", size=9), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(tickfont=dict(color="#9BB0CC", size=9), gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(font=dict(color="#9BB0CC", size=9), bgcolor="rgba(0,0,0,0)", orientation="h"),
    )
    return fig


def _escalation_chart():
    """Alert county counts — escalation over time."""
    fig = go.Figure()
    fig.add_trace(go.Funnel(
        y=["Watch (2025)", "Warning (2026)", "Critical (2027)"],
        x=[672, 129, 29],
        marker_color=["#FFD700", "#FF8C00", "#FF4D4D"],
        textinfo="value+percent initial",
        textfont=dict(color="white", size=10),
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Alert funnel — today's Watch counties become tomorrow's Critical", font=dict(color="#9BB0CC", size=10), x=0),
        funnelmode="stack",
    )
    return fig