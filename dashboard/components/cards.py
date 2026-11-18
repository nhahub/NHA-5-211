"""KPI cards, threat badges, and Plotly dark layout helpers."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.detection.threat_scoring import THREAT_ICONS


def render_metric_card(label: str, value: str, subtext: str = "") -> None:
    """Render a styled dark-navy gradient KPI card."""
    sub_html = f'<div class="metric-sub">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def threat_badge_html(level: str) -> str:
    """Render a styled threat-level badge with pulsing animation on Critical."""
    clean = level.strip().capitalize()
    cls = {
        "Low": "threat-low",
        "Medium": "threat-medium",
        "High": "threat-high",
        "Critical": "threat-critical",
    }.get(clean, "threat-low")
    icon = THREAT_ICONS.get(clean, "⚪")
    return f'<span class="{cls}">{icon} {clean}</span>'


def threat_badge_text(level: str) -> str:
    """Plain emoji-prefixed text badge suitable for st.dataframe columns."""
    clean = level.strip().capitalize()
    return f"{THREAT_ICONS.get(clean, '⚪')} {clean}"


def apply_soc_plotly_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply consistent dark SOC navy styling to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(16, 27, 51, 0.0)",
        plot_bgcolor="rgba(11, 19, 36, 0.65)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
        margin=dict(l=20, r=20, t=44, b=24),
        height=height,
        legend=dict(bgcolor="rgba(6, 11, 22, 0.55)", bordercolor="#1e3256", borderwidth=1),
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.12)", zerolinecolor="rgba(148, 163, 184, 0.2)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.12)", zerolinecolor="rgba(148, 163, 184, 0.2)")
    return fig
