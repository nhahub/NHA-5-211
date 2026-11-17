"""UI Components, Theme, and Layout Helpers for Smart IDS v3 Dashboard.

Provides custom CSS for the dark Security Operations Center (SOC) aesthetic,
pulsing threat badge animations, KPI cards, and Plotly theme styling.
"""

from typing import Optional
import streamlit as st

CUSTOM_CSS = """
<style>
/* Main Background and Text */
.stApp {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #0e1626;
    border-right: 1px solid #1e293b;
}

/* Header Banner */
.soc-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
}

.soc-header h1 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 12px;
}

.soc-header p {
    margin: 6px 0 0 0;
    color: #94a3b8;
    font-size: 0.95rem;
}

/* KPI Metric Cards */
.kpi-container {
    background: linear-gradient(145deg, #131b2e, #0f172a);
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 14px 18px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    text-align: left;
}

.kpi-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 4px;
}

.kpi-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
}

.kpi-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 4px;
}

/* Threat Severity Badges */
.threat-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-align: center;
}

.threat-low {
    background: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
}

.threat-medium {
    background: #451a03;
    color: #fbbf24;
    border: 1px solid #d97706;
}

.threat-high {
    background: #431407;
    color: #fb923c;
    border: 1px solid #ea580c;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.65; transform: scale(1.02); }
}

.threat-critical {
    background: #450a0a;
    color: #f87171;
    border: 1px solid #dc2626;
    animation: pulse 1.5s infinite;
}

/* Tab & Table Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 8px 16px;
    background-color: #131b2e;
    color: #94a3b8;
    border: 1px solid #1e293b;
}

.stTabs [aria-selected="true"] {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
    border-color: #38bdf8 !important;
}

/* Clean up borders */
div[data-testid="stMetricValue"] {
    color: #f8fafc;
}
</style>
"""


def inject_custom_css() -> None:
    """Inject SOC dark theme stylesheet with pulsing animation."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_threat_badge_html(threat_level: str) -> str:
    """Generate HTML snippet for styled threat level badge.

    Args:
        threat_level: 'Low', 'Medium', 'High', or 'Critical'.

    Returns:
        HTML string.
    """
    tl_lower = threat_level.lower()
    css_class = f"threat-{tl_lower}" if tl_lower in ("low", "medium", "high", "critical") else "threat-low"
    return f'<span class="threat-badge {css_class}">{threat_level.upper()}</span>'


def render_kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    badge_html: Optional[str] = None,
) -> None:
    """Render a styled KPI card in the SOC theme.

    Args:
        title: Card label.
        value: Headline numeric or text value.
        subtitle: Secondary descriptive text.
        badge_html: Optional HTML threat badge.
    """
    badge_part = f"<div style='margin-top:6px;'>{badge_html}</div>" if badge_html else ""
    html = f"""
    <div class="kpi-container">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtitle}</div>
        {badge_part}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


PLOTLY_DARK_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#94a3b8", "family": "Inter, sans-serif"},
        "xaxis": {
            "gridcolor": "#1e293b",
            "zerolinecolor": "#1e293b",
            "tickfont": {"color": "#94a3b8"},
        },
        "yaxis": {
            "gridcolor": "#1e293b",
            "zerolinecolor": "#1e293b",
            "tickfont": {"color": "#94a3b8"},
        },
    }
}
