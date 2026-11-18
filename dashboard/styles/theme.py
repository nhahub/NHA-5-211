"""Dark SOC Navy/Cyan CSS design system for Streamlit."""

from __future__ import annotations

import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary: #060b16;
    --bg-secondary: #0b1324;
    --bg-card: #101b33;
    --border-subtle: #1e3256;
    --accent-cyan: #00d4ff;
    --accent-blue: #3b82f6;
    --text-primary: #e2e8f0;
    --text-muted: #94a3b8;
}

.stApp {
    background: radial-gradient(circle at top right, #0d1d3a 0%, #060b16 55%, #040811 100%);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #091122 0%, #050a14 100%);
    border-right: 1px solid var(--border-subtle);
}

.soc-banner {
    background: linear-gradient(135deg, rgba(16, 27, 51, 0.96) 0%, rgba(11, 24, 48, 0.92) 60%, rgba(0, 212, 255, 0.12) 100%);
    border: 1px solid rgba(0, 212, 255, 0.30);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}

.soc-banner h1 {
    margin: 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #ffffff 0%, #38bdf8 60%, #00d4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.soc-banner p {
    margin: 0.35rem 0 0 0;
    color: var(--text-muted);
    font-size: 0.92rem;
}

.metric-card {
    background: linear-gradient(145deg, #101b33 0%, #0b1324 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28);
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(56, 189, 248, 0.55);
    transform: translateY(-1px);
}

.metric-label {
    color: var(--text-muted);
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.metric-value {
    color: #f8fafc;
    font-size: 1.55rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.25rem;
}

.metric-sub {
    color: #38bdf8;
    font-size: 0.78rem;
    margin-top: 0.2rem;
}

.threat-low {
    background: #0d2b1d;
    color: #4ade80;
    border: 1px solid #22c55e;
    border-radius: 6px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 0.82rem;
    display: inline-block;
}

.threat-medium {
    background: #2e2609;
    color: #facc15;
    border: 1px solid #eab308;
    border-radius: 6px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 0.82rem;
    display: inline-block;
}

.threat-high {
    background: #331d0a;
    color: #fb923c;
    border: 1px solid #f97316;
    border-radius: 6px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 0.82rem;
    display: inline-block;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

.threat-critical {
    background: #3a1a1a;
    color: #f87171;
    border: 1px solid #f87171;
    border-radius: 6px;
    padding: 3px 10px;
    font-weight: 700;
    font-size: 0.82rem;
    display: inline-block;
    animation: pulse 1.5s infinite;
}

.mode-badge-synth {
    background: rgba(234, 179, 8, 0.14);
    border: 1px solid rgba(234, 179, 8, 0.55);
    color: #fde047;
    border-radius: 8px;
    padding: 0.45rem 0.8rem;
    font-size: 0.83rem;
    margin-bottom: 0.8rem;
}

.mode-badge-real {
    background: rgba(34, 197, 94, 0.14);
    border: 1px solid rgba(34, 197, 94, 0.55);
    color: #86efac;
    border-radius: 8px;
    padding: 0.45rem 0.8rem;
    font-size: 0.83rem;
    margin-bottom: 0.8rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(11, 19, 36, 0.75);
    padding: 6px;
    border-radius: 10px;
    border: 1px solid var(--border-subtle);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
    padding: 8px 14px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.30), rgba(0, 212, 255, 0.22)) !important;
    color: #ffffff !important;
    border: 1px solid rgba(56, 189, 248, 0.45) !important;
}
</style>
"""


def inject_theme() -> None:
    """Inject custom dark SOC CSS into the Streamlit application."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
