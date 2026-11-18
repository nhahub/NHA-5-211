"""SMART IDS v3 — Operational SOC Streamlit Dashboard Entry Point."""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure project root is importable when launched via `streamlit run app.py`
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import streamlit as st

from dashboard.components.sidebar import render_sidebar
from dashboard.data_access import load_cached_artifacts, load_evaluation_summary
from dashboard.pages.active_response import render_active_response_page
from dashboard.pages.dataset_explorer import render_dataset_explorer_page
from dashboard.pages.explainability import render_explainability_page
from dashboard.pages.model_performance import render_model_performance_page
from dashboard.pages.overview import render_overview_page
from dashboard.pages.predict_flow import render_predict_flow_page
from dashboard.pages.system_settings import render_system_settings_page
from dashboard.styles.theme import inject_theme
from src.config import get_app_config

app_cfg = get_app_config()
st.set_page_config(
    page_title=app_cfg.dashboard.page_title,
    page_icon=app_cfg.dashboard.page_icon,
    layout=app_cfg.dashboard.layout,
    initial_sidebar_state="expanded",
)

inject_theme()
sidebar_state = render_sidebar()

st.markdown(
    """
    <div class="soc-banner">
        <h1>🛡️ SMART IDS v3 — Hybrid Multi-Modal SOC Platform</h1>
        <p>Supervised Attack Attribution (XGBoost · MLP · 1D-CNN) &nbsp;|&nbsp; Unsupervised Zero-Day Autoencoder &nbsp;|&nbsp; SHAP Explainability &nbsp;|&nbsp; Dry-Run Active Mitigation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

eval_summary = load_evaluation_summary()
artifacts = load_cached_artifacts()
if artifacts.lfs_pointers:
    st.warning(
        "**Model Artifacts Not Fetched (Git LFS Pointers):** Detected un-pulled Git LFS pointer stub(s) "
        f"(`{', '.join(artifacts.lfs_pointers[:5])}`). Run `git lfs pull` (or `python scripts/train.py --run-eval`) "
        "to enable live detection, batch CSV scoring, and interactive SHAP; evaluation reports and synthetic SOC demo "
        "telemetry remain available below."
    )
elif not artifacts.models:
    st.warning(
        "**Model Binaries Absent (`models/`):** No trained detection models are currently loaded. Run `git lfs pull` "
        "or `python scripts/train.py --run-eval` to populate `models/`. Historical evaluation reports and synthetic SOC "
        "demo telemetry remain available below."
    )

src_info = eval_summary.get("dataset_source", {})
if src_info:
    if src_info.get("is_synthetic"):
        st.markdown(
            '<div class="mode-badge-synth">⚠️ <strong>SYNTHETIC DEMONSTRATION MODE:</strong> '
            "Models were trained on the schema-faithful synthetic CICIDS2017 demonstration corpus because "
            "<code>data/raw/</code> had no raw CSVs. Place official CICIDS2017 CSVs in <code>data/raw/</code> "
            "and run <code>python scripts/train.py --run-eval</code> for benchmark results.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="mode-badge-real">✅ <strong>CICIDS2017 BENCHMARK MODE:</strong> '
            f"Trained on {src_info.get('sampled_row_count', 0):,} genuine CICIDS2017 flow records.</div>",
            unsafe_allow_html=True,
        )

t_overview, t_data, t_perf, t_pred, t_xai, t_resp, t_sys = st.tabs(
    [
        "🏠 Overview",
        "📊 Dataset Explorer",
        "📈 Model Performance",
        "🔮 Predict Flow",
        "🧠 Explainability",
        "🛡️ Active Response",
        "⚙️ System Settings",
    ]
)

with t_overview:
    overview_logs = render_overview_page(sidebar_state)

with t_data:
    render_dataset_explorer_page()

with t_perf:
    render_model_performance_page()

with t_pred:
    render_predict_flow_page(sidebar_state)

with t_xai:
    render_explainability_page()

with t_resp:
    render_active_response_page(sidebar_state, overview_logs)

with t_sys:
    render_system_settings_page(sidebar_state)
