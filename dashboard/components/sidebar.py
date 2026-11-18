"""Sidebar configuration controls, data source status, and model availability widget."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from dashboard.data_access import load_cached_artifacts
from src.config import get_app_config


def render_sidebar() -> dict[str, Any]:
    """Render left sidebar control surface and return active UI settings."""
    app_cfg = get_app_config()
    artifacts = load_cached_artifacts()

    with st.sidebar:
        st.markdown("### 🛡️ SMART IDS v3 Control")
        st.caption("Multi-Model IDS & Zero-Day Anomaly Tracker")
        st.divider()

        st.markdown("#### ⚙️ Configuration")
        dry_run_mode = st.toggle(
            "Dry-Run Safety Mode",
            value=True,
            help="Academic safety lock: simulates iptables DROP commands without altering OS firewall rules.",
        )

        available_models = ["Hybrid Ensemble"] + [
            m for m in ("XGBoost", "MLP", "CNN", "Autoencoder") if m in artifacts.models
        ]
        active_model = st.selectbox(
            "Active Detection Model",
            options=available_models,
            index=0,
        )

        confidence_threshold = st.slider(
            "Actionable Confidence Floor",
            min_value=0.0,
            max_value=1.0,
            value=float(app_cfg.dashboard.default_confidence_threshold),
            step=0.05,
            help="Attack alerts with confidence below this floor stay in passive LOG tier.",
        )

        default_ae_thresh = float(artifacts.anomaly_threshold)
        anomaly_threshold = st.number_input(
            "Autoencoder Anomaly Threshold (MSE)",
            min_value=0.01,
            max_value=50.0,
            value=round(default_ae_thresh, 4),
            step=0.05,
            help="Calibrated at the 95th percentile of benign validation reconstruction error.",
        )

        st.divider()
        st.markdown("#### 📂 Telemetry Source")
        data_source_mode = st.radio(
            "Overview Data Feed",
            options=["Test Evaluation Split", "Upload Flow CSV", "Demo Telemetry (Synthetic)"],
            index=0,
        )

        st.divider()
        st.markdown("#### 🧠 Loaded Models")
        model_files = {
            "MLP": "best_mlp.keras",
            "CNN": "best_cnn.keras",
            "Autoencoder": "best_autoencoder.keras",
            "XGBoost": "xgb_model.json",
        }
        pointer_set = set(artifacts.lfs_pointers)
        for m_name, m_file in model_files.items():
            if m_name in artifacts.models:
                st.success(f"✅ {m_name} — Loaded")
            elif m_file in pointer_set:
                st.warning(f"⚠️ {m_name} — Git LFS pointer (run `git lfs pull`)")
            else:
                st.warning(f"⚠️ {m_name} — Not found")

        pre_ok = artifacts.scaler is not None and artifacts.label_encoder is not None
        if pre_ok:
            st.caption(f"Scaler & Encoder ready ({len(artifacts.feature_names)} features)")
        elif pointer_set:
            st.error(
                "Preprocessing `.pkl` files are Git LFS pointers. Run `git lfs pull` or `python scripts/train.py`."
            )
        else:
            st.error("Preprocessing artifacts missing. Run `git lfs pull` or `python scripts/train.py`.")

        st.divider()
        st.caption(f"Last UI Sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return {
        "dry_run_mode": dry_run_mode,
        "active_model": active_model,
        "confidence_threshold": confidence_threshold,
        "anomaly_threshold": float(anomaly_threshold),
        "data_source_mode": data_source_mode,
    }
