"""Tab 7: System Settings — Loaded Models, Hardware Profile, Versions, and Threshold Config."""

from __future__ import annotations

import platform
import sys
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.data_access import (
    get_cached_hardware_profile,
    load_cached_artifacts,
    load_evaluation_summary,
)
from src.config import get_app_config, get_model_config


def render_system_settings_page(sidebar_state: dict[str, Any]) -> None:
    """Render system diagnostics, loaded artifacts inventory, and configuration details."""
    st.subheader("System Settings, Model Registry & Runtime Diagnostics")

    app_cfg = get_app_config()
    model_cfg = get_model_config()
    artifacts = load_cached_artifacts()
    eval_summary = load_evaluation_summary()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🧠 Persisted Model & Artifact Inventory")
        inv_rows = [
            {
                "Artifact": "MLP Classifier (`best_mlp.keras`)",
                "Status": "Loaded" if "MLP" in artifacts.models else "Missing",
            },
            {
                "Artifact": "1D-CNN Classifier (`best_cnn.keras`)",
                "Status": "Loaded" if "CNN" in artifacts.models else "Missing",
            },
            {
                "Artifact": "Autoencoder (`best_autoencoder.keras`)",
                "Status": "Loaded" if "Autoencoder" in artifacts.models else "Missing",
            },
            {
                "Artifact": "XGBoost Classifier (`xgb_model.json`)",
                "Status": "Loaded" if "XGBoost" in artifacts.models else "Missing",
            },
            {
                "Artifact": "StandardScaler (`scaler.pkl`)",
                "Status": "Loaded" if artifacts.scaler is not None else "Missing",
            },
            {
                "Artifact": "MedianImputer (`imputer.pkl`)",
                "Status": "Loaded" if artifacts.imputer is not None else "Missing",
            },
            {
                "Artifact": "5-Sigma Clipper (`outlier_clipper.pkl`)",
                "Status": "Loaded" if artifacts.outlier_clipper is not None else "Missing",
            },
            {
                "Artifact": "LabelEncoder (`label_encoder.pkl`)",
                "Status": "Loaded" if artifacts.label_encoder is not None else "Missing",
            },
            {
                "Artifact": f"Feature Schema (`feature_names.json`, {len(artifacts.feature_names)} cols)",
                "Status": "Loaded" if artifacts.feature_names else "Missing",
            },
        ]
        st.dataframe(pd.DataFrame(inv_rows), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("#### ⚙️ Active Runtime & Safety Configuration")
        hw = get_cached_hardware_profile()
        runtime_info = {
            "project_name": app_cfg.project.name,
            "version": app_cfg.project.version,
            "python_version": sys.version.split()[0],
            "os_platform": platform.platform(),
            "tensorflow_device": hw.tf_device,
            "xgboost_device": hw.xgb_device,
            "cuda_driver_detected": hw.cuda_driver_detected,
            "dry_run_mode": sidebar_state["dry_run_mode"],
            "selected_classifier": sidebar_state["active_model"],
            "confidence_threshold": sidebar_state["confidence_threshold"],
            "calibrated_anomaly_threshold": artifacts.anomaly_threshold,
            "active_anomaly_threshold": sidebar_state["anomaly_threshold"],
            "ensemble_strategy": model_cfg.ensemble.strategy,
        }
        st.json(runtime_info)

    if artifacts.anomaly_threshold_meta:
        st.markdown("#### 🧬 Calibrated Autoencoder Threshold Metadata")
        st.json(artifacts.anomaly_threshold_meta)

    if eval_summary.get("dataset_source"):
        st.markdown("#### 📂 Dataset Provenance & Manifest")
        st.json(eval_summary.get("dataset_source"))
