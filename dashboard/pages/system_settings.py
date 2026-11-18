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
        pointer_set = set(artifacts.lfs_pointers)

        def _status(is_loaded: bool, fname: str) -> str:
            if is_loaded:
                return "Loaded"
            if fname in pointer_set:
                return "LFS Pointer (run git lfs pull)"
            return "Missing"

        inv_rows = [
            {
                "Artifact": "MLP Classifier (`best_mlp.keras`)",
                "Status": _status("MLP" in artifacts.models, "best_mlp.keras"),
            },
            {
                "Artifact": "1D-CNN Classifier (`best_cnn.keras`)",
                "Status": _status("CNN" in artifacts.models, "best_cnn.keras"),
            },
            {
                "Artifact": "Autoencoder (`best_autoencoder.keras`)",
                "Status": _status("Autoencoder" in artifacts.models, "best_autoencoder.keras"),
            },
            {
                "Artifact": "XGBoost Classifier (`xgb_model.json`)",
                "Status": _status("XGBoost" in artifacts.models, "xgb_model.json"),
            },
            {
                "Artifact": "StandardScaler (`scaler.pkl`)",
                "Status": _status(artifacts.scaler is not None, "scaler.pkl"),
            },
            {
                "Artifact": "MedianImputer (`imputer.pkl`)",
                "Status": _status(artifacts.imputer is not None, "imputer.pkl"),
            },
            {
                "Artifact": "5-Sigma Clipper (`outlier_clipper.pkl`)",
                "Status": _status(artifacts.outlier_clipper is not None, "outlier_clipper.pkl"),
            },
            {
                "Artifact": "LabelEncoder (`label_encoder.pkl`)",
                "Status": _status(artifacts.label_encoder is not None, "label_encoder.pkl"),
            },
            {
                "Artifact": f"Feature Schema (`feature_names.json`, {len(artifacts.feature_names)} cols)",
                "Status": _status(bool(artifacts.feature_names), "feature_names.json"),
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
