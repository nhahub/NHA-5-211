"""Cached model, evaluation, dataset, and telemetry accessors for the dashboard."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.config import get_app_config, get_model_config
from src.data.loader import CICIDS2017_FEATURES
from src.detection.inference import UnifiedDetectionEngine
from src.detection.threat_scoring import get_threat_level
from src.response.response_engine import ActiveResponseEngine
from src.utils.model_registry import LoadedArtifacts, ModelRegistry
from src.utils.reproducibility import HardwareProfile, configure_hardware


@st.cache_resource(show_spinner=False)
def get_cached_hardware_profile() -> HardwareProfile:
    """Detect TensorFlow/XGBoost GPU availability once per server lifetime."""
    model_cfg = get_model_config()
    return configure_hardware(use_gpu=model_cfg.hardware.use_gpu)


@st.cache_resource(show_spinner="Loading SMART IDS v3 models and scalers...")
def load_cached_artifacts() -> LoadedArtifacts:
    """Load persisted models, scalers, encoders, and manifest once per server session."""
    app_cfg = get_app_config()
    registry = ModelRegistry(
        models_dir=app_cfg.paths.resolve("models_dir"),
        artifacts_dir=app_cfg.paths.resolve("artifacts_dir"),
        xgboost_profile=app_cfg.deployment.xgboost_imbalance_profile,
    )
    return registry.load_all(load_keras=True)


def get_response_engine() -> ActiveResponseEngine:
    """Retrieve or initialize the session-scoped ActiveResponseEngine."""
    if "response_engine" not in st.session_state:
        app_cfg = get_app_config()
        engine = ActiveResponseEngine(
            dry_run=app_cfg.response.dry_run,
            rate_limit_pps=app_cfg.response.rate_limit_pps,
            log_file=app_cfg.paths.resolve("response_log_file"),
            history_file=app_cfg.paths.resolve("incident_history_file"),
            enforce_academic_dry_run_only=True,
        )
        engine.load_history()
        st.session_state["response_engine"] = engine
    return st.session_state["response_engine"]


def get_detection_engine() -> UnifiedDetectionEngine:
    """Build UnifiedDetectionEngine reusing cached artifacts and session response engine."""
    artifacts = load_cached_artifacts()
    model_cfg = get_model_config()
    return UnifiedDetectionEngine(
        artifacts=artifacts,
        ensemble_cfg=model_cfg.ensemble,
        response_engine=get_response_engine(),
        calibrators=artifacts.calibrators,
    )


@st.cache_data(show_spinner=False)
def load_evaluation_summary() -> dict[str, Any]:
    """Load persisted evaluation_summary.json if available."""
    app_cfg = get_app_config()
    path = app_cfg.paths.resolve("reports_dir") / "evaluation_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_model_comparison_df() -> pd.DataFrame:
    """Load persisted model_comparison.csv if available."""
    app_cfg = get_app_config()
    path = app_cfg.paths.resolve("reports_dir") / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict[str, Any]:
    """Load persisted models/model_metadata.json (XGBoost deployment profiles) if available."""
    app_cfg = get_app_config()
    path = app_cfg.paths.resolve("models_dir") / "model_metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_training_histories() -> dict[str, dict[str, list[float]]]:
    """Load persisted training_histories.json if available."""
    app_cfg = get_app_config()
    path = app_cfg.paths.resolve("models_dir") / "training_histories.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_preprocessing_audit() -> dict[str, Any]:
    """Load preprocessing_audit.json if available."""
    app_cfg = get_app_config()
    path = app_cfg.paths.resolve("processed_data_dir") / "preprocessing_audit.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_shap_summary() -> dict[str, Any]:
    """Load artifacts/shap_summary.json if available."""
    app_cfg = get_app_config()
    path = app_cfg.paths.resolve("artifacts_dir") / "shap_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_test_split_sample(max_rows: int = 2000) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load a reproducible slice of the held-out test set + metadata for live SOC evaluation."""
    app_cfg = get_app_config()
    proc_dir = app_cfg.paths.resolve("processed_data_dir")
    npz_path = proc_dir / "preprocessed_splits.npz"
    meta_path = proc_dir / "test_metadata.csv"
    if not npz_path.exists():
        return np.empty((0, 0)), np.empty((0,)), pd.DataFrame()

    data = np.load(npz_path)
    X_test = data["X_test_scaled"]
    y_test = data["y_test"]
    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    n = min(max_rows, len(X_test))
    return X_test[:n], y_test[:n], meta.iloc[:n].reset_index(drop=True) if not meta.empty else meta


@st.cache_data(show_spinner=False)
def build_test_detection_logs(
    model_choice: str = "Hybrid Ensemble",
    max_rows: int = 800,
    min_actionable_confidence: float = 0.40,
) -> pd.DataFrame:
    """Run real cached inference over held-out test split samples for the SOC Overview."""
    artifacts = load_cached_artifacts()
    X_test, y_test, meta = load_test_split_sample(max_rows=max_rows)
    if len(X_test) == 0 or not artifacts.models:
        return pd.DataFrame()

    engine = get_detection_engine()
    class_names = engine.class_names

    # Invert scaling to obtain realistic unscaled DataFrame or score directly from scaled features
    anom_res = engine.anomaly_detector.score(X_test)
    sup_avail = [m for m in ("XGBoost", "MLP", "CNN") if m in artifacts.models]
    if model_choice in sup_avail:
        probas = engine._predict_supervised_proba(X_test, model_choice)
        pred_idx = np.argmax(probas, axis=1)
        confs = np.max(probas, axis=1)
        labels = [class_names[int(i)] for i in pred_idx]
        used = model_choice
    elif model_choice == "Autoencoder":
        labels = [
            "Zero-Day Anomaly (Novel)" if bool(f) else class_names[engine.benign_index]
            for f in anom_res.is_anomaly
        ]
        confs = np.where(
            anom_res.is_anomaly,
            np.clip(0.5 + 0.49 * (1.0 - np.exp(-(anom_res.severity_ratio - 1.0))), 0.5, 0.9999),
            np.clip(1.0 - 0.5 * anom_res.severity_ratio, 0.5, 0.9999),
        )
        used = "Autoencoder"
    elif sup_avail:
        probas_dict = {m: engine._predict_supervised_proba(X_test, m) for m in sup_avail}
        decision = engine.ensemble.decide(
            model_probas=probas_dict,
            anomaly_scores=anom_res.scores,
            anomaly_threshold=anom_res.threshold,
        )
        confs = decision.confidences
        labels = decision.predicted_labels
        used = f"Hybrid Ensemble ({decision.strategy_used})"
    else:
        return pd.DataFrame()

    n = len(X_test)
    src_ips = (
        meta["Source IP"].astype(str).tolist()
        if not meta.empty and "Source IP" in meta.columns
        else [f"192.168.10.{(i % 45) + 10}" for i in range(n)]
    )
    dst_ports = (
        pd.to_numeric(meta["Destination Port"], errors="coerce").fillna(80).astype(int).tolist()
        if not meta.empty and "Destination Port" in meta.columns
        else [80] * n
    )
    timestamps = (
        meta["Timestamp"].astype(str).tolist()
        if not meta.empty and "Timestamp" in meta.columns
        else [
            (datetime(2017, 7, 7, 9, 0, tzinfo=UTC) + timedelta(seconds=i * 2)).isoformat() for i in range(n)
        ]
    )
    true_labels = (
        meta["True_Label"].astype(str).tolist()
        if not meta.empty and "True_Label" in meta.columns
        else [class_names[int(idx)] for idx in y_test]
    )

    rows = []
    for i in range(n):
        t_lvl = get_threat_level(
            confidence=float(confs[i]),
            attack_class=labels[i],
            is_anomaly=bool(anom_res.is_anomaly[i]),
            anomaly_ratio=float(anom_res.severity_ratio[i]),
            min_actionable_confidence=min_actionable_confidence,
        )
        action = {"Low": "LOG", "Medium": "ALERT", "High": "RATE_LIMIT", "Critical": "BLOCK"}[t_lvl]
        rows.append(
            {
                "timestamp": timestamps[i],
                "src_ip": src_ips[i],
                "dst_port": dst_ports[i],
                "true_label": true_labels[i],
                "attack_class": labels[i],
                "confidence": round(float(confs[i]), 4),
                "anomaly_score": round(float(anom_res.scores[i]), 4),
                "is_anomaly": bool(anom_res.is_anomaly[i]),
                "threat_level": t_lvl,
                "model_used": used,
                "action": action,
            }
        )
    return pd.DataFrame(rows)


def generate_demo_logs(n_rows: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate explicitly labeled synthetic SOC log telemetry when no test split exists."""
    rng = np.random.default_rng(seed)
    classes = [
        "BENIGN",
        "DoS Hulk",
        "PortScan",
        "DDoS",
        "DoS GoldenEye",
        "FTP-Patator",
        "SSH-Patator",
        "Web Attack",
    ]
    probs = [0.65, 0.10, 0.08, 0.06, 0.03, 0.03, 0.03, 0.02]
    models = ["Hybrid Ensemble", "XGBoost", "MLP", "1D-CNN", "Autoencoder"]
    now = datetime.now(UTC)

    rows = []
    for i in range(n_rows):
        cls_name = str(rng.choice(classes, p=probs))
        conf = float(np.clip(rng.normal(0.88 if cls_name != "BENIGN" else 0.94, 0.11), 0.25, 0.999))
        anom_flag = bool(cls_name != "BENIGN" and rng.random() > 0.35)
        t_lvl = get_threat_level(
            conf, attack_class=cls_name, is_anomaly=anom_flag, anomaly_ratio=2.2 if anom_flag else 0.4
        )
        action = {"Low": "LOG", "Medium": "ALERT", "High": "RATE_LIMIT", "Critical": "BLOCK"}[t_lvl]
        rows.append(
            {
                "timestamp": (now - timedelta(seconds=(n_rows - i) * 3)).isoformat(),
                "src_ip": f"192.168.10.{rng.integers(10, 50)}"
                if cls_name == "BENIGN"
                else f"203.0.113.{rng.integers(10, 35)}",
                "dst_port": int(rng.choice([80, 443, 21, 22, 53, 8080])),
                "attack_class": cls_name,
                "confidence": round(conf, 4),
                "anomaly_score": round(
                    float(rng.uniform(1.1, 4.8) if anom_flag else rng.uniform(0.05, 0.7)), 4
                ),
                "is_anomaly": anom_flag,
                "threat_level": t_lvl,
                "model_used": str(rng.choice(models)),
                "action": action,
            }
        )
    return pd.DataFrame(rows)


def load_sample_raw_dataset(max_rows: int = 5000) -> tuple[pd.DataFrame, str]:
    """Load a raw/synthetic sample CSV for Dataset Explorer inspection."""
    app_cfg = get_app_config()
    raw_dir = app_cfg.paths.resolve("raw_data_dir")
    syn_path = app_cfg.paths.resolve("synthetic_data_dir") / "cicids2017_synthetic_demo.csv"

    raw_csvs = sorted(raw_dir.rglob("*.csv")) if raw_dir.exists() else []
    if raw_csvs:
        df = pd.read_csv(raw_csvs[0], nrows=max_rows, low_memory=False)
        df.columns = df.columns.astype(str).str.strip()
        return df, f"CICIDS2017 Raw ({raw_csvs[0].name}, first {len(df):,} rows)"

    if syn_path.exists():
        df = pd.read_csv(syn_path, nrows=max_rows, low_memory=False)
        df.columns = df.columns.astype(str).str.strip()
        return df, f"Synthetic Demonstration Corpus ({syn_path.name}, first {len(df):,} rows)"

    return pd.DataFrame(columns=CICIDS2017_FEATURES), "No dataset found on disk"


def get_figure_path(filename: str) -> Path | None:
    """Resolve figure path from reports/figures/ if present."""
    app_cfg = get_app_config()
    p = app_cfg.paths.resolve("figures_dir") / filename
    return p if p.exists() else None
