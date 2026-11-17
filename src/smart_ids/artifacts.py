"""Centralized artifact serialization and deserialization for Smart IDS v3.

Manages saving and loading of all model weights, fitted scalers, label encoders,
feature name schemas, and anomaly threshold configurations to/from the models directory.
Guarantees consistent contracts across notebooks and the Streamlit dashboard.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb

from smart_ids.config import MODELS_DIR

logger = logging.getLogger("smart_ids.artifacts")


def save_scaler(scaler: StandardScaler, path: Optional[Path] = None) -> Path:
    """Save fitted StandardScaler to disk."""
    dest = path or (MODELS_DIR / "scaler.pkl")
    joblib.dump(scaler, dest)
    logger.info("Saved StandardScaler to %s", dest)
    return dest


def load_scaler(path: Optional[Path] = None) -> StandardScaler:
    """Load fitted StandardScaler from disk."""
    src = path or (MODELS_DIR / "scaler.pkl")
    scaler = joblib.load(src)
    logger.info("Loaded StandardScaler from %s", src)
    return scaler


def save_label_encoder(le: LabelEncoder, path: Optional[Path] = None) -> Path:
    """Save fitted LabelEncoder to disk."""
    dest = path or (MODELS_DIR / "label_encoder.pkl")
    joblib.dump(le, dest)
    logger.info("Saved LabelEncoder to %s", dest)
    return dest


def load_label_encoder(path: Optional[Path] = None) -> LabelEncoder:
    """Load fitted LabelEncoder from disk."""
    src = path or (MODELS_DIR / "label_encoder.pkl")
    le = joblib.load(src)
    logger.info("Loaded LabelEncoder from %s", src)
    return le


def save_feature_names(feature_names: List[str], path: Optional[Path] = None) -> Path:
    """Save ordered feature name list to JSON."""
    dest = path or (MODELS_DIR / "feature_names.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    logger.info("Saved %d feature names to %s", len(feature_names), dest)
    return dest


def load_feature_names(path: Optional[Path] = None) -> List[str]:
    """Load ordered feature name list from JSON."""
    src = path or (MODELS_DIR / "feature_names.json")
    with open(src, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    logger.info("Loaded %d feature names from %s", len(feature_names), src)
    return feature_names


def save_anomaly_threshold(
    threshold: float,
    percentile: float = 95.0,
    path: Optional[Path] = None,
) -> Path:
    """Save calibrated Autoencoder anomaly threshold and metadata to JSON."""
    dest = path or (MODELS_DIR / "anomaly_threshold.json")
    data = {
        "anomaly_threshold": float(threshold),
        "percentile": float(percentile),
        "metric": "Mean Squared Error (MSE)",
        "calibration": f"{percentile}th percentile of benign validation reconstruction error",
    }
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved anomaly threshold (%.6f) to %s", threshold, dest)
    return dest


def load_anomaly_threshold(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load calibrated Autoencoder anomaly threshold configuration."""
    src = path or (MODELS_DIR / "anomaly_threshold.json")
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded anomaly threshold metadata from %s", src)
    return data


def save_xgboost_model(model: xgb.XGBClassifier, path: Optional[Path] = None) -> Path:
    """Save trained XGBoost model to JSON."""
    dest = path or (MODELS_DIR / "xgb_model.json")
    model.save_model(str(dest))
    logger.info("Saved XGBoost model to %s", dest)
    return dest


def load_xgboost_model(path: Optional[Path] = None) -> xgb.XGBClassifier:
    """Load trained XGBoost model from JSON."""
    src = path or (MODELS_DIR / "xgb_model.json")
    model = xgb.XGBClassifier()
    model.load_model(str(src))
    logger.info("Loaded XGBoost model from %s", src)
    return model


def load_all_artifacts(
    models_dir: Path = MODELS_DIR,
) -> Dict[str, Any]:
    """Load all available models and preprocessors defensively.

    Returns a dictionary of loaded objects. Missing files will simply
    be omitted rather than raising an uncaught exception, allowing
    graceful degradation.

    Returns:
        Dict mapping artifact keys ('MLP', 'CNN', 'Autoencoder', 'XGBoost',
        'scaler', 'label_encoder', 'feature_names', 'threshold_data') to objects.
    """
    artifacts: Dict[str, Any] = {}

    # 1. Keras MLP
    mlp_paths = [models_dir / "best_mlp.keras", models_dir / "mlp_model.keras"]
    for p in mlp_paths:
        if p.exists():
            try:
                artifacts["MLP"] = keras.models.load_model(str(p))
                break
            except Exception as e:
                logger.error("Failed to load MLP from %s: %s", p, e)

    # 2. Keras CNN
    cnn_paths = [models_dir / "best_cnn.keras", models_dir / "cnn_model.keras"]
    for p in cnn_paths:
        if p.exists():
            try:
                artifacts["CNN"] = keras.models.load_model(str(p))
                break
            except Exception as e:
                logger.error("Failed to load CNN from %s: %s", p, e)

    # 3. Keras Autoencoder
    ae_paths = [models_dir / "best_autoencoder.keras", models_dir / "autoencoder.keras"]
    for p in ae_paths:
        if p.exists():
            try:
                artifacts["Autoencoder"] = keras.models.load_model(str(p))
                break
            except Exception as e:
                logger.error("Failed to load Autoencoder from %s: %s", p, e)

    # 4. XGBoost
    xgb_p = models_dir / "xgb_model.json"
    if xgb_p.exists():
        try:
            artifacts["XGBoost"] = load_xgboost_model(xgb_p)
        except Exception as e:
            logger.error("Failed to load XGBoost from %s: %s", xgb_p, e)

    # 5. Preprocessors and Metadata
    scaler_p = models_dir / "scaler.pkl"
    if scaler_p.exists():
        artifacts["scaler"] = load_scaler(scaler_p)

    le_p = models_dir / "label_encoder.pkl"
    if le_p.exists():
        artifacts["label_encoder"] = load_label_encoder(le_p)

    feat_p = models_dir / "feature_names.json"
    if feat_p.exists():
        artifacts["feature_names"] = load_feature_names(feat_p)

    thresh_p = models_dir / "anomaly_threshold.json"
    if thresh_p.exists():
        artifacts["threshold_data"] = load_anomaly_threshold(thresh_p)

    comp_p = models_dir / "model_comparison.csv"
    if comp_p.exists():
        artifacts["comparison_df"] = pd.read_csv(comp_p)

    logger.info("Artifact loading summary: %d artifacts loaded successfully.", len(artifacts))
    return artifacts
