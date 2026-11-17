"""Model explainability and feature attribution via SHapley Additive exPlanations (SHAP).

Implements game-theoretic feature attribution across two paradigms:
1. Fast, exact TreeExplainer for XGBoost (global importance bar and attack beeswarm plots).
2. Model-agnostic KernelExplainer for deep MLP (sample-efficient perturbation).
3. Single-sample Waterfall decomposition isolating individual feature contributions for critical detections.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb

from smart_ids.config import CLASS_NAMES, MODELS_DIR

logger = logging.getLogger("smart_ids.explain")


def compute_shap_xgboost(
    xgb_model: xgb.XGBClassifier,
    X_test: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str],
    n_samples: int = 500,
) -> Tuple[shap.TreeExplainer, Any, pd.DataFrame]:
    """Compute exact SHAP values for the XGBoost ensemble via TreeExplainer.

    Args:
        xgb_model: Trained XGBClassifier.
        X_test: Test features.
        feature_names: Human-readable feature name strings.
        n_samples: Number of samples to evaluate (default 500).

    Returns:
        Tuple of (explainer, shap_values, X_shap_df).
    """
    n_eval = min(n_samples, len(X_test))
    if isinstance(X_test, pd.DataFrame):
        X_sub = X_test.iloc[:n_eval].copy()
    else:
        X_sub = pd.DataFrame(X_test[:n_eval], columns=feature_names)

    logger.info("Computing exact TreeExplainer SHAP values for XGBoost on %d samples...", n_eval)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sub)

    return explainer, shap_values, X_sub


def compute_shap_mlp(
    mlp_model: keras.Model,
    X_test: np.ndarray,
    feature_names: List[str],
    n_background: int = 10,
    n_samples: int = 20,
) -> Tuple[shap.KernelExplainer, np.ndarray, np.ndarray]:
    """Compute model-agnostic SHAP values for the deep MLP via KernelExplainer.

    Scales down background reference and sample counts to keep computation
    interactive and tractable. Defensively handles structural shape differences
    in SHAP multi-class outputs.

    Args:
        mlp_model: Trained Keras MLP.
        X_test: Test feature matrix.
        feature_names: List of feature names.
        n_background: Reference background size (default 10).
        n_samples: Number of test samples to explain (default 20).

    Returns:
        Tuple of (explainer, mean_abs_shap_per_feature, shap_values_raw).
    """
    n_eval = min(n_samples, len(X_test))
    background = shap.sample(X_test, n_background)

    def mlp_predict_wrapper(x: np.ndarray) -> np.ndarray:
        return mlp_model.predict(x, verbose=0)

    logger.info(
        "Computing KernelExplainer SHAP for MLP (background=%d, samples=%d)...",
        n_background,
        n_eval,
    )
    explainer = shap.KernelExplainer(mlp_predict_wrapper, background)
    shap_values = explainer.shap_values(X_test[:n_eval])

    # Defensive aggregation across output classes
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    mean_abs_shap = np.atleast_1d(mean_abs_shap).flatten()
    return explainer, mean_abs_shap, shap_values


def generate_waterfall_plot(
    explainer: Any,
    sample_vector: np.ndarray,
    feature_names: List[str],
    target_class_idx: int = 1,
    max_display: int = 15,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate a single-sample waterfall decomposition for a critical detection.

    Decomposes the prediction into signed, additive feature contributions
    starting from the base expected value.

    Args:
        explainer: SHAP TreeExplainer or KernelExplainer.
        sample_vector: 1D array of single flow's scaled features.
        feature_names: List of feature names.
        target_class_idx: Class index being explained (default 1 for first attack).
        max_display: Top N features to show in the waterfall (default 15).
        save_path: Destination path for saving figure.

    Returns:
        Matplotlib Figure.
    """
    sample_2d = sample_vector.reshape(1, -1)
    shap_vals = explainer.shap_values(sample_2d)

    # Extract single sample vector for target class
    if isinstance(shap_vals, list):
        single_shap = shap_vals[target_class_idx][0]
    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
        single_shap = shap_vals[0, :, target_class_idx]
    else:
        single_shap = shap_vals[0]

    # Extract base value
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[target_class_idx]

    explanation = shap.Explanation(
        values=single_shap,
        base_values=float(base_val),
        data=sample_vector,
        feature_names=feature_names,
    )

    fig = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, max_display=max_display, show=False)
    plt.title(f"SHAP Waterfall Attribution: Class '{CLASS_NAMES[target_class_idx]}'", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved SHAP waterfall plot to %s", save_path)
    return fig
