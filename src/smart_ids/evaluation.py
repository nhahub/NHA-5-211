"""Model evaluation, latency benchmarking, confusion matrices, and ROC analysis.

Provides unified evaluation routines across multi-class supervised models (weighted metrics)
and the unsupervised Autoencoder (binary anomaly metrics), with per-sample microsecond
latency benchmarking and publication-quality diagnostic plots.
"""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras

from smart_ids.config import (
    CLASS_NAMES,
    LATENCY_BENCHMARK_RUNS,
    MODELS_DIR,
)

logger = logging.getLogger("smart_ids.evaluation")


def measure_latency(
    model: Any,
    X: np.ndarray,
    model_type: str = "keras",
    n_runs: int = LATENCY_BENCHMARK_RUNS,
) -> float:
    """Benchmark average per-sample inference latency in microseconds.

    Executes 1 warm-up run (to trigger JIT compilation and warm caches),
    followed by n_runs repeated trials against a single representative sample.

    Args:
        model: Trained model (Keras or XGBoost).
        X: Feature matrix from which the single sample is taken.
        model_type: 'keras', 'cnn', 'ae', or 'xgboost'.
        n_runs: Number of timed repetitions (default 100).

    Returns:
        Mean inference latency in microseconds (us).
    """
    single = X[:1]
    if model_type == "cnn" and single.ndim == 2:
        single = single.reshape(1, single.shape[1], 1)

    # 1. Untimed warm-up run
    if model_type in ("keras", "cnn", "ae"):
        _ = model.predict(single, verbose=0)
    else:
        _ = model.predict(single)

    # 2. Timed repetitions
    times: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        if model_type in ("keras", "cnn", "ae"):
            _ = model.predict(single, verbose=0)
        else:
            _ = model.predict(single)
        times.append((time.perf_counter() - t0) * 1e6)

    avg_latency = float(np.mean(times))
    logger.info("Latency benchmark (%s): %.2f us (mean over %d runs)", model_type, avg_latency, n_runs)
    return avg_latency


def compute_reconstruction_error(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Compute per-sample Mean Squared Error between input and reconstruction.

    Args:
        model: Trained Autoencoder.
        X: Feature matrix.

    Returns:
        1D array of reconstruction errors (one per flow).
    """
    X_recon = model.predict(X, verbose=0)
    return np.mean(np.square(X - X_recon), axis=1)


def calibrate_anomaly_threshold(
    model: keras.Model,
    X_benign_val: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Calibrate zero-day anomaly threshold on benign validation errors.

    Sets threshold at the specified percentile of the benign validation error distribution,
    strictly bounding expected benign false positives at (100 - percentile)%.

    Args:
        model: Trained Autoencoder.
        X_benign_val: Benign-only validation feature matrix.
        percentile: Cutoff percentile (default 95.0).

    Returns:
        Calibrated scalar threshold.
    """
    benign_errors = compute_reconstruction_error(model, X_benign_val)
    threshold = float(np.percentile(benign_errors, percentile))
    logger.info("Calibrated Anomaly Threshold (%.1f percentile): %.6f", percentile, threshold)
    return threshold


def evaluate_supervised(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latency_us: float = 0.0,
) -> Dict[str, float]:
    """Compute multi-class evaluation metrics using weighted averaging.

    Weighted averaging is used because the test set is naturally imbalanced.

    Args:
        y_true: Integer ground-truth labels.
        y_pred: Integer predicted labels.
        latency_us: Benchmarked per-sample latency in microseconds.

    Returns:
        Dict with accuracy, precision, recall, f1, and latency_us.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Latency (us)": latency_us,
        "Evaluation Type": "Multi-class (Weighted)",
    }


def evaluate_autoencoder_binary(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    latency_us: float = 0.0,
) -> Dict[str, float]:
    """Compute binary anomaly detection metrics for the Autoencoder.

    Ground truth: 0 = Benign, 1 = Attack (Any anomaly).

    Args:
        y_true_binary: Binary ground truth (0 = Benign, 1 = Attack).
        y_pred_binary: Binary predicted anomaly flags (0 = Normal, 1 = Anomaly).
        latency_us: Benchmarked per-sample latency in microseconds.

    Returns:
        Dict with binary accuracy, precision, recall, f1, and latency_us.
    """
    acc = float(accuracy_score(y_true_binary, y_pred_binary))
    prec = float(precision_score(y_true_binary, y_pred_binary, average="binary", zero_division=0))
    rec = float(recall_score(y_true_binary, y_pred_binary, average="binary", zero_division=0))
    f1 = float(f1_score(y_true_binary, y_pred_binary, average="binary", zero_division=0))

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Latency (us)": latency_us,
        "Evaluation Type": "Binary (Zero-Day Anomaly vs Benign)",
    }


def plot_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a 2x2 grid of row-normalized confusion matrices for all 4 models.

    Args:
        confusion_matrices: Dict mapping model name to 2D confusion matrix array.
        class_names: List of 8 class names.
        save_path: Optional destination file path.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    model_names = list(confusion_matrices.keys())

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        cm = confusion_matrices[name]

        # Row-normalize to show conditional probabilities P(pred | true)
        row_sums = cm.sum(axis=1)[:, np.newaxis] + 1e-8
        cm_norm = cm.astype(float) / row_sums

        is_binary = cm.shape[0] == 2
        labels = ["Benign", "Attack"] if is_binary else class_names

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=True,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 8 if not is_binary else 12},
        )
        ax.set_title(f"{name} Confusion Matrix {'(Binary Anomaly)' if is_binary else '(Normalized)'}", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved confusion matrices plot to %s", save_path)
    return fig


def plot_roc_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    title: str,
    ax: plt.Axes,
    binary: bool = False,
) -> None:
    """Plot per-class One-vs-Rest ROC curves and macro-average AUC.

    Args:
        y_true: True integer labels (or binary 0/1 if binary=True).
        y_proba: Predicted probability matrix or 1D continuous reconstruction error.
        class_names: List of class names.
        title: Subplot title.
        ax: Matplotlib Axes.
        binary: Set True for Autoencoder continuous anomaly score.
    """
    if binary:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, color="#ef4444", label=f"Anomaly AUC = {roc_auc:.3f}")
    else:
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        valid_curves = 0

        # Plot individual per-class curves
        for i in range(min(y_proba.shape[1], len(class_names))):
            if len(np.unique(y_bin[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=1.2, alpha=0.75, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
                mean_tpr += np.interp(all_fpr, fpr, tpr)
                valid_curves += 1

        if valid_curves > 0:
            mean_tpr /= valid_curves
            macro_auc = auc(all_fpr, mean_tpr)
            ax.plot(all_fpr, mean_tpr, "k--", lw=2.5, label=f"Macro-avg (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], ":", color="gray", lw=1.5, label="Random Guess (AUC=0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
