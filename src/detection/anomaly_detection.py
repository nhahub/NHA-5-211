"""Independent Autoencoder and classical benign-only anomaly detectors."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.config import AnomalyBaselinesConfig
from src.models.autoencoder import compute_reconstruction_error
from src.utils.logging import get_logger

logger = get_logger("smart_ids.detection.anomaly")


@dataclass
class AnomalyResult:
    """Per-batch or per-flow anomaly evaluation output."""

    scores: np.ndarray
    threshold: float
    is_anomaly: np.ndarray
    severity_ratio: np.ndarray


class AnomalyDetector:
    """Scores traffic flows using the benign-trained Autoencoder and calibrated threshold."""

    def __init__(self, autoencoder_model: Any | None, threshold: float = 1.0) -> None:
        self.model = autoencoder_model
        self.threshold = float(threshold)

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def score(self, X_scaled: np.ndarray, threshold_override: float | None = None) -> AnomalyResult:
        """Compute reconstruction MSE, anomaly flags, and normalized severity ratio."""
        thresh = float(threshold_override if threshold_override is not None else self.threshold)
        n = len(X_scaled)
        if self.model is None:
            zeros = np.zeros(n, dtype=np.float64)
            return AnomalyResult(
                scores=zeros,
                threshold=thresh,
                is_anomaly=np.zeros(n, dtype=bool),
                severity_ratio=zeros,
            )

        errors = compute_reconstruction_error(self.model, X_scaled)
        is_anom = errors > thresh
        ratio = errors / max(thresh, 1e-9)
        return AnomalyResult(
            scores=errors,
            threshold=thresh,
            is_anomaly=is_anom,
            severity_ratio=ratio,
        )


@dataclass
class ClassicalAnomalyBaseline:
    """Benign-only classical outlier detector wrapped with higher=more-anomalous scoring."""

    name: str
    estimator: Any
    threshold: float
    train_samples_used: int
    fit_duration_sec: float = 0.0

    def score(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return negated scikit-learn decision_function scores (higher = more anomalous)."""
        arr = np.asarray(X_scaled, dtype=np.float32)
        return np.asarray(-self.estimator.decision_function(arr), dtype=np.float64)

    def predict_result(self, X_scaled: np.ndarray, threshold_override: float | None = None) -> AnomalyResult:
        """Score a batch and classify anomalies against the calibrated benign threshold."""
        thresh = float(threshold_override if threshold_override is not None else self.threshold)
        scores = self.score(X_scaled)
        is_anom = scores > thresh
        denom = abs(thresh) if abs(thresh) > 1e-9 else 1e-9
        ratio = (scores - thresh) / denom + 1.0
        return AnomalyResult(
            scores=scores,
            threshold=thresh,
            is_anomaly=is_anom,
            severity_ratio=ratio,
        )


def _subsample_benign(X_benign: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if len(X_benign) <= max_samples:
        return np.asarray(X_benign, dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_benign), size=int(max_samples), replace=False)
    return np.asarray(X_benign[idx], dtype=np.float32)


def fit_classical_anomaly_baselines(
    X_benign_train: np.ndarray,
    X_benign_val: np.ndarray,
    cfg: AnomalyBaselinesConfig | None = None,
    percentile: float = 95.0,
    seed: int = 42,
) -> dict[str, ClassicalAnomalyBaseline]:
    """Fit Isolation Forest, One-Class SVM, and LOF strictly on BENIGN training flows."""
    conf = cfg or AnomalyBaselinesConfig()
    X_tr = np.asarray(X_benign_train, dtype=np.float32)
    X_va = np.asarray(X_benign_val, dtype=np.float32)
    if len(X_tr) < 5 or len(X_va) == 0:
        logger.warning(
            "Insufficient benign rows (%d train / %d val) for classical anomaly baselines.",
            len(X_tr),
            len(X_va),
        )
        return {}

    baselines: dict[str, ClassicalAnomalyBaseline] = {}

    # 1. Isolation Forest on full benign training split
    t0 = time.perf_counter()
    if_max = min(int(conf.isolation_forest_max_samples), len(X_tr))
    iso = IsolationForest(
        n_estimators=int(conf.isolation_forest_n_estimators),
        max_samples=if_max,
        contamination="auto",
        random_state=int(seed),
        n_jobs=-1,
    )
    iso.fit(X_tr)
    if_val_scores = np.asarray(-iso.decision_function(X_va), dtype=np.float64)
    if_thresh = float(np.percentile(if_val_scores, percentile))
    if_dur = float(time.perf_counter() - t0)
    baselines["Isolation Forest"] = ClassicalAnomalyBaseline(
        name="Isolation Forest",
        estimator=iso,
        threshold=if_thresh,
        train_samples_used=int(len(X_tr)),
        fit_duration_sec=if_dur,
    )

    # 2. One-Class SVM (RBF kernel) on bounded benign training subset to avoid O(N^2) blowup
    t0 = time.perf_counter()
    X_oc = _subsample_benign(X_tr, int(conf.ocsvm_max_train_samples), seed=seed)
    ocsvm = OneClassSVM(kernel="rbf", gamma=conf.ocsvm_gamma, nu=float(conf.ocsvm_nu))
    ocsvm.fit(X_oc)
    oc_val_scores = np.asarray(-ocsvm.decision_function(X_va), dtype=np.float64)
    oc_thresh = float(np.percentile(oc_val_scores, percentile))
    oc_dur = float(time.perf_counter() - t0)
    baselines["One-Class SVM"] = ClassicalAnomalyBaseline(
        name="One-Class SVM",
        estimator=ocsvm,
        threshold=oc_thresh,
        train_samples_used=int(len(X_oc)),
        fit_duration_sec=oc_dur,
    )

    # 3. Local Outlier Factor (novelty=True) on bounded benign training subset
    t0 = time.perf_counter()
    X_lof = _subsample_benign(X_tr, int(conf.lof_max_train_samples), seed=seed + 1)
    k_lof = max(2, min(int(conf.lof_n_neighbors), len(X_lof) - 1))
    lof = LocalOutlierFactor(n_neighbors=k_lof, novelty=True, n_jobs=-1)
    lof.fit(X_lof)
    lof_val_scores = np.asarray(-lof.decision_function(X_va), dtype=np.float64)
    lof_thresh = float(np.percentile(lof_val_scores, percentile))
    lof_dur = float(time.perf_counter() - t0)
    baselines["LOF"] = ClassicalAnomalyBaseline(
        name="LOF",
        estimator=lof,
        threshold=lof_thresh,
        train_samples_used=int(len(X_lof)),
        fit_duration_sec=lof_dur,
    )

    logger.info(
        "Trained classical anomaly baselines strictly on BENIGN rows: %s",
        {
            name: f"n={b.train_samples_used}, thresh={b.threshold:.4f}, time={b.fit_duration_sec:.2f}s"
            for name, b in baselines.items()
        },
    )
    return baselines
