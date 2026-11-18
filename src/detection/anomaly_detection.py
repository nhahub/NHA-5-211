"""Independent Autoencoder zero-day anomaly detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.models.autoencoder import compute_reconstruction_error


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
