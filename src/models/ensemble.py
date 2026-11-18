"""Hybrid ensemble strategies, probability calibration, and comparative decision engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from src.config import EnsembleConfig


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) for multi-class probabilities."""
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for b_lo, b_hi in zip(bin_boundaries[:-1], bin_boundaries[1:], strict=False):
        in_bin = (confidences > b_lo) & (confidences <= b_hi)
        count = int(np.sum(in_bin))
        if count > 0:
            avg_conf = float(np.mean(confidences[in_bin]))
            avg_acc = float(np.mean(accuracies[in_bin]))
            ece += (count / n) * abs(avg_acc - avg_conf)
    return float(ece)


def compute_multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute multi-class Brier score across all classes."""
    n_classes = y_proba.shape[1]
    one_hot = np.eye(n_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


class TemperatureScaler:
    """Post-hoc temperature scaling calibrator fit strictly on validation probabilities."""

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, probs: np.ndarray, y_val: np.ndarray) -> TemperatureScaler:
        clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
        logits = np.log(clipped)

        def _nll(t_arr: np.ndarray) -> float:
            t = max(float(t_arr[0]), 0.05)
            scaled = logits / t
            scaled -= np.max(scaled, axis=1, keepdims=True)
            exp_s = np.exp(scaled)
            p_cal = exp_s / np.sum(exp_s, axis=1, keepdims=True)
            return float(log_loss(y_val, p_cal, labels=list(range(probs.shape[1]))))

        res = minimize(_nll, x0=[1.0], bounds=[(0.1, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0]) if res.success else 1.0
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
        logits = np.log(clipped) / self.temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_s = np.exp(logits)
        return (exp_s / np.sum(exp_s, axis=1, keepdims=True)).astype(np.float64)


@dataclass
class HybridEnsembleDecision:
    """Structured hybrid supervised + unsupervised decision output."""

    predicted_indices: np.ndarray
    predicted_labels: list[str]
    confidences: np.ndarray
    probabilities: np.ndarray
    anomaly_scores: np.ndarray | None
    is_anomaly: np.ndarray | None
    anomaly_override_flags: np.ndarray
    strategy_used: str


class HybridEnsembleEngine:
    """Combine supervised classifiers (XGBoost, MLP, CNN) and Autoencoder anomaly signal."""

    def __init__(
        self,
        class_names: list[str],
        cfg: EnsembleConfig | None = None,
        weights: dict[str, float] | None = None,
        best_model_name: str = "XGBoost",
        benign_index: int = 0,
        calibrators: dict[str, TemperatureScaler] | None = None,
    ) -> None:
        self.class_names = class_names
        self.cfg = cfg or EnsembleConfig()
        self.weights = weights or dict(self.cfg.supervised_weights)
        self.best_model_name = best_model_name
        self.benign_index = benign_index
        self.calibrators = calibrators or {}

    def _calibrate(self, name: str, probs: np.ndarray) -> np.ndarray:
        if name in self.calibrators:
            return self.calibrators[name].transform(probs)
        return np.asarray(probs, dtype=np.float64)

    def combine_probabilities(
        self,
        model_probas: dict[str, np.ndarray],
        strategy: str | None = None,
    ) -> tuple[np.ndarray, str]:
        """Compute ensemble class probability matrix under the requested strategy."""
        chosen = strategy or self.cfg.strategy
        available = {k: self._calibrate(k, v) for k, v in model_probas.items() if v is not None}
        if not available:
            raise ValueError("At least one supervised model probability matrix is required.")

        if chosen == "best_single":
            target = self.best_model_name if self.best_model_name in available else next(iter(available))
            return available[target], f"best_single:{target}"

        if chosen == "soft_voting":
            stacked = np.stack(list(available.values()), axis=0)
            return np.mean(stacked, axis=0), "soft_voting"

        # confidence_weighted and hybrid_anomaly_override both compute weighted voting
        # scaled by per-model validation weight and per-sample margin/confidence
        total_weight = np.zeros((next(iter(available.values())).shape[0], 1), dtype=np.float64)
        blended = np.zeros_like(next(iter(available.values())), dtype=np.float64)

        for name, probs in available.items():
            base_w = float(self.weights.get(name, 1.0))
            if chosen in ("confidence_weighted", "hybrid_anomaly_override"):
                # Weight each model dynamically by its validation weight * sample confidence
                sample_conf = np.max(probs, axis=1, keepdims=True)
                eff_w = base_w * (0.5 + 0.5 * sample_conf)
            else:
                eff_w = np.full((len(probs), 1), base_w, dtype=np.float64)
            blended += probs * eff_w
            total_weight += eff_w

        blended /= np.maximum(total_weight, 1e-12)
        return blended, chosen

    def decide(
        self,
        model_probas: dict[str, np.ndarray],
        anomaly_scores: np.ndarray | None = None,
        anomaly_threshold: float | None = None,
        strategy: str | None = None,
    ) -> HybridEnsembleDecision:
        """Produce unified predictions preserving separation of supervised vs AE signals."""
        probs, effective_strategy = self.combine_probabilities(model_probas, strategy=strategy)
        pred_idx = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        is_anomaly: np.ndarray | None = None
        override_flags = np.zeros(len(pred_idx), dtype=bool)
        labels = [self.class_names[int(i)] for i in pred_idx]

        if anomaly_scores is not None and anomaly_threshold is not None:
            is_anomaly = anomaly_scores > float(anomaly_threshold)
            if (strategy or self.cfg.strategy) == "hybrid_anomaly_override":
                # Keep Autoencoder orthogonal: if supervised says BENIGN (or low confidence)
                # while Autoencoder flags a strong structural anomaly, surface a zero-day alert!
                for idx in range(len(pred_idx)):
                    is_benign_pred = int(pred_idx[idx]) == self.benign_index
                    low_sup_conf = float(confidences[idx]) < self.cfg.anomaly_override_min_confidence
                    if bool(is_anomaly[idx]) and (is_benign_pred or low_sup_conf):
                        override_flags[idx] = True
                        labels[idx] = "Zero-Day Anomaly (Novel)"

        return HybridEnsembleDecision(
            predicted_indices=pred_idx,
            predicted_labels=labels,
            confidences=confidences,
            probabilities=probs,
            anomaly_scores=anomaly_scores,
            is_anomaly=is_anomaly,
            anomaly_override_flags=override_flags,
            strategy_used=effective_strategy,
        )
