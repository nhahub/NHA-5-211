"""Hybrid ensemble strategies, probability calibration, stacking, and comparative decision engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss

from src.config import EnsembleConfig


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) for multi-class probabilities."""
    probs = np.asarray(y_proba, dtype=np.float64)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
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
    probs = np.asarray(y_proba, dtype=np.float64)
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


class TemperatureScaler:
    """Post-hoc float64 Brent temperature-scaling calibrator fit on validation probabilities."""

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self.nll_before: float = 0.0
        self.nll_after: float = 0.0

    @staticmethod
    def _to_logits(probs: np.ndarray) -> np.ndarray:
        # Cast strictly to float64 before clipping/log: float32 underflows L-BFGS finite-diff steps.
        p64 = np.asarray(probs, dtype=np.float64)
        clipped = np.clip(p64, 1e-12, 1.0 - 1e-12)
        return np.log(clipped)

    @staticmethod
    def _apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
        t = max(float(temperature), 0.05)
        scaled = logits / t
        scaled -= np.max(scaled, axis=1, keepdims=True)
        exp_s = np.exp(scaled)
        return (exp_s / np.sum(exp_s, axis=1, keepdims=True)).astype(np.float64)

    def fit(self, probs: np.ndarray, y_val: np.ndarray) -> TemperatureScaler:
        if len(y_val) == 0:
            self.temperature = 1.0
            self.nll_before = 0.0
            self.nll_after = 0.0
            return self

        logits = self._to_logits(probs)
        y_arr = np.asarray(y_val, dtype=np.int64)
        row_idx = np.arange(len(y_arr))

        def _nll(t: float) -> float:
            p_cal = self._apply_temperature(logits, t)
            return float(-np.mean(np.log(np.clip(p_cal[row_idx, y_arr], 1e-15, 1.0))))

        self.nll_before = _nll(1.0)
        res = minimize_scalar(_nll, bounds=(0.05, 10.0), method="bounded")
        candidate_t = float(res.x) if res.success else 1.0
        candidate_nll = float(res.fun) if res.success else self.nll_before

        if candidate_nll <= self.nll_before:
            self.temperature = candidate_t
            self.nll_after = candidate_nll
        else:
            self.temperature = 1.0
            self.nll_after = self.nll_before
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        logits = self._to_logits(probs)
        return self._apply_temperature(logits, self.temperature)


class StackingMetaLearner:
    """Validation-only LogisticRegression stacking & logloss-weighted voting meta-learner."""

    def __init__(self) -> None:
        self.model_order: list[str] = []
        self.n_classes: int = 0
        self.meta_clf: LogisticRegression | None = None
        self.val_weights: dict[str, float] = {}
        self.val_logloss: dict[str, float] = {}
        self.val_macro_f1: dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        return self.meta_clf is not None and len(self.model_order) > 0

    @staticmethod
    def _to_log_features(proba: np.ndarray) -> np.ndarray:
        return np.log(np.clip(np.asarray(proba, dtype=np.float64), 1e-6, 1.0 - 1e-6))

    def fit(
        self,
        val_probas: dict[str, np.ndarray],
        y_val: np.ndarray,
        n_classes: int | None = None,
        calibrators: dict[str, TemperatureScaler] | None = None,
        seed: int = 42,
    ) -> StackingMetaLearner:
        available = {k: np.asarray(v, dtype=np.float64) for k, v in val_probas.items() if v is not None}
        if not available:
            raise ValueError("Cannot fit StackingMetaLearner without validation probability matrices.")

        self.model_order = sorted(available.keys())
        first_mat = available[self.model_order[0]]
        self.n_classes = int(n_classes or first_mat.shape[1])
        cal_map = calibrators or {}

        calibrated: dict[str, np.ndarray] = {}
        raw_weights: dict[str, float] = {}
        y_arr = np.asarray(y_val, dtype=np.int64)
        labels_idx = list(range(self.n_classes))

        for name in self.model_order:
            mat = cal_map[name].transform(available[name]) if name in cal_map else available[name]
            calibrated[name] = mat
            preds = np.argmax(mat, axis=1)
            ll = float(log_loss(y_arr, np.clip(mat, 1e-12, 1.0), labels=labels_idx))
            mf1 = float(f1_score(y_arr, preds, average="macro", zero_division=0))
            self.val_logloss[name] = ll
            self.val_macro_f1[name] = mf1
            raw_weights[name] = float(np.exp(-3.0 * ll))

        min_ll = min(self.val_logloss.values())
        exp_scores = {k: float(np.exp(-3.0 * (ll - min_ll))) for k, ll in self.val_logloss.items()}
        w_sum = sum(exp_scores.values()) or 1.0
        self.val_weights = {k: float(v / w_sum) for k, v in exp_scores.items()}

        if len(np.unique(y_arr)) < 2:
            self.meta_clf = None
            return self

        X_meta = np.hstack([self._to_log_features(calibrated[name]) for name in self.model_order])
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(X_meta, y_arr)
        self.meta_clf = clf
        return self

    def predict_proba(
        self,
        model_probas: dict[str, np.ndarray],
        calibrators: dict[str, TemperatureScaler] | None = None,
        already_calibrated: bool = True,
    ) -> np.ndarray:
        if self.meta_clf is None:
            raise RuntimeError("StackingMetaLearner has not been fitted.")
        cal_map = calibrators or {}
        parts = []
        for name in self.model_order:
            if name not in model_probas:
                raise KeyError(f"Missing required base model '{name}' for stacking meta-learner.")
            mat = np.asarray(model_probas[name], dtype=np.float64)
            if not already_calibrated and name in cal_map:
                mat = cal_map[name].transform(mat)
            parts.append(self._to_log_features(mat))

        X_meta = np.hstack(parts)
        raw = np.asarray(self.meta_clf.predict_proba(X_meta), dtype=np.float64)
        if raw.shape[1] == self.n_classes and np.array_equal(
            self.meta_clf.classes_, np.arange(self.n_classes)
        ):
            return raw
        full = np.zeros((raw.shape[0], self.n_classes), dtype=np.float64)
        full[:, np.asarray(self.meta_clf.classes_, dtype=np.int64)] = raw
        return full

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "model_order": list(self.model_order),
            "val_weights": {k: round(float(v), 6) for k, v in self.val_weights.items()},
            "val_logloss": {k: round(float(v), 6) for k, v in self.val_logloss.items()},
            "val_macro_f1": {k: round(float(v), 6) for k, v in self.val_macro_f1.items()},
            "meta_learner": "LogisticRegression(log_probs)",
        }


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
        stacker: StackingMetaLearner | None = None,
        val_weights: dict[str, float] | None = None,
    ) -> None:
        self.class_names = class_names
        self.cfg = cfg or EnsembleConfig()
        self.weights = weights or dict(self.cfg.supervised_weights)
        self.best_model_name = best_model_name
        self.benign_index = benign_index
        self.calibrators = calibrators or {}
        self.stacker = stacker
        self.val_weights = val_weights or (stacker.val_weights if stacker else dict(self.weights))

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

        if chosen == "hybrid_anomaly_override":
            base_strat = (
                self.cfg.base_supervised_strategy
                if self.cfg.base_supervised_strategy != "hybrid_anomaly_override"
                else "best_single"
            )
            probs, sub_name = self.combine_probabilities(available, strategy=base_strat)
            return probs, f"hybrid_anomaly_override({sub_name})"

        if chosen == "best_single":
            target = self.best_model_name if self.best_model_name in available else next(iter(available))
            return available[target], f"best_single:{target}"

        if chosen == "soft_voting":
            stacked = np.stack(list(available.values()), axis=0)
            return np.mean(stacked, axis=0), "soft_voting"

        if chosen == "stacking":
            if (
                self.stacker is not None
                and self.stacker.is_fitted
                and set(self.stacker.model_order).issubset(available.keys())
            ):
                return self.stacker.predict_proba(available, already_calibrated=True), "stacking"
            chosen = "val_weighted_voting"

        # confidence_weighted and val_weighted_voting combine models by per-model weights
        total_weight = np.zeros((next(iter(available.values())).shape[0], 1), dtype=np.float64)
        blended = np.zeros_like(next(iter(available.values())), dtype=np.float64)
        source_weights = self.val_weights if chosen == "val_weighted_voting" else self.weights

        for name, probs in available.items():
            base_w = float(source_weights.get(name, 1.0))
            if chosen == "confidence_weighted":
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
