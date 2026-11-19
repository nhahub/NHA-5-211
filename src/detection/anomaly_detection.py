"""Independent Autoencoder and classical benign-only anomaly detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score
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


@dataclass
class BenignScoreNormalizer:
    """Robust score normalizer fit strictly on benign validation anomaly scores."""

    center: float
    scale: float
    percentile_ref: float = 95.0

    @classmethod
    def fit(cls, benign_val_scores: np.ndarray, percentile: float = 95.0) -> BenignScoreNormalizer:
        arr = np.asarray(benign_val_scores, dtype=np.float64)
        if len(arr) == 0:
            return cls(center=0.0, scale=1.0, percentile_ref=float(percentile))
        center = float(np.median(arr))
        p_hi = float(np.percentile(arr, percentile))
        std = float(np.std(arr))
        scale = max(p_hi - center, std, 1e-6)
        return cls(center=center, scale=scale, percentile_ref=float(percentile))

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return (np.asarray(scores, dtype=np.float64) - self.center) / self.scale


@dataclass
class AnomalyFusionDetector:
    """Convex score-level fusion over benign-normalized Autoencoder, OCSVM, and Isolation Forest."""

    weights: dict[str, float]
    normalizers: dict[str, BenignScoreNormalizer]
    threshold: float
    percentile: float = 95.0
    ae_component_name: str = "Autoencoder"
    best_single_validation_detector: str = "Autoencoder"
    best_overall_validation_detector: str = "Score-Level Fusion (AE+OCSVM+IF)"
    validation_selection_summary: list[dict[str, Any]] = field(default_factory=list)
    name: str = "Score-Level Fusion (AE+OCSVM+IF)"

    def score_from_components(self, component_scores: dict[str, np.ndarray]) -> np.ndarray:
        first = next(iter(component_scores.values()))
        fused = np.zeros(len(first), dtype=np.float64)
        for comp_name, w in self.weights.items():
            if w <= 0.0 or comp_name not in component_scores:
                continue
            norm = self.normalizers[comp_name]
            fused += float(w) * norm.transform(component_scores[comp_name])
        return fused

    def predict_result_from_components(
        self,
        component_scores: dict[str, np.ndarray],
        threshold_override: float | None = None,
    ) -> AnomalyResult:
        thresh = float(threshold_override if threshold_override is not None else self.threshold)
        scores = self.score_from_components(component_scores)
        is_anom = scores > thresh
        denom = abs(thresh) if abs(thresh) > 1e-9 else 1e-9
        ratio = (scores - thresh) / denom + 1.0
        return AnomalyResult(
            scores=scores,
            threshold=thresh,
            is_anomaly=is_anom,
            severity_ratio=ratio,
        )


def _leave_attack_out_val_score(
    scores_val: np.ndarray,
    y_val: np.ndarray,
    benign_index: int,
    percentile: float = 95.0,
) -> tuple[float, float, float]:
    """Score anomaly vector strictly on benign val percentile + non-test leave-attack-out folds.

    Returns ``(selection_objective, mean_loo_f1, mean_loo_roc_auc)``.
    """
    ben_mask = y_val == benign_index
    ben_scores = scores_val[ben_mask]
    if len(ben_scores) == 0:
        return 0.0, 0.0, 0.0

    thresh = float(np.percentile(ben_scores, percentile))
    y_pred = (scores_val > thresh).astype(np.int64)
    atk_classes = [int(c) for c in np.unique(y_val) if int(c) != benign_index]
    if not atk_classes:
        return 0.0, 0.0, 0.0

    loo_f1s: list[float] = []
    loo_aucs: list[float] = []
    for cls_id in atk_classes:
        mask = ben_mask | (y_val == cls_id)
        y_bin = (y_val[mask] == cls_id).astype(np.int64)
        p_bin = y_pred[mask]
        s_bin = scores_val[mask]
        loo_f1s.append(float(f1_score(y_bin, p_bin, zero_division=0)))
        if len(np.unique(y_bin)) > 1:
            loo_aucs.append(float(roc_auc_score(y_bin, s_bin)))

    mean_f1 = float(np.mean(loo_f1s)) if loo_f1s else 0.0
    mean_auc = float(np.mean(loo_aucs)) if loo_aucs else 0.0
    # Aggregate full-val F1 alongside macro leave-attack-out F1 & ROC-AUC
    y_bin_all = (~ben_mask).astype(np.int64)
    full_val_f1 = float(f1_score(y_bin_all, y_pred, zero_division=0))
    objective = 0.5 * mean_f1 + 0.3 * full_val_f1 + 0.2 * mean_auc
    return objective, mean_f1, mean_auc


def fit_and_select_anomaly_fusion(
    val_scores_by_detector: dict[str, np.ndarray],
    y_val: np.ndarray,
    benign_index: int,
    percentile: float = 95.0,
    grid_step: float = 0.2,
    fusion_ae_candidates: tuple[str, ...] = ("Autoencoder", "Autoencoder (Deep Reg)", "VAE"),
) -> AnomalyFusionDetector:
    """Tune score-level fusion weights strictly on benign validation + non-test validation attacks."""
    ben_mask = y_val == benign_index
    normalizers = {
        name: BenignScoreNormalizer.fit(scores[ben_mask], percentile=percentile)
        for name, scores in val_scores_by_detector.items()
    }

    val_summary: list[dict[str, Any]] = []
    single_objs: dict[str, float] = {}
    for det_name, scores in val_scores_by_detector.items():
        obj, loo_f1, loo_auc = _leave_attack_out_val_score(
            scores, y_val, benign_index=benign_index, percentile=percentile
        )
        single_objs[det_name] = obj
        val_summary.append(
            {
                "Detector": det_name,
                "Type": "single",
                "Weights": det_name,
                "Val_LOO_F1": round(loo_f1, 4),
                "Val_LOO_ROC_AUC": round(loo_auc, 4),
                "Val_Selection_Objective": round(obj, 4),
            }
        )

    avail_ae = [name for name in fusion_ae_candidates if name in val_scores_by_detector]
    best_ae_name = (
        max(avail_ae, key=lambda n: single_objs[n]) if avail_ae else next(iter(val_scores_by_detector))
    )
    fusion_keys = [
        k for k in (best_ae_name, "One-Class SVM", "Isolation Forest") if k in val_scores_by_detector
    ]

    norm_val = {k: normalizers[k].transform(val_scores_by_detector[k]) for k in fusion_keys}
    best_weights = {fusion_keys[0]: 1.0}
    best_fused_obj = -1.0
    best_fused_f1 = 0.0
    best_fused_auc = 0.0

    steps = max(2, round(1.0 / max(grid_step, 0.05)))
    if len(fusion_keys) == 1:
        best_weights = {fusion_keys[0]: 1.0}
        fused = norm_val[fusion_keys[0]]
        best_fused_obj, best_fused_f1, best_fused_auc = _leave_attack_out_val_score(
            fused, y_val, benign_index=benign_index, percentile=percentile
        )
    elif len(fusion_keys) == 2:
        k0, k1 = fusion_keys
        for i in range(steps + 1):
            w0 = round(i / steps, 4)
            w1 = round(1.0 - w0, 4)
            fused = w0 * norm_val[k0] + w1 * norm_val[k1]
            obj, lf1, lauc = _leave_attack_out_val_score(
                fused, y_val, benign_index=benign_index, percentile=percentile
            )
            if obj > best_fused_obj:
                best_fused_obj, best_fused_f1, best_fused_auc = obj, lf1, lauc
                best_weights = {k0: w0, k1: w1}
    else:
        k0, k1, k2 = fusion_keys[:3]
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                k = steps - i - j
                w0 = round(i / steps, 4)
                w1 = round(j / steps, 4)
                w2 = round(k / steps, 4)
                fused = w0 * norm_val[k0] + w1 * norm_val[k1] + w2 * norm_val[k2]
                obj, lf1, lauc = _leave_attack_out_val_score(
                    fused, y_val, benign_index=benign_index, percentile=percentile
                )
                if obj > best_fused_obj:
                    best_fused_obj, best_fused_f1, best_fused_auc = obj, lf1, lauc
                    best_weights = {k0: w0, k1: w1, k2: w2}

    fused_val_scores = np.zeros(len(y_val), dtype=np.float64)
    for k, w in best_weights.items():
        fused_val_scores += w * norm_val[k]
    fused_thresh = float(np.percentile(fused_val_scores[ben_mask], percentile)) if np.any(ben_mask) else 1.0

    weight_desc = ", ".join(f"{k}={w:.2f}" for k, w in best_weights.items() if w > 0)
    val_summary.append(
        {
            "Detector": "Score-Level Fusion (AE+OCSVM+IF)",
            "Type": "fusion",
            "Weights": weight_desc,
            "Val_LOO_F1": round(best_fused_f1, 4),
            "Val_LOO_ROC_AUC": round(best_fused_auc, 4),
            "Val_Selection_Objective": round(best_fused_obj, 4),
        }
    )

    best_single = max(single_objs, key=lambda n: single_objs[n], default="Autoencoder")
    best_overall = (
        "Score-Level Fusion (AE+OCSVM+IF)"
        if best_fused_obj >= single_objs.get(best_single, -1.0)
        else best_single
    )
    logger.info(
        "Anomaly validation selection (leave-attack-out, non-test): best_single='%s' (obj=%.4f), "
        "fusion weights={%s} (obj=%.4f) -> winner='%s'",
        best_single,
        single_objs.get(best_single, 0.0),
        weight_desc,
        best_fused_obj,
        best_overall,
    )
    return AnomalyFusionDetector(
        weights=best_weights,
        normalizers={k: normalizers[k] for k in fusion_keys},
        threshold=fused_thresh,
        percentile=float(percentile),
        ae_component_name=best_ae_name,
        best_single_validation_detector=best_single,
        best_overall_validation_detector=best_overall,
        validation_selection_summary=val_summary,
    )


_DEGENERATE_FUSION_NOTE = (
    "Validation selected a single-detector fusion; no gain from fusing detectors on this data."
)


def describe_score_level_fusion(
    fusion_weights: dict[str, float] | None,
    default_name: str = "Score-Level Fusion (AE+OCSVM+IF)",
) -> tuple[str, bool, str | None]:
    """Return ``(display_label, is_degenerate, note)`` identifying single-detector fusion weights."""
    active = {k: float(w) for k, w in (fusion_weights or {}).items() if float(w) > 1e-6}
    if len(active) == 1:
        sole_name, sole_weight = next(iter(active.items()))
        if abs(sole_weight - 1.0) <= 1e-6:
            return f"Score-Level Fusion (degenerate: equals {sole_name})", True, _DEGENERATE_FUSION_NOTE
    return default_name, False, None
