"""Optuna hyperparameter optimization and SMOTE vs Class-Weighting experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import f1_score, log_loss, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from src.config import XGBoostConfig
from src.models.xgboost_model import (
    expand_xgboost_probas,
    fit_xgboost_safe_multiclass,
)
from src.utils.logging import get_logger

logger = get_logger("smart_ids.training.optimization")
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ImbalanceExperimentResult:
    """Comparison of SMOTE vs Class-Weighting evaluated on naturally imbalanced validation set."""

    smote_metrics: dict[str, float]
    class_weight_metrics: dict[str, float]
    baseline_unweighted_metrics: dict[str, float]
    selected_strategy: str
    justification: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class XGBoostOptimizationResult:
    """Comparison between report baseline XGBoost and Optuna-tuned XGBoost."""

    baseline_val_metrics: dict[str, float]
    tuned_val_metrics: dict[str, float]
    best_params: dict[str, Any]
    n_trials_completed: int
    selected_variant: str  # "tuned" or "baseline"
    best_model: Any
    cv_macro_f1_mean: float = 0.0
    cv_macro_f1_std: float = 0.0
    macro_f1_improvement: float = 0.0
    meaningful_improvement: bool = False
    selection_note: str = ""

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "baseline_val_metrics": self.baseline_val_metrics,
            "tuned_val_metrics": self.tuned_val_metrics,
            "best_params": self.best_params,
            "n_trials_completed": self.n_trials_completed,
            "selected_variant": self.selected_variant,
            "cv_macro_f1_mean": round(float(self.cv_macro_f1_mean), 6),
            "cv_macro_f1_std": round(float(self.cv_macro_f1_std), 6),
            "macro_f1_improvement": round(float(self.macro_f1_improvement), 6),
            "meaningful_improvement": bool(self.meaningful_improvement),
            "selection_note": self.selection_note,
        }


def _eval_val_scores(y_val: np.ndarray, probs: np.ndarray, num_classes: int) -> dict[str, float]:
    preds = np.argmax(probs, axis=1)
    return {
        "val_macro_f1": float(f1_score(y_val, preds, average="macro", zero_division=0)),
        "val_weighted_f1": float(f1_score(y_val, preds, average="weighted", zero_division=0)),
        "val_macro_recall": float(recall_score(y_val, preds, average="macro", zero_division=0)),
        "val_mlogloss": float(log_loss(y_val, probs, labels=list(range(num_classes)))),
    }


def compare_smote_vs_class_weight(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_train_sm: np.ndarray,
    y_train_sm: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    num_classes: int,
    smote_class_weights: dict[int, float] | None = None,
    xgb_device: str = "cpu",
    seed: int = 42,
) -> ImbalanceExperimentResult:
    """Experimentally evaluate Unweighted vs Class-Weight vs SMOTE and pick best Val Macro-F1."""
    logger.info("Running imbalance handling experiment (Unweighted vs Class-Weight vs SMOTE)...")

    # Fast, controlled probe using shallow regularized XGBoost on identical validation set
    probe_override = {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.1}

    # 1. Unweighted raw imbalanced baseline ("none")
    clf_raw, cls_raw = fit_xgboost_safe_multiclass(
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        total_classes=num_classes,
        params_override=probe_override,
        device=xgb_device,
        seed=seed,
        early_stopping_rounds=15,
    )
    raw_metrics = _eval_val_scores(
        y_val, expand_xgboost_probas(clf_raw.predict_proba(X_val_scaled), cls_raw, num_classes), num_classes
    )

    # 2. Inverse-frequency class weighting without synthetic oversampling
    sample_weights = np.array([class_weights[int(c)] for c in y_train], dtype=np.float32)
    clf_cw, cls_cw = fit_xgboost_safe_multiclass(
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        total_classes=num_classes,
        params_override=probe_override,
        device=xgb_device,
        seed=seed,
        sample_weight=sample_weights,
        early_stopping_rounds=15,
    )
    cw_metrics = _eval_val_scores(
        y_val, expand_xgboost_probas(clf_cw.predict_proba(X_val_scaled), cls_cw, num_classes), num_classes
    )

    # 3. Capped SMOTE on eligible classes + class-weight fallback for skipped rare classes
    sm_sample_weights = (
        np.array([smote_class_weights[int(c)] for c in y_train_sm], dtype=np.float32)
        if smote_class_weights
        else None
    )
    clf_sm, cls_sm = fit_xgboost_safe_multiclass(
        X_train=X_train_sm,
        y_train=y_train_sm,
        X_val=X_val_scaled,
        y_val=y_val,
        total_classes=num_classes,
        params_override=probe_override,
        device=xgb_device,
        seed=seed,
        sample_weight=sm_sample_weights,
        early_stopping_rounds=15,
    )
    sm_metrics = _eval_val_scores(
        y_val, expand_xgboost_probas(clf_sm.predict_proba(X_val_scaled), cls_sm, num_classes), num_classes
    )

    # Strictly data-driven selection: maximize validation Macro-F1 across all three candidates
    candidates = [
        ("none", raw_metrics),
        ("class_weight", cw_metrics),
        ("smote", sm_metrics),
    ]
    ordered = sorted(
        candidates,
        key=lambda item: (
            item[1]["val_macro_f1"],
            item[1]["val_weighted_f1"],
            -item[1]["val_mlogloss"],
        ),
        reverse=True,
    )
    selected, win_m = ordered[0]
    runner_name, runner_m = ordered[1]
    margin = win_m["val_macro_f1"] - runner_m["val_macro_f1"]
    display_labels = {
        "none": "Unweighted ('none')",
        "class_weight": "Class-Weight ('class_weight')",
        "smote": "Hybrid SMOTE ('smote')",
    }
    others = "; ".join(
        f"{display_labels[name]} Macro-F1={m['val_macro_f1']:.4f} "
        f"(Weighted-F1={m['val_weighted_f1']:.4f}, mlogloss={m['val_mlogloss']:.4f})"
        for name, m in ordered[1:]
    )
    justification = (
        f"Data-driven selection chose '{selected}' with highest validation Macro-F1={win_m['val_macro_f1']:.4f} "
        f"(Weighted-F1={win_m['val_weighted_f1']:.4f}, Macro-Recall={win_m['val_macro_recall']:.4f}, "
        f"mlogloss={win_m['val_mlogloss']:.4f}), beating runner-up '{runner_name}' by {margin:+.4f} "
        f"Macro-F1 ({others})."
    )

    logger.info("Imbalance experiment winner: %s | %s", selected, justification)
    return ImbalanceExperimentResult(
        smote_metrics=sm_metrics,
        class_weight_metrics=cw_metrics,
        baseline_unweighted_metrics=raw_metrics,
        selected_strategy=selected,
        justification=justification,
    )


def _estimate_baseline_cv_macro_f1(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    num_classes: int,
    cfg: XGBoostConfig,
    xgb_device: str,
    seed: int,
    n_splits: int = 5,
    use_class_weight: bool = False,
) -> tuple[float, float]:
    """Estimate baseline XGBoost fold mean and std Macro-F1 via stratified K-fold CV."""
    counts = np.bincount(y_raw)
    nz = counts[counts > 0]
    eff_splits = min(int(n_splits), int(nz.min())) if len(nz) else 0
    if eff_splits < 2:
        return 0.0, 0.0

    skf = StratifiedKFold(n_splits=eff_splits, shuffle=True, random_state=seed)
    fold_f1s: list[float] = []
    for tr_i, va_i in skf.split(X_raw, y_raw):
        X_tr_f, y_tr_f = X_raw[tr_i], y_raw[tr_i]
        X_va_f, y_va_f = X_raw[va_i], y_raw[va_i]
        sw_f = None
        if use_class_weight:
            uniq = np.unique(y_tr_f)
            cw = compute_class_weight("balanced", classes=uniq, y=y_tr_f)
            cw_map = {int(c): float(w) for c, w in zip(uniq, cw, strict=False)}
            sw_f = np.array([cw_map[int(c)] for c in y_tr_f], dtype=np.float32)

        clf_f, cls_f = fit_xgboost_safe_multiclass(
            X_train=X_tr_f,
            y_train=y_tr_f,
            X_val=X_va_f,
            y_val=y_va_f,
            total_classes=num_classes,
            cfg=cfg,
            device=xgb_device,
            seed=seed,
            sample_weight=sw_f,
            early_stopping_rounds=20,
        )
        probs_f = expand_xgboost_probas(clf_f.predict_proba(X_va_f), cls_f, num_classes)
        fold_f1s.append(float(f1_score(y_va_f, np.argmax(probs_f, axis=1), average="macro", zero_division=0)))
    return float(np.mean(fold_f1s)), float(np.std(fold_f1s, ddof=1))


def optimize_xgboost_with_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    cfg: XGBoostConfig,
    xgb_device: str = "cpu",
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
    cv_macro_f1_std: float | None = None,
    cv_raw_X: np.ndarray | None = None,
    cv_raw_y: np.ndarray | None = None,
    cv_n_splits: int = 5,
    use_cv_class_weight: bool = False,
) -> XGBoostOptimizationResult:
    """Compare baseline XGBoost (report spec) against Optuna-tuned XGBoost on validation set."""
    logger.info("Training report baseline XGBoost (300 estimators, depth=6, lr=0.1)...")
    baseline_model, base_cls = fit_xgboost_safe_multiclass(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        total_classes=num_classes,
        cfg=cfg,
        device=xgb_device,
        seed=seed,
        sample_weight=sample_weight,
    )
    baseline_probs = expand_xgboost_probas(baseline_model.predict_proba(X_val), base_cls, num_classes)
    baseline_metrics = _eval_val_scores(y_val, baseline_probs, num_classes)
    logger.info("Baseline XGBoost validation metrics: %s", baseline_metrics)

    if cv_macro_f1_std is not None:
        cv_mean = float(baseline_metrics["val_macro_f1"])
        cv_std = float(cv_macro_f1_std)
    else:
        src_X = cv_raw_X if cv_raw_X is not None else X_train
        src_y = cv_raw_y if cv_raw_y is not None else y_train
        cv_mean, cv_std = _estimate_baseline_cv_macro_f1(
            X_raw=src_X,
            y_raw=src_y,
            num_classes=num_classes,
            cfg=cfg,
            xgb_device=xgb_device,
            seed=seed,
            n_splits=cv_n_splits,
            use_class_weight=use_cv_class_weight,
        )

    if not cfg.optuna.enabled or cfg.optuna.n_trials <= 0:
        return XGBoostOptimizationResult(
            baseline_val_metrics=baseline_metrics,
            tuned_val_metrics=baseline_metrics,
            best_params=dict(cfg.baseline),
            n_trials_completed=0,
            selected_variant="baseline",
            best_model=baseline_model,
            cv_macro_f1_mean=cv_mean,
            cv_macro_f1_std=cv_std,
            macro_f1_improvement=0.0,
            meaningful_improvement=False,
            selection_note="Optuna optimization disabled (n_trials=0); retaining baseline.",
        )

    # Subsample training rows during Optuna search if very large, keeping stratification
    rng = np.random.default_rng(seed)
    cap = cfg.optuna.sample_cap_for_tuning
    if len(X_train) > cap:
        idx = rng.choice(len(X_train), size=cap, replace=False)
        X_tune = X_train[idx]
        y_tune = y_train[idx]
        sw_tune = sample_weight[idx] if sample_weight is not None else None
    else:
        X_tune, y_tune, sw_tune = X_train, y_train, sample_weight

    def _objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.22, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        }
        clf, trial_cls = fit_xgboost_safe_multiclass(
            X_train=X_tune,
            y_train=y_tune,
            X_val=X_val,
            y_val=y_val,
            total_classes=num_classes,
            cfg=cfg,
            params_override=params,
            device=xgb_device,
            seed=seed,
            sample_weight=sw_tune,
            early_stopping_rounds=20,
        )
        val_probs = expand_xgboost_probas(clf.predict_proba(X_val), trial_cls, num_classes)
        return float(log_loss(y_val, val_probs, labels=list(range(num_classes))))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        _objective,
        n_trials=cfg.optuna.n_trials,
        timeout=cfg.optuna.timeout_seconds,
        show_progress_bar=False,
    )

    best_params = dict(study.best_params)
    logger.info("Optuna best XGBoost params (%d trials): %s", len(study.trials), best_params)

    # Retrain best Optuna config on full training partition
    tuned_model, tuned_cls = fit_xgboost_safe_multiclass(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        total_classes=num_classes,
        cfg=cfg,
        params_override=best_params,
        device=xgb_device,
        seed=seed,
        sample_weight=sample_weight,
        early_stopping_rounds=20,
    )
    tuned_probs = expand_xgboost_probas(tuned_model.predict_proba(X_val), tuned_cls, num_classes)
    tuned_metrics = _eval_val_scores(y_val, tuned_probs, num_classes)
    logger.info("Optuna-tuned XGBoost validation metrics: %s", tuned_metrics)

    # Select tuned model ONLY if it improves validation Macro-F1 by more than the baseline CV std
    improvement = float(tuned_metrics["val_macro_f1"] - baseline_metrics["val_macro_f1"])
    threshold = max(float(cv_std), 1e-6)
    meaningful_improvement = improvement > threshold
    if meaningful_improvement:
        selected_variant = "tuned"
        best_model = tuned_model
        selection_note = (
            f"Meaningful improvement: tuned val Macro-F1 ({tuned_metrics['val_macro_f1']:.4f}) beat "
            f"baseline ({baseline_metrics['val_macro_f1']:.4f}) by {improvement:+.4f} (> CV std={cv_std:.4f})."
        )
    else:
        selected_variant = "baseline"
        best_model = baseline_model
        selection_note = (
            f"No meaningful improvement: Optuna tuned val Macro-F1 ({tuned_metrics['val_macro_f1']:.4f}) "
            f"changed by {improvement:+.4f} vs baseline ({baseline_metrics['val_macro_f1']:.4f}), within "
            f"baseline CV std={cv_std:.4f}; retaining baseline."
        )

    logger.info("Selected XGBoost variant: %s | %s", selected_variant, selection_note)
    return XGBoostOptimizationResult(
        baseline_val_metrics=baseline_metrics,
        tuned_val_metrics=tuned_metrics,
        best_params=best_params,
        n_trials_completed=len(study.trials),
        selected_variant=selected_variant,
        best_model=best_model,
        cv_macro_f1_mean=cv_mean,
        cv_macro_f1_std=cv_std,
        macro_f1_improvement=improvement,
        meaningful_improvement=meaningful_improvement,
        selection_note=selection_note,
    )
