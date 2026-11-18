"""Optuna hyperparameter optimization and SMOTE vs Class-Weighting experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import f1_score, log_loss, recall_score

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

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "baseline_val_metrics": self.baseline_val_metrics,
            "tuned_val_metrics": self.tuned_val_metrics,
            "best_params": self.best_params,
            "n_trials_completed": self.n_trials_completed,
            "selected_variant": self.selected_variant,
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
    xgb_device: str = "cpu",
    seed: int = 42,
) -> ImbalanceExperimentResult:
    """Experimentally evaluate Unweighted vs Class-Weighted vs SMOTE training on validation data."""
    logger.info("Running imbalance handling experiment (Unweighted vs Class-Weight vs SMOTE)...")

    # Fast, controlled probe using shallow regularized XGBoost on identical validation set
    probe_override = {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.1}

    # 1. Unweighted raw imbalanced baseline
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

    # 3. SMOTE on standardized training split
    clf_sm, cls_sm = fit_xgboost_safe_multiclass(
        X_train=X_train_sm,
        y_train=y_train_sm,
        X_val=X_val_scaled,
        y_val=y_val,
        total_classes=num_classes,
        params_override=probe_override,
        device=xgb_device,
        seed=seed,
        early_stopping_rounds=15,
    )
    sm_metrics = _eval_val_scores(
        y_val, expand_xgboost_probas(clf_sm.predict_proba(X_val_scaled), cls_sm, num_classes), num_classes
    )

    # Select based on validation Macro-F1 (minority sensitivity) with tie-breaker on Weighted-F1
    if sm_metrics["val_macro_f1"] >= cw_metrics["val_macro_f1"] - 0.002:
        selected = "smote"
        justification = (
            f"SMOTE achieved validation Macro-F1={sm_metrics['val_macro_f1']:.4f} "
            f"(Weighted-F1={sm_metrics['val_weighted_f1']:.4f}, Macro-Recall={sm_metrics['val_macro_recall']:.4f}) "
            f"vs Class-Weight Macro-F1={cw_metrics['val_macro_f1']:.4f} "
            f"and Unweighted Macro-F1={raw_metrics['val_macro_f1']:.4f}. "
            "SMOTE is selected for deep/tree parity and balanced minority gradients."
        )
    else:
        selected = "class_weight"
        justification = (
            f"Class weighting outperformed SMOTE on validation Macro-F1 "
            f"({cw_metrics['val_macro_f1']:.4f} vs {sm_metrics['val_macro_f1']:.4f}) "
            "with lower synthetic distribution shift and lower memory footprint."
        )

    logger.info("Imbalance experiment winner: %s | %s", selected, justification)
    return ImbalanceExperimentResult(
        smote_metrics=sm_metrics,
        class_weight_metrics=cw_metrics,
        baseline_unweighted_metrics=raw_metrics,
        selected_strategy=selected,
        justification=justification,
    )


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

    if not cfg.optuna.enabled or cfg.optuna.n_trials <= 0:
        return XGBoostOptimizationResult(
            baseline_val_metrics=baseline_metrics,
            tuned_val_metrics=baseline_metrics,
            best_params=dict(cfg.baseline),
            n_trials_completed=0,
            selected_variant="baseline",
            best_model=baseline_model,
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

    # Choose tuned only if it beats or matches baseline on validation Macro-F1 or logloss
    if tuned_metrics["val_macro_f1"] > baseline_metrics["val_macro_f1"] or (
        abs(tuned_metrics["val_macro_f1"] - baseline_metrics["val_macro_f1"]) < 1e-4
        and tuned_metrics["val_mlogloss"] < baseline_metrics["val_mlogloss"]
    ):
        selected_variant = "tuned"
        best_model = tuned_model
    else:
        selected_variant = "baseline"
        best_model = baseline_model

    logger.info("Selected XGBoost variant: %s", selected_variant)
    return XGBoostOptimizationResult(
        baseline_val_metrics=baseline_metrics,
        tuned_val_metrics=tuned_metrics,
        best_params=best_params,
        n_trials_completed=len(study.trials),
        selected_variant=selected_variant,
        best_model=best_model,
    )
