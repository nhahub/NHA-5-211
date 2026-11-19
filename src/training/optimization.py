"""Optuna hyperparameter optimization and SMOTE vs Class-Weighting experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import f1_score, log_loss, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from src.config import ModelConfig, XGBoostConfig
from src.data.preprocessing import apply_smote_safely, compute_hybrid_smote_class_weights
from src.models.xgboost_model import (
    expand_xgboost_probas,
    fit_xgboost_safe_multiclass,
)
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger("smart_ids.training.optimization")
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ModelImbalanceOutcome:
    """Validation comparison across {none, class_weight, smote} for a single model family."""

    model_name: str
    metrics_by_strategy: dict[str, dict[str, float]]
    selected_strategy: str
    justification: str
    epochs_cap_applied: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImbalanceExperimentResult:
    """Comparison of SMOTE vs Class-Weighting evaluated on naturally imbalanced validation set."""

    smote_metrics: dict[str, float]
    class_weight_metrics: dict[str, float]
    baseline_unweighted_metrics: dict[str, float]
    selected_strategy: str
    justification: str
    per_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    selected_strategies_by_model: dict[str, str] = field(default_factory=dict)
    neural_epochs_cap: int | None = None
    selection_mode: str = "global"

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
    cv_method_note: str = ""

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
            "cv_method_note": self.cv_method_note,
        }


def _eval_val_scores(y_val: np.ndarray, probs: np.ndarray, num_classes: int) -> dict[str, float]:
    preds = np.argmax(probs, axis=1)
    return {
        "val_macro_f1": float(f1_score(y_val, preds, average="macro", zero_division=0)),
        "val_weighted_f1": float(f1_score(y_val, preds, average="weighted", zero_division=0)),
        "val_macro_recall": float(recall_score(y_val, preds, average="macro", zero_division=0)),
        "val_mlogloss": float(log_loss(y_val, probs, labels=list(range(num_classes)))),
    }


SIMPLICITY_ORDER: tuple[str, ...] = ("none", "class_weight", "smote")


def select_imbalance_strategy(
    metrics_by_strategy: dict[str, dict[str, float]],
    tie_tolerance: float = 0.002,
) -> tuple[str, str]:
    """Select the best imbalance strategy on validation Macro-F1 with simplicity tie-breaking.

    If multiple candidates fall within ``tie_tolerance`` of the highest validation Macro-F1,
    prefer the simplest strategy strictly in the order ``none > class_weight > smote`` and never
    phrase the margin as "beating" another candidate.
    """
    if not metrics_by_strategy:
        raise ValueError("metrics_by_strategy must not be empty.")

    simplicity_rank = {name: rank for rank, name in enumerate(SIMPLICITY_ORDER)}
    ordered_by_score = sorted(
        metrics_by_strategy.items(),
        key=lambda item: (
            float(item[1].get("val_macro_f1", 0.0)),
            float(item[1].get("val_weighted_f1", 0.0)),
            -float(item[1].get("val_mlogloss", 0.0)),
            -simplicity_rank.get(item[0], 99),
        ),
        reverse=True,
    )
    top_name, top_m = ordered_by_score[0]
    best_f1 = float(top_m.get("val_macro_f1", 0.0))
    tol = max(0.0, float(tie_tolerance))

    contenders = [
        (name, m)
        for name, m in ordered_by_score
        if (best_f1 - float(m.get("val_macro_f1", 0.0))) <= tol + 1e-12
    ]

    display_labels = {
        "none": "Unweighted ('none')",
        "class_weight": "Class-Weight ('class_weight')",
        "smote": "Hybrid SMOTE ('smote')",
    }

    if len(contenders) > 1:
        contenders_by_simplicity = sorted(
            contenders,
            key=lambda item: (
                simplicity_rank.get(item[0], 99),
                -float(item[1].get("val_macro_f1", 0.0)),
            ),
        )
        selected, win_m = contenders_by_simplicity[0]
        tied_str = ", ".join(
            f"{display_labels.get(name, name)} Macro-F1={float(m.get('val_macro_f1', 0.0)):.4f}"
            for name, m in contenders_by_simplicity
        )
        others = "; ".join(
            f"{display_labels.get(name, name)} Macro-F1={float(m.get('val_macro_f1', 0.0)):.4f} "
            f"(Weighted-F1={float(m.get('val_weighted_f1', 0.0)):.4f}, "
            f"mlogloss={float(m.get('val_mlogloss', 0.0)):.4f})"
            for name, m in ordered_by_score
            if name != selected
        )
        gap_to_top = best_f1 - float(win_m.get("val_macro_f1", 0.0))
        justification = (
            f"Tie within tolerance (tol={tol:.4f}): {tied_str} fall within {tol:.4f} of peak "
            f"validation Macro-F1={best_f1:.4f} (gap={gap_to_top:.4f} vs '{top_name}'); "
            f"tie within tolerance, simplest strategy selected -> '{selected}' "
            f"(Weighted-F1={float(win_m.get('val_weighted_f1', 0.0)):.4f}, "
            f"Macro-Recall={float(win_m.get('val_macro_recall', 0.0)):.4f}, "
            f"mlogloss={float(win_m.get('val_mlogloss', 0.0)):.4f}" + (f"; {others})." if others else ").")
        )
        return selected, justification

    selected, win_m = top_name, top_m
    if len(ordered_by_score) > 1:
        runner_name, runner_m = ordered_by_score[1]
        margin = best_f1 - float(runner_m.get("val_macro_f1", 0.0))
        others = "; ".join(
            f"{display_labels.get(name, name)} Macro-F1={float(m.get('val_macro_f1', 0.0)):.4f} "
            f"(Weighted-F1={float(m.get('val_weighted_f1', 0.0)):.4f}, "
            f"mlogloss={float(m.get('val_mlogloss', 0.0)):.4f})"
            for name, m in ordered_by_score[1:]
        )
        justification = (
            f"Data-driven selection chose '{selected}' with highest validation Macro-F1={best_f1:.4f} "
            f"(Weighted-F1={float(win_m.get('val_weighted_f1', 0.0)):.4f}, "
            f"Macro-Recall={float(win_m.get('val_macro_recall', 0.0)):.4f}, "
            f"mlogloss={float(win_m.get('val_mlogloss', 0.0)):.4f}), beating runner-up '{runner_name}' "
            f"by {margin:+.4f} Macro-F1 (> tolerance={tol:.4f}; {others})."
        )
    else:
        justification = f"Data-driven selection chose '{selected}' (validation Macro-F1={best_f1:.4f})."
    return selected, justification


def prepare_strategy_training_data(
    strategy: str,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_train_sm: np.ndarray,
    y_train_sm: np.ndarray,
    class_weights: dict[int, float] | None,
    smote_class_weights: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, float] | None, np.ndarray | None]:
    """Return (X_fit, y_fit, keras_class_weight, xgb_sample_weight) for a chosen strategy."""
    norm = str(strategy).strip().lower()
    if norm == "smote":
        cw = smote_class_weights
        sw = np.array([cw[int(c)] for c in y_train_sm], dtype=np.float32) if cw else None
        return X_train_sm, y_train_sm, cw, sw
    if norm == "class_weight":
        cw = class_weights
        sw = np.array([cw[int(c)] for c in y_train], dtype=np.float32) if cw else None
        return X_train_scaled, y_train, cw, sw
    return X_train_scaled, y_train, None, None


def _evaluate_xgb_imbalance_strategies(
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
) -> dict[str, dict[str, float]]:
    """Evaluate {none, class_weight, smote} on a fast controlled XGBoost probe."""
    probe_override = {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.1}
    results: dict[str, dict[str, float]] = {}
    for strategy in SIMPLICITY_ORDER:
        X_fit, y_fit, _, sw = prepare_strategy_training_data(
            strategy, X_train_scaled, y_train, X_train_sm, y_train_sm, class_weights, smote_class_weights
        )
        clf, present_cls = fit_xgboost_safe_multiclass(
            X_train=X_fit,
            y_train=y_fit,
            X_val=X_val_scaled,
            y_val=y_val,
            total_classes=num_classes,
            params_override=probe_override,
            device=xgb_device,
            seed=seed,
            sample_weight=sw,
            early_stopping_rounds=15,
        )
        probs = expand_xgboost_probas(clf.predict_proba(X_val_scaled), present_cls, num_classes)
        results[strategy] = _eval_val_scores(y_val, probs, num_classes)
    return results


def _evaluate_neural_imbalance_strategies(
    model_name: str,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_train_sm: np.ndarray,
    y_train_sm: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    num_classes: int,
    model_cfg: ModelConfig,
    smote_class_weights: dict[int, float] | None = None,
    neural_epochs_cap: int = 10,
    seed: int = 42,
) -> tuple[dict[str, dict[str, float]], int]:
    """Evaluate {none, class_weight, smote} directly on MLP or 1D-CNN with capped epochs."""
    import keras

    from src.models.cnn import build_cnn1d
    from src.models.mlp import build_mlp

    num_features = int(X_train_scaled.shape[1])
    is_cnn = model_name == "CNN"
    base_epochs = model_cfg.cnn.epochs if is_cnn else model_cfg.mlp.epochs
    epochs_cap = max(1, min(int(base_epochs), int(neural_epochs_cap)))
    batch_size = model_cfg.cnn.batch_size if is_cnn else model_cfg.mlp.batch_size
    patience = model_cfg.cnn.early_stopping_patience if is_cnn else model_cfg.mlp.early_stopping_patience

    X_val_in = X_val_scaled.reshape(-1, num_features, 1) if is_cnn else X_val_scaled
    results: dict[str, dict[str, float]] = {}

    for strategy in SIMPLICITY_ORDER:
        keras.backend.clear_session()
        set_global_seed(seed)
        X_fit, y_fit, keras_cw, _ = prepare_strategy_training_data(
            strategy, X_train_scaled, y_train, X_train_sm, y_train_sm, class_weights, smote_class_weights
        )
        X_fit_in = X_fit.reshape(-1, num_features, 1) if is_cnn else X_fit
        model = (
            build_cnn1d(num_features, num_classes, cfg=model_cfg.cnn)
            if is_cnn
            else build_mlp(num_features, num_classes, cfg=model_cfg.mlp)
        )
        es_cb = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=max(1, min(3, int(patience))),
            restore_best_weights=True,
            verbose=0,
        )
        model.fit(
            X_fit_in,
            y_fit,
            validation_data=(X_val_in, y_val),
            epochs=epochs_cap,
            batch_size=batch_size,
            callbacks=[es_cb],
            class_weight=keras_cw,
            verbose=0,
        )
        probs = model.predict(X_val_in, batch_size=max(256, int(batch_size)), verbose=0)
        results[strategy] = _eval_val_scores(y_val, probs, num_classes)
        del model

    keras.backend.clear_session()
    return results, epochs_cap


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
    tie_tolerance: float = 0.002,
) -> ImbalanceExperimentResult:
    """Experimentally evaluate Unweighted vs Class-Weight vs SMOTE and pick best Val Macro-F1."""
    logger.info("Running global imbalance handling experiment (Unweighted vs Class-Weight vs SMOTE)...")
    xgb_metrics = _evaluate_xgb_imbalance_strategies(
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_train_sm=X_train_sm,
        y_train_sm=y_train_sm,
        X_val_scaled=X_val_scaled,
        y_val=y_val,
        class_weights=class_weights,
        num_classes=num_classes,
        smote_class_weights=smote_class_weights,
        xgb_device=xgb_device,
        seed=seed,
    )
    selected, justification = select_imbalance_strategy(
        metrics_by_strategy=xgb_metrics,
        tie_tolerance=tie_tolerance,
    )

    logger.info("Imbalance experiment winner: %s | %s", selected, justification)
    return ImbalanceExperimentResult(
        smote_metrics=xgb_metrics["smote"],
        class_weight_metrics=xgb_metrics["class_weight"],
        baseline_unweighted_metrics=xgb_metrics["none"],
        selected_strategy=selected,
        justification=justification,
        selected_strategies_by_model={"MLP": selected, "CNN": selected, "XGBoost": selected},
        selection_mode="global",
    )


def compare_imbalance_per_model(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_train_sm: np.ndarray,
    y_train_sm: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    num_classes: int,
    model_cfg: ModelConfig | None = None,
    smote_class_weights: dict[int, float] | None = None,
    xgb_device: str = "cpu",
    seed: int = 42,
    tie_tolerance: float = 0.002,
    neural_epochs_cap: int = 10,
) -> ImbalanceExperimentResult:
    """Select the optimal imbalance strategy independently for MLP, 1D-CNN, and XGBoost."""
    cfg = model_cfg or ModelConfig()
    logger.info(
        "Running per-model imbalance strategy selection (MLP, 1D-CNN @ cap=%d epochs, XGBoost)...",
        int(neural_epochs_cap),
    )

    per_model_outcomes: dict[str, ModelImbalanceOutcome] = {}

    for neural_name in ("MLP", "CNN"):
        metrics_map, applied_cap = _evaluate_neural_imbalance_strategies(
            model_name=neural_name,
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            X_train_sm=X_train_sm,
            y_train_sm=y_train_sm,
            X_val_scaled=X_val_scaled,
            y_val=y_val,
            class_weights=class_weights,
            num_classes=num_classes,
            model_cfg=cfg,
            smote_class_weights=smote_class_weights,
            neural_epochs_cap=neural_epochs_cap,
            seed=seed,
        )
        sel, just = select_imbalance_strategy(metrics_map, tie_tolerance=tie_tolerance)
        per_model_outcomes[neural_name] = ModelImbalanceOutcome(
            model_name=neural_name,
            metrics_by_strategy=metrics_map,
            selected_strategy=sel,
            justification=just,
            epochs_cap_applied=applied_cap,
        )
        logger.info("Per-model imbalance [%s]: selected='%s' | %s", neural_name, sel, just)

    xgb_metrics = _evaluate_xgb_imbalance_strategies(
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_train_sm=X_train_sm,
        y_train_sm=y_train_sm,
        X_val_scaled=X_val_scaled,
        y_val=y_val,
        class_weights=class_weights,
        num_classes=num_classes,
        smote_class_weights=smote_class_weights,
        xgb_device=xgb_device,
        seed=seed,
    )
    xgb_sel, xgb_just = select_imbalance_strategy(xgb_metrics, tie_tolerance=tie_tolerance)
    per_model_outcomes["XGBoost"] = ModelImbalanceOutcome(
        model_name="XGBoost",
        metrics_by_strategy=xgb_metrics,
        selected_strategy=xgb_sel,
        justification=xgb_just,
        epochs_cap_applied=None,
    )
    logger.info("Per-model imbalance [XGBoost]: selected='%s' | %s", xgb_sel, xgb_just)

    selected_by_model = {k: v.selected_strategy for k, v in per_model_outcomes.items()}
    summary_just = (
        "Per-model imbalance selection (independent validation Macro-F1 with "
        f"tie_tolerance={tie_tolerance:.4f}, neural_epochs_cap={int(neural_epochs_cap)}): "
        + ", ".join(f"{k} -> '{v}'" for k, v in selected_by_model.items())
        + "."
    )
    return ImbalanceExperimentResult(
        smote_metrics=xgb_metrics["smote"],
        class_weight_metrics=xgb_metrics["class_weight"],
        baseline_unweighted_metrics=xgb_metrics["none"],
        selected_strategy=xgb_sel,
        justification=summary_just,
        per_model={k: v.to_dict() for k, v in per_model_outcomes.items()},
        selected_strategies_by_model=selected_by_model,
        neural_epochs_cap=int(neural_epochs_cap),
        selection_mode="per_model",
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
    imbalance_strategy: str | None = None,
    smote_kwargs: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Estimate baseline XGBoost fold mean and std Macro-F1 via stratified K-fold CV."""
    counts = np.bincount(y_raw)
    nz = counts[counts > 0]
    eff_splits = min(int(n_splits), int(nz.min())) if len(nz) else 0
    if eff_splits < 2:
        return 0.0, 0.0

    eff_strategy = imbalance_strategy or ("class_weight" if use_class_weight else "none")
    sm_kw = smote_kwargs or {}
    skf = StratifiedKFold(n_splits=eff_splits, shuffle=True, random_state=seed)
    fold_f1s: list[float] = []
    for fold_idx, (tr_i, va_i) in enumerate(skf.split(X_raw, y_raw), start=1):
        X_tr_f, y_tr_f = X_raw[tr_i], y_raw[tr_i]
        X_va_f, y_va_f = X_raw[va_i], y_raw[va_i]

        uniq = np.unique(y_tr_f)
        cw = compute_class_weight("balanced", classes=uniq, y=y_tr_f)
        cw_map = {int(c): float(w) for c, w in zip(uniq, cw, strict=False)}

        sw_f = None
        X_fit, y_fit = X_tr_f, y_tr_f
        if eff_strategy == "smote":
            X_fit, y_fit, sm_info = apply_smote_safely(
                X_tr_f,
                y_tr_f,
                max_k_neighbors=int(sm_kw.get("smote_max_k_neighbors", 5)),
                sampling_strategy=sm_kw.get("smote_sampling_strategy", "auto"),
                cap_per_class=sm_kw.get("smote_cap_per_class", 5000),
                max_multiplier=sm_kw.get("smote_max_multiplier", 5.0),
                min_real_samples=int(sm_kw.get("smote_min_real_samples", 50)),
                seed=seed + fold_idx,
            )
            hybrid_cw = compute_hybrid_smote_class_weights(sm_info, cw_map)
            sw_f = np.array([hybrid_cw[int(c)] for c in y_fit], dtype=np.float32) if hybrid_cw else None
        elif eff_strategy == "class_weight":
            sw_f = np.array([cw_map[int(c)] for c in y_tr_f], dtype=np.float32)

        clf_f, cls_f = fit_xgboost_safe_multiclass(
            X_train=X_fit,
            y_train=y_fit,
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
    cv_macro_f1_mean: float | None = None,
    cv_method_note: str = "",
    cv_imbalance_strategy: str | None = None,
    smote_kwargs: dict[str, Any] | None = None,
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
        cv_mean = (
            float(cv_macro_f1_mean)
            if cv_macro_f1_mean is not None
            else float(baseline_metrics["val_macro_f1"])
        )
        cv_std = float(cv_macro_f1_std)
        method_desc = cv_method_note or (
            f"Unified {cv_n_splits}-fold StratifiedKFold baseline Macro-F1 std ({cv_std:.4f}, ddof=1)."
        )
    else:
        src_X = cv_raw_X if cv_raw_X is not None else X_train
        src_y = cv_raw_y if cv_raw_y is not None else y_train
        eff_imb = cv_imbalance_strategy or ("class_weight" if use_cv_class_weight else "none")
        cv_mean, cv_std = _estimate_baseline_cv_macro_f1(
            X_raw=src_X,
            y_raw=src_y,
            num_classes=num_classes,
            cfg=cfg,
            xgb_device=xgb_device,
            seed=seed,
            n_splits=cv_n_splits,
            use_class_weight=use_cv_class_weight,
            imbalance_strategy=eff_imb,
            smote_kwargs=smote_kwargs,
        )
        method_desc = cv_method_note or (
            f"{cv_n_splits}-fold StratifiedKFold on {len(src_X):,} standardized rows using "
            f"'{eff_imb}' imbalance handling (sample ddof=1 std across fold Macro-F1 scores)."
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
            cv_method_note=method_desc,
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
        cv_method_note=method_desc,
    )
