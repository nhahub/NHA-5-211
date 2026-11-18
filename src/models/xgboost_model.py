"""XGBoost multi-class baseline and Optuna-tunable gradient-boosted tree builders."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.config import XGBoostConfig


def build_xgboost_classifier(
    num_classes: int,
    cfg: XGBoostConfig | None = None,
    params_override: dict[str, Any] | None = None,
    device: str = "cpu",
    seed: int = 42,
    early_stopping_rounds: int | None = 20,
) -> xgb.XGBClassifier:
    """Instantiate an XGBClassifier preserving the report's baseline parameters."""
    conf = cfg or XGBoostConfig()
    base_params: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "n_jobs": -1,
    }
    base_params.update(conf.baseline)
    if params_override:
        base_params.update(params_override)

    base_params["num_class"] = int(num_classes)
    base_params["random_state"] = int(seed)
    base_params["device"] = device
    if early_stopping_rounds is not None:
        base_params["early_stopping_rounds"] = int(early_stopping_rounds)
    else:
        base_params.pop("early_stopping_rounds", None)

    return xgb.XGBClassifier(**base_params)


def expand_xgboost_probas(
    raw_probs: np.ndarray,
    present_classes: np.ndarray,
    total_classes: int,
) -> np.ndarray:
    """Project local contiguous-class XGBoost probabilities back onto global class indices."""
    probs = np.asarray(raw_probs, dtype=np.float64)
    if probs.shape[1] == total_classes and np.array_equal(present_classes, np.arange(total_classes)):
        return probs
    full = np.zeros((probs.shape[0], total_classes), dtype=np.float64)
    full[:, present_classes] = probs
    return full


def fit_xgboost_safe_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    total_classes: int,
    cfg: XGBoostConfig | None = None,
    params_override: dict[str, Any] | None = None,
    device: str = "cpu",
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
    early_stopping_rounds: int | None = 20,
) -> tuple[xgb.XGBClassifier, np.ndarray]:
    """Fit XGBoost safely even when y_train omits global classes or y_val has unseen labels."""
    present_classes = np.unique(y_train).astype(np.int64)
    val_has_unseen = bool(set(np.unique(y_val).tolist()) - set(present_classes.tolist()))
    contiguous = np.array_equal(present_classes, np.arange(total_classes))

    if contiguous and not val_has_unseen:
        clf = build_xgboost_classifier(
            num_classes=total_classes,
            cfg=cfg,
            params_override=params_override,
            device=device,
            seed=seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        fit_kw: dict[str, Any] = {"eval_set": [(X_val, y_val)], "verbose": False}
        if sample_weight is not None:
            fit_kw["sample_weight"] = sample_weight
        clf.fit(X_train, y_train, **fit_kw)
        return clf, present_classes

    # Remap non-contiguous training labels onto 0..len(present_classes)-1 and hold out an
    # internal training validation slice if validation contains unseen classes (e.g. by_day).
    cls_to_local = {int(c): idx for idx, c in enumerate(present_classes)}
    y_local = np.array([cls_to_local[int(c)] for c in y_train], dtype=np.int64)

    use_es = early_stopping_rounds
    counts = np.bincount(y_local)
    can_split_internal = len(counts) > 1 and int(counts.min()) >= 2 and len(y_local) >= 20
    if not val_has_unseen:
        y_val_local = np.array([cls_to_local[int(c)] for c in y_val], dtype=np.int64)
        X_fit, y_fit, sw_fit = X_train, y_local, sample_weight
        X_es, y_es = X_val, y_val_local
    elif can_split_internal:
        idx = np.arange(len(y_local))
        i_tr, i_es = train_test_split(idx, test_size=0.12, stratify=y_local, random_state=seed)
        X_fit, y_fit = X_train[i_tr], y_local[i_tr]
        X_es, y_es = X_train[i_es], y_local[i_es]
        sw_fit = sample_weight[i_tr] if sample_weight is not None else None
    else:
        X_fit, y_fit, sw_fit = X_train, y_local, sample_weight
        X_es, y_es = None, None
        use_es = None

    clf = build_xgboost_classifier(
        num_classes=len(present_classes),
        cfg=cfg,
        params_override=params_override,
        device=device,
        seed=seed,
        early_stopping_rounds=use_es,
    )
    fit_kw = {"verbose": False}
    if X_es is not None and y_es is not None:
        fit_kw["eval_set"] = [(X_es, y_es)]
    if sw_fit is not None:
        fit_kw["sample_weight"] = sw_fit
    clf.fit(X_fit, y_fit, **fit_kw)
    return clf, present_classes
