"""XGBoost multi-class baseline and Optuna-tunable gradient-boosted tree builders."""

from __future__ import annotations

from typing import Any

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
