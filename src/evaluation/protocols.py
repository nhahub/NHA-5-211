"""Secondary evaluation protocols: open-set cross-day (by_day), chronological drift (time_block), and feature ablation."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

from src.config import AppConfig, ModelConfig
from src.data.cleaner import clean_cicids_dataframe
from src.data.feature_engineering import (
    HighMissingColumnDropper,
    MedianImputerTransformer,
    SigmaOutlierClipper,
)
from src.data.loader import discover_csv_files, load_cicids2017_corpus
from src.data.preprocessing import (
    PreparedDataset,
    _classify_day_bucket,
    _row_hash_list,
    _split_indices_time_block,
    apply_smote_safely,
    compute_hybrid_smote_class_weights,
    drop_features_from_prepared_dataset,
    run_preprocessing_pipeline,
)
from src.evaluation.metrics import (
    _binary_metrics,
    evaluate_autoencoder,
    evaluate_supervised_model,
)
from src.explainability.shap_explainer import compute_xgboost_global_shap
from src.models.autoencoder import (
    build_autoencoder,
    calibrate_threshold,
    compute_reconstruction_error,
    extract_benign_subset,
    find_benign_class_index,
)
from src.models.cnn import build_cnn1d
from src.models.mlp import build_mlp
from src.models.xgboost_model import expand_xgboost_probas, fit_xgboost_safe_multiclass
from src.training.callbacks import build_keras_callbacks
from src.utils.logging import get_logger
from src.utils.reproducibility import configure_hardware, set_global_seed

logger = get_logger("smart_ids.evaluation.protocols")


def _select_supervised_train_arrays(
    dataset: PreparedDataset,
    imbalance_strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[int, float] | None]:
    """Select standardized or SMOTE training arrays and matching weights for a strategy."""
    if imbalance_strategy == "smote":
        hybrid_cw = dataset.smote_class_weights
        sw = (
            np.array([hybrid_cw[int(c)] for c in dataset.y_train_sm], dtype=np.float32) if hybrid_cw else None
        )
        return dataset.X_train_sm, dataset.y_train_sm, sw, hybrid_cw
    if imbalance_strategy == "class_weight":
        sample_w = np.array([dataset.class_weights[int(c)] for c in dataset.y_train], dtype=np.float32)
        return dataset.X_train_scaled, dataset.y_train, sample_w, dataset.class_weights
    return dataset.X_train_scaled, dataset.y_train, None, None


def _internal_train_es_split(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
    es_fraction: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Hold out an internal stratified validation slice from training rows for early stopping."""
    counts = np.bincount(y)
    nz = counts[counts > 0]
    can_stratify = len(nz) > 1 and int(nz.min()) >= 2 and int(len(y) * es_fraction) >= len(nz)
    idx = np.arange(len(y))
    i_tr, i_es = train_test_split(
        idx,
        test_size=es_fraction,
        stratify=y if can_stratify else None,
        random_state=seed,
    )
    sw_tr = sample_weight[i_tr] if sample_weight is not None else None
    return X[i_tr], y[i_tr], sw_tr, X[i_es], y[i_es]


def _can_run_by_day_split(app_cfg: AppConfig) -> tuple[bool, str]:
    """Check whether raw_data_dir has CSVs covering Monday-Wednesday, Thursday, and Friday."""
    raw_dir = app_cfg.paths.resolve("raw_data_dir")
    csv_files = discover_csv_files(raw_dir)
    if not csv_files:
        return False, "No raw CICIDS2017 CSV files found in raw_data_dir."
    buckets: set[str] = set()
    try:
        for path in csv_files:
            buckets.add(_classify_day_bucket(path.name))
    except ValueError as exc:
        return False, str(exc)
    required = {"train_mon_wed", "thursday", "friday"}
    if not required.issubset(buckets):
        missing = sorted(required - buckets)
        return False, f"Raw CSVs do not cover required by_day buckets: missing {missing}."
    return True, "ok"


def _summarize_multiclass_by_day(
    eval_dict: dict[str, Any],
    y_true: np.ndarray,
    proba: np.ndarray,
) -> dict[str, Any]:
    """Extract compact closed-set multi-class metrics including present-class Macro-F1."""
    y_pred = np.argmax(proba, axis=1)
    present_idx = [int(c) for c in np.unique(y_true)]
    f1_macro_present = float(f1_score(y_true, y_pred, labels=present_idx, average="macro", zero_division=0))
    return {
        "accuracy": float(eval_dict["accuracy"]),
        "precision_weighted": float(eval_dict["precision_weighted"]),
        "recall_weighted": float(eval_dict["recall_weighted"]),
        "f1_weighted": float(eval_dict["f1_weighted"]),
        "f1_macro": float(eval_dict["f1_macro"]),
        "f1_macro_present_test_classes": f1_macro_present,
        "roc_auc_macro_ovr": float(eval_dict["roc_auc_macro_ovr"]),
    }


def _fit_and_predict_keras_pair(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_es: np.ndarray,
    y_es: np.ndarray,
    X_test: np.ndarray,
    num_features: int,
    num_classes: int,
    model_cfg: ModelConfig,
    epochs: int,
    keras_cw: dict[int, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train in-memory MLP and 1D-CNN classifiers without modifying persisted checkpoints."""
    mlp = build_mlp(num_features, num_classes, cfg=model_cfg.mlp)
    mlp_cb = build_keras_callbacks(
        checkpoint_path=None,
        early_stopping_patience=model_cfg.mlp.early_stopping_patience,
        reduce_lr_patience=model_cfg.mlp.reduce_lr_patience,
        reduce_lr_factor=model_cfg.mlp.reduce_lr_factor,
        min_lr=model_cfg.mlp.min_lr,
    )
    mlp.fit(
        X_fit,
        y_fit,
        validation_data=(X_es, y_es),
        epochs=epochs,
        batch_size=model_cfg.mlp.batch_size,
        callbacks=mlp_cb,
        class_weight=keras_cw,
        verbose=0,
    )
    p_mlp = np.asarray(mlp.predict(X_test, batch_size=1024, verbose=0), dtype=np.float64)

    cnn = build_cnn1d(num_features, num_classes, cfg=model_cfg.cnn)
    cnn_cb = build_keras_callbacks(
        checkpoint_path=None,
        early_stopping_patience=model_cfg.cnn.early_stopping_patience,
        reduce_lr_patience=model_cfg.cnn.reduce_lr_patience,
        reduce_lr_factor=model_cfg.cnn.reduce_lr_factor,
        min_lr=model_cfg.cnn.min_lr,
    )
    cnn.fit(
        X_fit.reshape(-1, num_features, 1),
        y_fit,
        validation_data=(X_es.reshape(-1, num_features, 1), y_es),
        epochs=epochs,
        batch_size=model_cfg.cnn.batch_size,
        callbacks=cnn_cb,
        class_weight=keras_cw,
        verbose=0,
    )
    p_cnn = np.asarray(
        cnn.predict(X_test.reshape(-1, num_features, 1), batch_size=1024, verbose=0),
        dtype=np.float64,
    )
    return p_mlp, p_cnn


def _fit_and_eval_in_memory_autoencoder(
    dataset: PreparedDataset,
    app_cfg: AppConfig,
    model_cfg: ModelConfig,
    epochs: int,
    bootstrap_n: int = 0,
    seed: int = 42,
) -> tuple[dict[str, Any], float]:
    """Train in-memory benign-only Autoencoder and evaluate on dataset test flows."""
    num_features = dataset.X_train_scaled.shape[1]
    X_benign_train, benign_idx, _ = extract_benign_subset(
        dataset.X_train_scaled, dataset.y_train, dataset.label_encoder
    )
    X_benign_val, _, _ = extract_benign_subset(dataset.X_val_scaled, dataset.y_val, dataset.label_encoder)
    ae_model, _, _ = build_autoencoder(num_features, cfg=model_cfg.autoencoder)
    ae_cb = build_keras_callbacks(
        checkpoint_path=None,
        early_stopping_patience=model_cfg.autoencoder.early_stopping_patience,
        reduce_lr_patience=model_cfg.autoencoder.reduce_lr_patience,
        reduce_lr_factor=model_cfg.autoencoder.reduce_lr_factor,
        min_lr=model_cfg.autoencoder.min_lr,
    )
    ae_model.fit(
        X_benign_train,
        X_benign_train,
        validation_data=(X_benign_val, X_benign_val),
        epochs=epochs,
        batch_size=model_cfg.autoencoder.batch_size,
        callbacks=ae_cb,
        verbose=0,
    )
    benign_val_errors = compute_reconstruction_error(ae_model, X_benign_val)
    ae_threshold, _ = calibrate_threshold(
        benign_val_errors, percentile=model_cfg.autoencoder.threshold_percentile
    )
    ae_test_errors = compute_reconstruction_error(ae_model, dataset.X_test_scaled)
    ae_eval = evaluate_autoencoder(
        y_test_multiclass=dataset.y_test,
        recon_errors_test=ae_test_errors,
        benign_val_errors=benign_val_errors,
        threshold=ae_threshold,
        class_names=dataset.class_names,
        benign_index=benign_idx,
        percentiles=app_cfg.evaluation.autoencoder_percentiles,
        bootstrap_n=bootstrap_n,
        seed=seed,
    )
    return ae_eval, float(ae_threshold)


def evaluate_by_day_protocol(
    app_cfg: AppConfig,
    model_cfg: ModelConfig,
    xgb_params: dict[str, Any] | None = None,
    imbalance_strategy: str = "class_weight",
    keras_epochs: int | None = None,
    force_synthetic: bool = False,
) -> dict[str, Any]:
    """Run open-set cross-day evaluation: Mon-Wed train, part of Thu val, Thu-Fri unseen attacks."""
    if force_synthetic:
        return {
            "executed": False,
            "reason": "Skipped open-set cross-day (by_day) evaluation under synthetic mode (requires Monday-Friday CICIDS2017 CSVs).",
        }

    ok, reason = _can_run_by_day_split(app_cfg)
    if not ok:
        logger.warning("Skipping open-set cross-day (by_day) comparison: %s", reason)
        return {"executed": False, "reason": reason}

    seed = app_cfg.project.random_seed
    set_global_seed(seed)
    hardware = configure_hardware(
        use_gpu=model_cfg.hardware.use_gpu,
        memory_growth=model_cfg.hardware.memory_growth,
        mixed_precision=model_cfg.hardware.mixed_precision,
    )

    by_day_cfg = app_cfg.model_copy(deep=True)
    by_day_cfg.data.split_strategy = "by_day"
    dataset = run_preprocessing_pipeline(by_day_cfg, force_synthetic=False, save_outputs=False)

    num_features = dataset.X_train_scaled.shape[1]
    num_classes = len(dataset.class_names)
    benign_idx, benign_name = find_benign_class_index(dataset.label_encoder)
    y_test_bin = (dataset.y_test != benign_idx).astype(np.int64)

    X_sup_train, y_sup_train, sample_w, keras_cw = _select_supervised_train_arrays(
        dataset, imbalance_strategy
    )
    # Supervise Keras early stopping exclusively on an internal Mon-Wed holdout slice so
    # unseen Thursday attack classes cannot distort validation loss tracking.
    X_fit, y_fit, sw_fit, X_es, y_es = _internal_train_es_split(X_sup_train, y_sup_train, sample_w, seed=seed)

    multiclass_table: dict[str, dict[str, Any]] = {}
    binary_table: dict[str, dict[str, Any]] = {}

    # 1. XGBoost Multi-Class (Mon-Wed classes projected onto full class space)
    xgb_clf, present_cls = fit_xgboost_safe_multiclass(
        X_train=X_sup_train,
        y_train=y_sup_train,
        X_val=dataset.X_val_scaled,
        y_val=dataset.y_val,
        total_classes=num_classes,
        cfg=model_cfg.xgboost,
        params_override=xgb_params,
        device=hardware.xgb_device,
        seed=seed,
        sample_weight=sample_w,
    )
    p_xgb = expand_xgboost_probas(xgb_clf.predict_proba(dataset.X_test_scaled), present_cls, num_classes)
    xgb_mc_eval = evaluate_supervised_model(dataset.y_test, p_xgb, dataset.class_names, "XGBoost")
    multiclass_table["XGBoost"] = _summarize_multiclass_by_day(xgb_mc_eval, dataset.y_test, p_xgb)
    binary_table["XGBoost (1 - P_Benign)"] = _binary_metrics(
        y_test_bin, 1.0 - p_xgb[:, benign_idx], threshold=0.5
    )

    # 2. Dedicated Binary XGBoost (Benign vs Attack trained on Mon-Wed attacks, tested Thu-Fri)
    y_bin_tr = (y_fit != benign_idx).astype(np.int64)
    y_bin_es = (y_es != benign_idx).astype(np.int64)
    bin_params = {
        k: v
        for k, v in (model_cfg.xgboost.baseline | (xgb_params or {})).items()
        if k
        not in {"num_class", "objective", "eval_metric", "early_stopping_rounds", "random_state", "device"}
    }
    xgb_bin = xgb.XGBClassifier(
        **bin_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        device=hardware.xgb_device,
        early_stopping_rounds=20,
    )
    fit_bin_kw: dict[str, Any] = {"eval_set": [(X_es, y_bin_es)], "verbose": False}
    if sw_fit is not None:
        # Balanced binary weight for Attack vs Benign when using class_weight
        n_pos = max(1, int(np.sum(y_bin_tr == 1)))
        n_neg = max(1, int(np.sum(y_bin_tr == 0)))
        w_pos = float(n_neg / n_pos)
        fit_bin_kw["sample_weight"] = np.where(y_bin_tr == 1, w_pos, 1.0).astype(np.float32)
    xgb_bin.fit(X_fit, y_bin_tr, **fit_bin_kw)
    p_xgb_bin = np.asarray(xgb_bin.predict_proba(dataset.X_test_scaled)[:, 1], dtype=np.float64)
    binary_table["XGBoost (Binary)"] = _binary_metrics(y_test_bin, p_xgb_bin, threshold=0.5)

    # 3. MLP & 1D-CNN in-memory training (no checkpoint overwrite in models/)
    ep = int(keras_epochs or app_cfg.evaluation.by_day_keras_epochs)
    p_mlp, p_cnn = _fit_and_predict_keras_pair(
        X_fit=X_fit,
        y_fit=y_fit,
        X_es=X_es,
        y_es=y_es,
        X_test=dataset.X_test_scaled,
        num_features=num_features,
        num_classes=num_classes,
        model_cfg=model_cfg,
        epochs=ep,
        keras_cw=keras_cw,
    )
    mlp_mc_eval = evaluate_supervised_model(dataset.y_test, p_mlp, dataset.class_names, "MLP")
    multiclass_table["MLP"] = _summarize_multiclass_by_day(mlp_mc_eval, dataset.y_test, p_mlp)
    binary_table["MLP (1 - P_Benign)"] = _binary_metrics(
        y_test_bin, 1.0 - p_mlp[:, benign_idx], threshold=0.5
    )

    cnn_mc_eval = evaluate_supervised_model(dataset.y_test, p_cnn, dataset.class_names, "1D-CNN")
    multiclass_table["1D-CNN"] = _summarize_multiclass_by_day(cnn_mc_eval, dataset.y_test, p_cnn)
    binary_table["1D-CNN (1 - P_Benign)"] = _binary_metrics(
        y_test_bin, 1.0 - p_cnn[:, benign_idx], threshold=0.5
    )

    # 4. Autoencoder (BENIGN Mon-Wed train, threshold strictly calibrated on Thursday BENIGN val)
    ae_eval, ae_threshold = _fit_and_eval_in_memory_autoencoder(
        dataset=dataset,
        app_cfg=app_cfg,
        model_cfg=model_cfg,
        epochs=ep,
        bootstrap_n=0,
        seed=seed,
    )
    binary_table["Autoencoder"] = {
        k: float(ae_eval[k])
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr")
    }

    split_cls = dataset.audit_report.get("split_classes_present", {})
    train_cls = split_cls.get("train", [])
    test_cls = split_cls.get("test", [])
    unseen_test_cls = sorted(c for c in set(test_cls) - set(train_cls) if c != benign_name)

    xgb_1p = binary_table.get("XGBoost (1 - P_Benign)", {})
    xgb_rec = float(xgb_1p.get("recall", 0.0))
    xgb_prec = float(xgb_1p.get("precision", 0.0))
    miss_pct = 100.0 * (1.0 - xgb_rec)

    return {
        "executed": True,
        "protocol_name": "open-set cross-day",
        "split_strategy": "by_day",
        "imbalance_strategy": imbalance_strategy,
        "keras_epochs": ep,
        "split_sizes": dataset.audit_report["split_sizes"],
        "split_day_files": dataset.audit_report.get("split_day_files", {}),
        "leakage_checks": dataset.audit_report["leakage_checks"],
        "train_classes": train_cls,
        "val_classes": split_cls.get("validation", []),
        "test_classes": test_cls,
        "unseen_test_classes": unseen_test_cls,
        "multiclass_models": multiclass_table,
        "binary_cross_day_models": binary_table,
        "autoencoder_threshold": float(ae_threshold),
        "caveat_note": (
            "Open-set cross-day protocol (by_day): by construction in CICIDS2017, all Thursday and Friday "
            f"attack families ({', '.join(unseen_test_cls)}) have zero training instances in Monday-Wednesday data. "
            "Closed-set multi-class Macro-F1 is therefore near zero by design rather than ordinary generalization "
            "degradation. The real empirical finding is that supervised XGBoost misses "
            f"~{miss_pct:.0f}% of unseen attack flows (binary recall={xgb_rec:.4f}, precision={xgb_prec:.4f}), "
            "which is why benign-only anomaly detectors are kept as a supplementary zero-day signal."
        ),
    }


def evaluate_time_block_protocol(
    app_cfg: AppConfig,
    model_cfg: ModelConfig,
    xgb_params: dict[str, Any] | None = None,
    imbalance_strategy: str = "none",
    keras_epochs: int | None = None,
    force_synthetic: bool = False,
) -> dict[str, Any]:
    """Evaluate closed-set temporal drift using chronological per-class ~60%/~20%/~20% splits."""
    seed = app_cfg.project.random_seed
    set_global_seed(seed)
    hardware = configure_hardware(
        use_gpu=model_cfg.hardware.use_gpu,
        memory_growth=model_cfg.hardware.memory_growth,
        mixed_precision=model_cfg.hardware.mixed_precision,
    )

    time_cfg = app_cfg.model_copy(deep=True)
    time_cfg.data.split_strategy = "time_block"
    dataset = run_preprocessing_pipeline(time_cfg, force_synthetic=force_synthetic, save_outputs=False)

    num_features = dataset.X_train_scaled.shape[1]
    num_classes = len(dataset.class_names)
    boot_n = int(app_cfg.evaluation.bootstrap_n_resamples)
    low_sup = int(app_cfg.evaluation.low_support_threshold)

    X_sup_train, y_sup_train, sample_w, keras_cw = _select_supervised_train_arrays(
        dataset, imbalance_strategy
    )

    # 1. XGBoost (baseline or selected variant) evaluated against chronological middle val / latest test
    xgb_clf, present_cls = fit_xgboost_safe_multiclass(
        X_train=X_sup_train,
        y_train=y_sup_train,
        X_val=dataset.X_val_scaled,
        y_val=dataset.y_val,
        total_classes=num_classes,
        cfg=model_cfg.xgboost,
        params_override=xgb_params,
        device=hardware.xgb_device,
        seed=seed,
        sample_weight=sample_w,
    )
    p_xgb = expand_xgboost_probas(xgb_clf.predict_proba(dataset.X_test_scaled), present_cls, num_classes)

    # 2. MLP & 1D-CNN trained in memory on earliest 60%, early-stopped on middle 20%
    ep = int(keras_epochs or app_cfg.evaluation.time_block_keras_epochs)
    p_mlp, p_cnn = _fit_and_predict_keras_pair(
        X_fit=X_sup_train,
        y_fit=y_sup_train,
        X_es=dataset.X_val_scaled,
        y_es=dataset.y_val,
        X_test=dataset.X_test_scaled,
        num_features=num_features,
        num_classes=num_classes,
        model_cfg=model_cfg,
        epochs=ep,
        keras_cw=keras_cw,
    )

    models_summary: dict[str, dict[str, Any]] = {}
    per_class_by_model: dict[str, list[dict[str, Any]]] = {}
    for display_name, proba in (("XGBoost", p_xgb), ("1D-CNN", p_cnn), ("MLP", p_mlp)):
        ev = evaluate_supervised_model(
            dataset.y_test,
            proba,
            dataset.class_names,
            display_name,
            bootstrap_n=boot_n,
            low_support_threshold=low_sup,
            seed=seed,
        )
        ci = ev.get("f1_macro_ci95", {})
        models_summary[display_name] = {
            "accuracy": float(ev["accuracy"]),
            "precision_weighted": float(ev["precision_weighted"]),
            "recall_weighted": float(ev["recall_weighted"]),
            "f1_weighted": float(ev["f1_weighted"]),
            "f1_macro": float(ev["f1_macro"]),
            "f1_macro_ci95": ci,
            "roc_auc_macro_ovr": float(ev["roc_auc_macro_ovr"]),
            "pr_auc_macro_ovr": float(ev["pr_auc_macro_ovr"]),
        }
        per_class_by_model[display_name] = list(ev.get("per_class_metrics", []))

    ae_eval, ae_threshold = _fit_and_eval_in_memory_autoencoder(
        dataset=dataset,
        app_cfg=app_cfg,
        model_cfg=model_cfg,
        epochs=ep,
        bootstrap_n=boot_n,
        seed=seed,
    )

    tb_details = dataset.audit_report.get("time_block_details", {})
    return {
        "executed": True,
        "split_strategy": "time_block",
        "imbalance_strategy": imbalance_strategy,
        "keras_epochs": ep,
        "ordering_source": tb_details.get("ordering_source", ""),
        "tiny_class_fallbacks": tb_details.get("tiny_class_fallbacks", {}),
        "per_class_allocations": tb_details.get("per_class_allocations", {}),
        "split_sizes": dataset.audit_report["split_sizes"],
        "post_clip_dedup_removed": dataset.audit_report.get("post_clip_dedup_removed", {}),
        "leakage_checks": dataset.audit_report["leakage_checks"],
        "models": models_summary,
        "per_class_metrics": per_class_by_model,
        "autoencoder": {
            "accuracy": float(ae_eval["accuracy"]),
            "precision": float(ae_eval["precision"]),
            "recall": float(ae_eval["recall"]),
            "f1": float(ae_eval["f1"]),
            "f1_ci95": ae_eval.get("f1_ci95", {}),
            "roc_auc": float(ae_eval["roc_auc"]),
            "pr_auc": float(ae_eval["pr_auc"]),
            "fpr": float(ae_eval["fpr"]),
            "threshold": float(ae_threshold),
        },
    }


def _indices_sha256(indices: np.ndarray | None) -> str:
    """Compute canonical SHA-256 hex digest of integer row indices."""
    if indices is None:
        return ""
    arr = np.ascontiguousarray(indices, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def assert_ablation_row_parity(ds_with: PreparedDataset, ds_without: PreparedDataset) -> None:
    """Fail loudly unless full-feature and ablated splits evaluate identical row indices and targets."""
    if not ds_with.audit_report.get("leakage_checks", {}).get("passed", False):
        raise AssertionError("Ablation baseline dataset failed full-feature zero-leakage verification.")
    for split_name, arr_w, arr_wo in (
        ("train", ds_with.train_row_indices, ds_without.train_row_indices),
        ("validation", ds_with.val_row_indices, ds_without.val_row_indices),
        ("test", ds_with.test_row_indices, ds_without.test_row_indices),
    ):
        if arr_w is None or arr_wo is None:
            raise AssertionError(
                f"Missing {split_name}_row_indices required to verify apples-to-apples ablation parity."
            )
        if len(arr_w) != len(arr_wo) or not np.array_equal(arr_w, arr_wo):
            raise AssertionError(
                f"Ablation {split_name} split row mismatch: with={len(arr_w)} vs without={len(arr_wo)}."
            )
    if len(ds_with.X_test_scaled) != len(ds_without.X_test_scaled):
        raise AssertionError(
            f"Ablation Test_Rows mismatch: with={len(ds_with.X_test_scaled)} "
            f"vs without={len(ds_without.X_test_scaled)}."
        )
    if not np.array_equal(ds_with.y_test, ds_without.y_test):
        raise AssertionError("Ablation y_test label sequence differs between variants.")


def _train_and_eval_xgb_for_ablation(
    dataset: PreparedDataset,
    model_cfg: ModelConfig,
    xgb_params: dict[str, Any] | None,
    imbalance_strategy: str,
    seed: int,
    xgb_device: str,
    compute_shap: bool,
) -> dict[str, Any]:
    """Fit matched XGBoost classifier cleanly on a preprocessed dataset variant and evaluate val/test."""
    num_classes = len(dataset.class_names)
    X_sup_train, y_sup_train, sample_w, _ = _select_supervised_train_arrays(dataset, imbalance_strategy)
    clf, present_cls = fit_xgboost_safe_multiclass(
        X_train=X_sup_train,
        y_train=y_sup_train,
        X_val=dataset.X_val_scaled,
        y_val=dataset.y_val,
        total_classes=num_classes,
        cfg=model_cfg.xgboost,
        params_override=xgb_params,
        device=xgb_device,
        seed=seed,
        sample_weight=sample_w,
    )
    p_val = expand_xgboost_probas(clf.predict_proba(dataset.X_val_scaled), present_cls, num_classes)
    p_test = expand_xgboost_probas(clf.predict_proba(dataset.X_test_scaled), present_cls, num_classes)

    val_eval = evaluate_supervised_model(dataset.y_val, p_val, dataset.class_names, "XGBoost")
    test_eval = evaluate_supervised_model(dataset.y_test, p_test, dataset.class_names, "XGBoost")

    top_shap: list[dict[str, Any]] = []
    if compute_shap and len(dataset.X_test_scaled) > 0:
        shap_cls_names = [dataset.class_names[int(i)] for i in present_cls]
        shap_res, _, _ = compute_xgboost_global_shap(
            xgb_model=clf,
            X_test=dataset.X_test_scaled,
            feature_names=dataset.feature_names,
            class_names=shap_cls_names,
            max_samples=min(250, len(dataset.X_test_scaled)),
            summary_fig_path=None,
        )
        top_shap = shap_res.feature_importance[:10]

    def _compact(ev: dict[str, Any]) -> dict[str, float]:
        return {
            "accuracy": float(ev["accuracy"]),
            "precision_weighted": float(ev["precision_weighted"]),
            "recall_weighted": float(ev["recall_weighted"]),
            "f1_weighted": float(ev["f1_weighted"]),
            "f1_macro": float(ev["f1_macro"]),
            "roc_auc_macro_ovr": float(ev["roc_auc_macro_ovr"]),
        }

    return {
        "n_features": int(len(dataset.feature_names)),
        "dropped_features": list(dataset.audit_report.get("dropped_features", [])),
        "split_sizes": dataset.audit_report["split_sizes"],
        "test_rows_sha256": _indices_sha256(dataset.test_row_indices),
        "leakage_checks": dataset.audit_report["leakage_checks"],
        "val": _compact(val_eval),
        "test": _compact(test_eval),
        "top_shap_features": top_shap,
    }


def evaluate_feature_ablation(
    app_cfg: AppConfig,
    model_cfg: ModelConfig,
    drop_features: list[str] | None = None,
    xgb_params: dict[str, Any] | None = None,
    imbalance_strategy: str = "smote",
    force_synthetic: bool = False,
    compute_shap: bool = True,
) -> dict[str, Any]:
    """Compare XGBoost with and without shortcut feature(s) on identical train/val/test rows."""
    target_drops = list(drop_features or app_cfg.evaluation.ablation_drop_features or ["Destination Port"])
    seed = app_cfg.project.random_seed
    set_global_seed(seed)
    hardware = configure_hardware(
        use_gpu=model_cfg.hardware.use_gpu,
        memory_growth=model_cfg.hardware.memory_growth,
        mixed_precision=model_cfg.hardware.mixed_precision,
    )

    cfg_with = app_cfg.model_copy(deep=True)
    cfg_with.data.drop_features = []
    cfg_without = app_cfg.model_copy(deep=True)
    cfg_without.data.drop_features = target_drops

    # Run full pipeline (split, clip, scale, dedup, leakage check) on all features first,
    # then drop target_drops from the final matrices of those exact same rows.
    ds_with = run_preprocessing_pipeline(cfg_with, force_synthetic=force_synthetic, save_outputs=False)
    ds_without = drop_features_from_prepared_dataset(ds_with, target_drops, cfg_without)
    assert_ablation_row_parity(ds_with, ds_without)

    res_with = _train_and_eval_xgb_for_ablation(
        ds_with, model_cfg, xgb_params, imbalance_strategy, seed, hardware.xgb_device, compute_shap
    )
    res_without = _train_and_eval_xgb_for_ablation(
        ds_without, model_cfg, xgb_params, imbalance_strategy, seed, hardware.xgb_device, compute_shap
    )

    delta_test_f1_macro = res_without["test"]["f1_macro"] - res_with["test"]["f1_macro"]
    delta_val_f1_macro = res_without["val"]["f1_macro"] - res_with["val"]["f1_macro"]
    delta_test_f1_weighted = res_without["test"]["f1_weighted"] - res_with["test"]["f1_weighted"]
    delta_test_acc = res_without["test"]["accuracy"] - res_with["test"]["accuracy"]
    delta_test_roc = res_without["test"]["roc_auc_macro_ovr"] - res_with["test"]["roc_auc_macro_ovr"]

    logger.info(
        "Feature ablation (%s) on identical %d test rows: Test Macro-F1 %.4f -> %.4f (delta=%+.4f)",
        target_drops,
        len(ds_with.X_test_scaled),
        res_with["test"]["f1_macro"],
        res_without["test"]["f1_macro"],
        delta_test_f1_macro,
    )

    return {
        "executed": True,
        "split_strategy": app_cfg.data.split_strategy,
        "imbalance_strategy": imbalance_strategy,
        "dropped_features": target_drops,
        "identical_test_rows_verified": True,
        "test_rows": int(len(ds_with.X_test_scaled)),
        "with_features": res_with,
        "without_features": res_without,
        "delta_test_f1_macro": round(float(delta_test_f1_macro), 6),
        "delta_val_f1_macro": round(float(delta_val_f1_macro), 6),
        "delta_test_f1_weighted": round(float(delta_test_f1_weighted), 6),
        "delta_test_accuracy": round(float(delta_test_acc), 6),
        "delta_test_roc_auc_macro_ovr": round(float(delta_test_roc), 6),
    }


def _prepare_cv_fold(
    X_tr_raw: pd.DataFrame,
    y_tr: np.ndarray,
    X_va_raw: pd.DataFrame,
    y_va: np.ndarray,
    cfg: AppConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit fold-only transformers, remove post-clip cross-fold collisions, and scale to float32."""
    col_dropper = HighMissingColumnDropper(nan_threshold=cfg.data.nan_column_drop_threshold)
    X_tr_k = col_dropper.fit_transform(X_tr_raw)
    X_va_k = col_dropper.transform(X_va_raw)

    imputer = MedianImputerTransformer()
    X_tr_imp = imputer.fit_transform(X_tr_k)
    X_va_imp = imputer.transform(X_va_k)

    clipper = SigmaOutlierClipper(n_sigma=cfg.data.outlier_clip_sigma)
    X_tr_clip = clipper.fit_transform(X_tr_imp)
    X_va_clip = clipper.transform(X_va_imp)

    # Deduplicate clipped fold train and strip any validation row matching fold train features.
    tr_hashes = _row_hash_list(X_tr_clip.to_numpy(dtype=np.float64, copy=False))
    seen_tr_pairs: set[tuple[bytes, int]] = set()
    tr_feat_set: set[bytes] = set()
    tr_keep = np.zeros(len(X_tr_clip), dtype=bool)
    for idx, (h, lbl) in enumerate(zip(tr_hashes, y_tr, strict=True)):
        pair = (h, int(lbl))
        if pair not in seen_tr_pairs:
            seen_tr_pairs.add(pair)
            tr_feat_set.add(h)
            tr_keep[idx] = True

    va_hashes = _row_hash_list(X_va_clip.to_numpy(dtype=np.float64, copy=False))
    va_keep = np.array([h not in tr_feat_set for h in va_hashes], dtype=bool)

    X_tr_dedup = X_tr_clip.loc[tr_keep].reset_index(drop=True)
    y_tr_dedup = y_tr[tr_keep]
    X_va_dedup = X_va_clip.loc[va_keep].reset_index(drop=True)
    y_va_dedup = y_va[va_keep]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_dedup).astype(np.float32)
    X_va_sc = scaler.transform(X_va_dedup).astype(np.float32)

    tr_scaled_set = set(_row_hash_list(X_tr_sc, decimals=6))
    va_scaled_hashes = _row_hash_list(X_va_sc, decimals=6)
    va_scaled_keep = np.array([h not in tr_scaled_set for h in va_scaled_hashes], dtype=bool)
    return X_tr_sc, y_tr_dedup, X_va_sc[va_scaled_keep], y_va_dedup[va_scaled_keep]


def run_xgboost_stratified_cv(
    app_cfg: AppConfig,
    model_cfg: ModelConfig,
    xgb_params: dict[str, Any] | None = None,
    imbalance_strategy: str = "class_weight",
    n_splits: int = 5,
    force_synthetic: bool = False,
) -> dict[str, Any]:
    """Run leak-free k-fold Stratified Cross-Validation for XGBoost on the development pool."""
    seed = app_cfg.project.random_seed
    set_global_seed(seed)
    hardware = configure_hardware(
        use_gpu=model_cfg.hardware.use_gpu,
        memory_growth=model_cfg.hardware.memory_growth,
        mixed_precision=model_cfg.hardware.mixed_precision,
    )

    raw_dir = app_cfg.paths.resolve("raw_data_dir")
    syn_dir = app_cfg.paths.resolve("synthetic_data_dir")
    df_raw, _ = load_cicids2017_corpus(
        raw_dir=raw_dir,
        synthetic_dir=syn_dir,
        data_cfg=app_cfg.data,
        seed=seed,
        force_synthetic=force_synthetic,
    )
    X_df, y_series, metadata, _ = clean_cicids_dataframe(
        df_raw,
        label_col=app_cfg.data.label_column,
        identifier_keywords=app_cfg.data.identifier_keywords,
        preserve_features=app_cfg.data.preserve_features,
        drop_features=app_cfg.data.drop_features,
        merge_web_attacks=app_cfg.data.merge_web_attacks,
        drop_ultra_rare_below=app_cfg.data.drop_ultra_rare_classes_below,
    )

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_series.astype(str))
    class_names = [str(c) for c in label_encoder.classes_]
    num_classes = len(class_names)

    # Keep the final held-out test split isolated from CV (respecting time_block if configured).
    if app_cfg.data.split_strategy == "time_block":
        idx_tr_tb, idx_va_tb, _, _ = _split_indices_time_block(
            metadata=metadata,
            y_all=y_all,
            class_names=class_names,
            train_fraction=app_cfg.data.split.time_block_train_fraction,
            val_fraction=app_cfg.data.split.time_block_val_fraction,
        )
        idx_dev = np.sort(np.concatenate([idx_tr_tb, idx_va_tb]))
    else:
        idx_all = np.arange(len(X_df))
        idx_dev, _ = train_test_split(
            idx_all,
            test_size=app_cfg.data.split.test_size,
            random_state=seed,
            stratify=y_all if app_cfg.data.split.stratify else None,
        )
    X_dev = X_df.iloc[idx_dev].reset_index(drop=True)
    y_dev = y_all[idx_dev].astype(np.int64)

    counts_dev = np.bincount(y_dev)
    nz_dev = counts_dev[counts_dev > 0]
    min_cls_count = int(nz_dev.min()) if len(nz_dev) else 0
    effective_splits = min(int(n_splits), min_cls_count)
    if effective_splits < 2:
        return {
            "executed": False,
            "reason": f"Insufficient minimum class support ({min_cls_count}) for stratified CV.",
        }
    if effective_splits < n_splits:
        logger.warning(
            "Reducing stratified CV folds from %d to %d due to smallest dev class count=%d.",
            n_splits,
            effective_splits,
            min_cls_count,
        )

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    fold_results: list[dict[str, Any]] = []

    for fold_idx, (tr_i, va_i) in enumerate(skf.split(X_dev, y_dev), start=1):
        X_tr_raw = X_dev.iloc[tr_i].reset_index(drop=True)
        y_tr_fold = y_dev[tr_i]
        X_va_raw = X_dev.iloc[va_i].reset_index(drop=True)
        y_va_fold = y_dev[va_i]

        X_tr_sc, y_tr_clean, X_va_sc, y_va_clean = _prepare_cv_fold(
            X_tr_raw, y_tr_fold, X_va_raw, y_va_fold, app_cfg
        )

        uniq_cls = np.unique(y_tr_clean)
        cw_vals = compute_class_weight(class_weight="balanced", classes=uniq_cls, y=y_tr_clean)
        fold_cw = {int(c): float(w) for c, w in zip(uniq_cls, cw_vals, strict=False)}

        if imbalance_strategy == "smote":
            X_fit, y_fit, sm_info = apply_smote_safely(
                X_tr_sc,
                y_tr_clean,
                max_k_neighbors=app_cfg.data.imbalance.smote_max_k_neighbors,
                sampling_strategy=app_cfg.data.imbalance.smote_sampling_strategy,
                cap_per_class=app_cfg.data.imbalance.smote_cap_per_class,
                max_multiplier=app_cfg.data.imbalance.smote_max_multiplier,
                min_real_samples=app_cfg.data.imbalance.smote_min_real_samples,
                seed=seed + fold_idx,
            )
            hybrid_cw = compute_hybrid_smote_class_weights(sm_info, fold_cw)
            sw_fit = np.array([hybrid_cw[int(c)] for c in y_fit], dtype=np.float32) if hybrid_cw else None
        elif imbalance_strategy == "class_weight":
            X_fit, y_fit = X_tr_sc, y_tr_clean
            sw_fit = np.array([fold_cw[int(c)] for c in y_fit], dtype=np.float32)
        else:
            X_fit, y_fit = X_tr_sc, y_tr_clean
            sw_fit = None

        clf, present_cls = fit_xgboost_safe_multiclass(
            X_train=X_fit,
            y_train=y_fit,
            X_val=X_va_sc,
            y_val=y_va_clean,
            total_classes=num_classes,
            cfg=model_cfg.xgboost,
            params_override=xgb_params,
            device=hardware.xgb_device,
            seed=seed,
            sample_weight=sw_fit,
        )
        p_va = expand_xgboost_probas(clf.predict_proba(X_va_sc), present_cls, num_classes)
        ev = evaluate_supervised_model(
            y_va_clean,
            p_va,
            class_names,
            f"XGBoost (Fold {fold_idx})",
            bootstrap_n=0,
            seed=seed,
        )
        fold_results.append(
            {
                "fold": fold_idx,
                "train_rows": int(len(X_fit)),
                "val_rows": int(len(X_va_sc)),
                "accuracy": float(ev["accuracy"]),
                "precision_weighted": float(ev["precision_weighted"]),
                "recall_weighted": float(ev["recall_weighted"]),
                "f1_weighted": float(ev["f1_weighted"]),
                "f1_macro": float(ev["f1_macro"]),
                "roc_auc_macro_ovr": float(ev["roc_auc_macro_ovr"]),
            }
        )

    ddof = 1 if len(fold_results) > 1 else 0
    macro_f1s = [r["f1_macro"] for r in fold_results]
    weighted_f1s = [r["f1_weighted"] for r in fold_results]
    accs = [r["accuracy"] for r in fold_results]
    rocs = [r["roc_auc_macro_ovr"] for r in fold_results]

    mean_macro_f1 = float(np.mean(macro_f1s))
    std_macro_f1 = float(np.std(macro_f1s, ddof=ddof))
    logger.info(
        "XGBoost %d-fold Stratified CV (%s): Macro-F1 = %.4f +/- %.4f",
        effective_splits,
        imbalance_strategy,
        mean_macro_f1,
        std_macro_f1,
    )
    return {
        "executed": True,
        "variant": "tuned" if xgb_params else "baseline",
        "n_splits": int(effective_splits),
        "requested_n_splits": int(n_splits),
        "imbalance_strategy": imbalance_strategy,
        "dev_pool_rows": int(len(idx_dev)),
        "mean_macro_f1": mean_macro_f1,
        "std_macro_f1": std_macro_f1,
        "mean_weighted_f1": float(np.mean(weighted_f1s)),
        "std_weighted_f1": float(np.std(weighted_f1s, ddof=ddof)),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs, ddof=ddof)),
        "mean_roc_auc": float(np.mean(rocs)),
        "std_roc_auc": float(np.std(rocs, ddof=ddof)),
        "folds": fold_results,
    }
