"""End-to-end training orchestrator for MLP, 1D-CNN, Autoencoder, and XGBoost."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.config import AppConfig, AutoencoderConfig, ModelConfig, sanitize_machine_paths
from src.data.preprocessing import PreparedDataset, run_preprocessing_pipeline
from src.detection.anomaly_detection import (
    AnomalyFusionDetector,
    fit_and_select_anomaly_fusion,
    fit_classical_anomaly_baselines,
)
from src.evaluation.protocols import run_xgboost_stratified_cv
from src.models.autoencoder import (
    build_autoencoder,
    build_deep_regularized_autoencoder,
    build_vae,
    calibrate_threshold,
    compute_reconstruction_error,
    extract_benign_subset,
    find_benign_class_index,
)
from src.models.cnn import build_cnn1d
from src.models.ensemble import (
    StackingMetaLearner,
    TemperatureScaler,
    compute_ece,
    compute_multiclass_brier,
)
from src.models.mlp import build_mlp
from src.models.xgboost_model import expand_xgboost_probas, fit_xgboost_safe_multiclass
from src.training.callbacks import build_keras_callbacks, safe_save_keras_model
from src.training.optimization import (
    ImbalanceExperimentResult,
    XGBoostOptimizationResult,
    _eval_val_scores,
    compare_imbalance_per_model,
    compare_smote_vs_class_weight,
    optimize_xgboost_with_optuna,
    prepare_strategy_training_data,
)
from src.utils.logging import get_logger
from src.utils.model_registry import ModelRegistry, validate_xgboost_profile
from src.utils.reproducibility import HardwareProfile, configure_hardware, set_global_seed

logger = get_logger("smart_ids.training.trainer")


@dataclass
class TrainingOutputs:
    """Artifacts and summaries emitted after training all four models."""

    models: dict[str, Any]
    histories: dict[str, dict[str, list[float]]]
    training_durations_sec: dict[str, float]
    overfitting_analysis: dict[str, dict[str, float]]
    anomaly_threshold: float
    anomaly_stats: dict[str, float]
    imbalance_experiment: ImbalanceExperimentResult
    xgb_optimization: XGBoostOptimizationResult
    calibration_report: dict[str, dict[str, Any]]
    calibrators: dict[str, TemperatureScaler]
    hardware_profile: HardwareProfile
    stacker: StackingMetaLearner | None = None
    best_supervised_model: str = "XGBoost"
    imbalance_strategies_by_model: dict[str, str] = field(default_factory=dict)
    anomaly_fusion: AnomalyFusionDetector | None = None
    neural_loss_experiments: dict[str, Any] = field(default_factory=dict)


def _extract_history_dict(history_obj: Any) -> dict[str, list[float]]:
    raw = getattr(history_obj, "history", {}) or {}
    return {str(k): [float(v) for v in vals] for k, vals in raw.items()}


def _compute_overfitting_gap(hist: dict[str, list[float]]) -> dict[str, float]:
    if not hist or "val_loss" not in hist or "loss" not in hist:
        return {}
    best_epoch = int(np.argmin(hist["val_loss"]))
    gap: dict[str, float] = {
        "best_epoch": float(best_epoch + 1),
        "total_epochs_run": float(len(hist["loss"])),
        "best_train_loss": float(hist["loss"][best_epoch]),
        "best_val_loss": float(hist["val_loss"][best_epoch]),
        "loss_generalization_gap": float(hist["val_loss"][best_epoch] - hist["loss"][best_epoch]),
    }
    if "accuracy" in hist and "val_accuracy" in hist:
        gap["best_train_acc"] = float(hist["accuracy"][best_epoch])
        gap["best_val_acc"] = float(hist["val_accuracy"][best_epoch])
        gap["acc_generalization_gap"] = float(hist["accuracy"][best_epoch] - hist["val_accuracy"][best_epoch])
    return gap


def _fit_benign_reconstruction_variant(
    name: str,
    model: Any,
    ckpt_path: Path,
    X_benign_train: np.ndarray,
    X_benign_val: np.ndarray,
    epochs: int,
    ae_cfg: AutoencoderConfig,
) -> tuple[Any, dict[str, list[float]], float, float, dict[str, float]]:
    """Train a benign-only reconstruction anomaly model, save it, and calibrate its threshold."""
    cbs = build_keras_callbacks(
        checkpoint_path=ckpt_path,
        early_stopping_patience=ae_cfg.early_stopping_patience,
        reduce_lr_patience=ae_cfg.reduce_lr_patience,
        reduce_lr_factor=ae_cfg.reduce_lr_factor,
        min_lr=ae_cfg.min_lr,
    )
    t0 = time.perf_counter()
    hist_obj = model.fit(
        X_benign_train,
        X_benign_train,
        validation_data=(X_benign_val, X_benign_val),
        epochs=epochs,
        batch_size=ae_cfg.batch_size,
        callbacks=cbs,
        verbose=2,
    )
    dur = float(time.perf_counter() - t0)
    safe_save_keras_model(model, ckpt_path)
    hist = _extract_history_dict(hist_obj)
    val_errors = compute_reconstruction_error(model, X_benign_val)
    thresh, stats = calibrate_threshold(val_errors, percentile=ae_cfg.threshold_percentile)
    logger.info(
        "%s finished in %.2fs | Calibrated %.1fth pct threshold=%.6f",
        name,
        dur,
        ae_cfg.threshold_percentile,
        thresh,
    )
    return model, hist, dur, thresh, stats


def _run_bounded_neural_loss_experiment(
    model_name: str,
    base_model: Any,
    base_history: dict[str, list[float]],
    ckpt_path: Path,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_features: int,
    num_classes: int,
    class_weight: dict[int, float] | None,
    base_epochs: int,
    model_cfg: ModelConfig,
    deadline_perf: float,
    seed: int,
) -> tuple[Any, dict[str, list[float]], str, dict[str, Any]]:
    """Test bounded label smoothing & focal loss variants; keep ONLY if val Macro-F1 beats baseline + tol."""
    import keras

    is_cnn = model_name == "CNN"
    batch_size = model_cfg.cnn.batch_size if is_cnn else model_cfg.mlp.batch_size
    sub_cfg = model_cfg.cnn if is_cnn else model_cfg.mlp
    tol = float(model_cfg.neural.improvement_tolerance)

    base_probs = base_model.predict(X_val, batch_size=max(256, int(batch_size)), verbose=0)
    base_metrics = _eval_val_scores(y_val, base_probs, num_classes)
    base_f1 = float(base_metrics["val_macro_f1"])

    candidates: list[dict[str, Any]] = [
        {
            "loss_variant": "cross_entropy (baseline)",
            "status": "baseline",
            **base_metrics,
            "delta_val_macro_f1": 0.0,
            "elapsed_sec": 0.0,
        }
    ]
    best_model = base_model
    best_hist = base_history
    best_loss_name = "cross_entropy"
    best_f1 = base_f1

    class _TimeCapCallback(keras.callbacks.Callback):
        def __init__(self, deadline: float) -> None:
            super().__init__()
            self.deadline = deadline

        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            if time.perf_counter() >= self.deadline:
                self.model.stop_training = True

    trial_epochs = max(1, min(int(base_epochs), int(model_cfg.neural.epochs_cap)))
    for loss_variant in ("label_smoothing", "focal_loss"):
        if time.perf_counter() >= deadline_perf:
            candidates.append(
                {
                    "loss_variant": loss_variant,
                    "status": "skipped_time_budget",
                    "val_macro_f1": None,
                    "delta_val_macro_f1": None,
                    "elapsed_sec": None,
                }
            )
            continue

        set_global_seed(seed)
        cand_model = (
            build_cnn1d(
                num_features,
                num_classes,
                cfg=model_cfg.cnn,
                loss_type=loss_variant,
                label_smoothing=model_cfg.neural.label_smoothing,
                focal_gamma=model_cfg.neural.focal_gamma,
            )
            if is_cnn
            else build_mlp(
                num_features,
                num_classes,
                cfg=model_cfg.mlp,
                loss_type=loss_variant,
                label_smoothing=model_cfg.neural.label_smoothing,
                focal_gamma=model_cfg.neural.focal_gamma,
            )
        )
        cbs = [
            *build_keras_callbacks(
                checkpoint_path=None,
                early_stopping_patience=sub_cfg.early_stopping_patience,
                reduce_lr_patience=sub_cfg.reduce_lr_patience,
                reduce_lr_factor=sub_cfg.reduce_lr_factor,
                min_lr=sub_cfg.min_lr,
            ),
            _TimeCapCallback(deadline_perf),
        ]
        t0 = time.perf_counter()
        cand_hist_obj = cand_model.fit(
            X_fit,
            y_fit,
            validation_data=(X_val, y_val),
            epochs=trial_epochs,
            batch_size=batch_size,
            callbacks=cbs,
            class_weight=class_weight,
            verbose=0,
        )
        elapsed = float(time.perf_counter() - t0)
        cand_probs = cand_model.predict(X_val, batch_size=max(256, int(batch_size)), verbose=0)
        cand_metrics = _eval_val_scores(y_val, cand_probs, num_classes)
        cand_f1 = float(cand_metrics["val_macro_f1"])
        delta = cand_f1 - base_f1
        beats_tol = cand_f1 > (base_f1 + tol) and cand_f1 > best_f1

        candidates.append(
            {
                "loss_variant": loss_variant,
                "status": "kept" if beats_tol else "reverted",
                **cand_metrics,
                "delta_val_macro_f1": round(delta, 6),
                "elapsed_sec": round(elapsed, 2),
            }
        )
        if beats_tol:
            best_model = cand_model
            best_hist = _extract_history_dict(cand_hist_obj)
            best_loss_name = loss_variant
            best_f1 = cand_f1

    if best_loss_name != "cross_entropy":
        safe_save_keras_model(best_model, ckpt_path)
        decision = (
            f"Kept '{best_loss_name}' for {model_name} (val Macro-F1={best_f1:.4f} vs baseline "
            f"{base_f1:.4f}, delta={best_f1 - base_f1:+.4f} > tolerance={tol:.4f})."
        )
    else:
        decision = (
            f"Reverted to baseline 'cross_entropy' for {model_name} (baseline val Macro-F1={base_f1:.4f}; "
            f"no variant exceeded +{tol:.4f} tolerance within bounded neural budget)."
        )
    logger.info("Neural loss experiment [%s]: %s", model_name, decision)
    return (
        best_model,
        best_hist,
        best_loss_name,
        {
            "model": model_name,
            "kept_variant": best_loss_name,
            "baseline_val_macro_f1": round(base_f1, 6),
            "kept_val_macro_f1": round(best_f1, 6),
            "improvement_tolerance": tol,
            "decision": decision,
            "candidates": candidates,
        },
    )


def train_all_models(
    dataset: PreparedDataset,
    app_cfg: AppConfig,
    model_cfg: ModelConfig,
    epochs_override: int | None = None,
    optuna_trials_override: int | None = None,
) -> TrainingOutputs:
    """Train MLP, 1D-CNN, Autoencoder, and XGBoost with full diagnostics and checkpointing."""
    seed = app_cfg.project.random_seed
    set_global_seed(seed)
    hardware = configure_hardware(
        use_gpu=model_cfg.hardware.use_gpu,
        memory_growth=model_cfg.hardware.memory_growth,
        mixed_precision=model_cfg.hardware.mixed_precision,
    )

    models_dir = app_cfg.paths.resolve("models_dir")
    artifacts_dir = app_cfg.paths.resolve("artifacts_dir")
    registry = ModelRegistry(models_dir=models_dir, artifacts_dir=artifacts_dir)

    num_features = dataset.X_train_scaled.shape[1]
    num_classes = len(dataset.class_names)

    # 1. Experimentally select imbalance strategy (per model when configured, else global)
    tie_tol = (
        app_cfg.imbalance_selection.tie_tolerance
        if app_cfg.imbalance_selection.tie_tolerance is not None
        else app_cfg.data.imbalance.imbalance_tie_tolerance
    )
    if app_cfg.data.imbalance.strategy == "auto_compare":
        if app_cfg.imbalance_selection.per_model:
            imb_result = compare_imbalance_per_model(
                X_train_scaled=dataset.X_train_scaled,
                y_train=dataset.y_train,
                X_train_sm=dataset.X_train_sm,
                y_train_sm=dataset.y_train_sm,
                X_val_scaled=dataset.X_val_scaled,
                y_val=dataset.y_val,
                class_weights=dataset.class_weights,
                num_classes=num_classes,
                model_cfg=model_cfg,
                smote_class_weights=dataset.smote_class_weights,
                xgb_device=hardware.xgb_device,
                seed=seed,
                tie_tolerance=tie_tol,
                neural_epochs_cap=app_cfg.imbalance_selection.neural_epochs_cap,
            )
            set_global_seed(seed)
        else:
            imb_result = compare_smote_vs_class_weight(
                X_train_scaled=dataset.X_train_scaled,
                y_train=dataset.y_train,
                X_train_sm=dataset.X_train_sm,
                y_train_sm=dataset.y_train_sm,
                X_val_scaled=dataset.X_val_scaled,
                y_val=dataset.y_val,
                class_weights=dataset.class_weights,
                num_classes=num_classes,
                smote_class_weights=dataset.smote_class_weights,
                xgb_device=hardware.xgb_device,
                seed=seed,
                tie_tolerance=tie_tol,
            )
    else:
        explicit_imb = app_cfg.data.imbalance.strategy
        imb_result = ImbalanceExperimentResult(
            smote_metrics={},
            class_weight_metrics={},
            baseline_unweighted_metrics={},
            selected_strategy=explicit_imb,
            justification=f"Configured explicitly via data.imbalance.strategy={explicit_imb}.",
            selected_strategies_by_model={
                "MLP": explicit_imb,
                "CNN": explicit_imb,
                "XGBoost": explicit_imb,
            },
            selection_mode="explicit",
        )

    selected_strategies: dict[str, str] = {
        "MLP": imb_result.selected_strategies_by_model.get("MLP", imb_result.selected_strategy),
        "CNN": imb_result.selected_strategies_by_model.get("CNN", imb_result.selected_strategy),
        "XGBoost": imb_result.selected_strategies_by_model.get("XGBoost", imb_result.selected_strategy),
    }

    X_mlp_train, y_mlp_train, mlp_class_weight, _ = prepare_strategy_training_data(
        selected_strategies["MLP"],
        dataset.X_train_scaled,
        dataset.y_train,
        dataset.X_train_sm,
        dataset.y_train_sm,
        dataset.class_weights,
        dataset.smote_class_weights,
    )
    X_cnn_train, y_cnn_train, cnn_class_weight, _ = prepare_strategy_training_data(
        selected_strategies["CNN"],
        dataset.X_train_scaled,
        dataset.y_train,
        dataset.X_train_sm,
        dataset.y_train_sm,
        dataset.class_weights,
        dataset.smote_class_weights,
    )
    X_xgb_train, y_xgb_train, _, xgb_sample_weight = prepare_strategy_training_data(
        selected_strategies["XGBoost"],
        dataset.X_train_scaled,
        dataset.y_train,
        dataset.X_train_sm,
        dataset.y_train_sm,
        dataset.class_weights,
        dataset.smote_class_weights,
    )
    xgb_imb = selected_strategies["XGBoost"]

    histories: dict[str, dict[str, list[float]]] = {}
    durations: dict[str, float] = {}
    overfit_stats: dict[str, dict[str, float]] = {}
    trained_models: dict[str, Any] = {}

    # 2. Train MLP (512 -> 256 -> 128 -> 64)
    logger.info(
        "Training MLP classifier on %d samples (imbalance strategy='%s')...",
        len(X_mlp_train),
        selected_strategies["MLP"],
    )
    mlp_epochs = epochs_override or model_cfg.mlp.epochs
    mlp = build_mlp(num_features, num_classes, cfg=model_cfg.mlp)
    mlp_ckpt = models_dir / model_cfg.mlp.checkpoint_name
    mlp_callbacks = build_keras_callbacks(
        checkpoint_path=mlp_ckpt,
        early_stopping_patience=model_cfg.mlp.early_stopping_patience,
        reduce_lr_patience=model_cfg.mlp.reduce_lr_patience,
        reduce_lr_factor=model_cfg.mlp.reduce_lr_factor,
        min_lr=model_cfg.mlp.min_lr,
    )
    t0 = time.perf_counter()
    mlp_hist = mlp.fit(
        X_mlp_train,
        y_mlp_train,
        validation_data=(dataset.X_val_scaled, dataset.y_val),
        epochs=mlp_epochs,
        batch_size=model_cfg.mlp.batch_size,
        callbacks=mlp_callbacks,
        class_weight=mlp_class_weight,
        verbose=2,
    )
    durations["MLP"] = float(time.perf_counter() - t0)
    safe_save_keras_model(mlp, mlp_ckpt)
    trained_models["MLP"] = mlp
    histories["MLP"] = _extract_history_dict(mlp_hist)
    overfit_stats["MLP"] = _compute_overfitting_gap(histories["MLP"])
    logger.info("MLP finished in %.2fs | %s", durations["MLP"], overfit_stats["MLP"])

    # 3. Train 1D-CNN
    logger.info(
        "Training 1D-CNN classifier on %d samples (imbalance strategy='%s')...",
        len(X_cnn_train),
        selected_strategies["CNN"],
    )
    cnn_epochs = epochs_override or model_cfg.cnn.epochs
    cnn = build_cnn1d(num_features, num_classes, cfg=model_cfg.cnn)
    cnn_ckpt = models_dir / model_cfg.cnn.checkpoint_name
    cnn_callbacks = build_keras_callbacks(
        checkpoint_path=cnn_ckpt,
        early_stopping_patience=model_cfg.cnn.early_stopping_patience,
        reduce_lr_patience=model_cfg.cnn.reduce_lr_patience,
        reduce_lr_factor=model_cfg.cnn.reduce_lr_factor,
        min_lr=model_cfg.cnn.min_lr,
    )
    X_train_cnn = X_cnn_train.reshape(-1, num_features, 1)
    X_val_cnn = dataset.X_val_scaled.reshape(-1, num_features, 1)
    t0 = time.perf_counter()
    cnn_hist = cnn.fit(
        X_train_cnn,
        y_cnn_train,
        validation_data=(X_val_cnn, dataset.y_val),
        epochs=cnn_epochs,
        batch_size=model_cfg.cnn.batch_size,
        callbacks=cnn_callbacks,
        class_weight=cnn_class_weight,
        verbose=2,
    )
    durations["CNN"] = float(time.perf_counter() - t0)
    safe_save_keras_model(cnn, cnn_ckpt)
    trained_models["CNN"] = cnn
    histories["CNN"] = _extract_history_dict(cnn_hist)
    overfit_stats["CNN"] = _compute_overfitting_gap(histories["CNN"])
    logger.info("1D-CNN finished in %.2fs | %s", durations["CNN"], overfit_stats["CNN"])

    neural_loss_experiments: dict[str, Any] = {}
    neural_losses_selected: dict[str, str] = {"MLP": "cross_entropy", "CNN": "cross_entropy"}
    if model_cfg.neural.try_loss_variants and len(X_mlp_train) >= 10:
        neural_deadline = time.perf_counter() + float(model_cfg.neural.max_training_minutes) * 60.0
        mlp, histories["MLP"], mlp_loss_name, mlp_loss_report = _run_bounded_neural_loss_experiment(
            model_name="MLP",
            base_model=mlp,
            base_history=histories["MLP"],
            ckpt_path=mlp_ckpt,
            X_fit=X_mlp_train,
            y_fit=y_mlp_train,
            X_val=dataset.X_val_scaled,
            y_val=dataset.y_val,
            num_features=num_features,
            num_classes=num_classes,
            class_weight=mlp_class_weight,
            base_epochs=mlp_epochs,
            model_cfg=model_cfg,
            deadline_perf=neural_deadline,
            seed=seed,
        )
        trained_models["MLP"] = mlp
        overfit_stats["MLP"] = _compute_overfitting_gap(histories["MLP"])
        neural_losses_selected["MLP"] = mlp_loss_name
        neural_loss_experiments["MLP"] = mlp_loss_report

        cnn, histories["CNN"], cnn_loss_name, cnn_loss_report = _run_bounded_neural_loss_experiment(
            model_name="CNN",
            base_model=cnn,
            base_history=histories["CNN"],
            ckpt_path=cnn_ckpt,
            X_fit=X_train_cnn,
            y_fit=y_cnn_train,
            X_val=X_val_cnn,
            y_val=dataset.y_val,
            num_features=num_features,
            num_classes=num_classes,
            class_weight=cnn_class_weight,
            base_epochs=cnn_epochs,
            model_cfg=model_cfg,
            deadline_perf=neural_deadline,
            seed=seed,
        )
        trained_models["CNN"] = cnn
        overfit_stats["CNN"] = _compute_overfitting_gap(histories["CNN"])
        neural_losses_selected["CNN"] = cnn_loss_name
        neural_loss_experiments["CNN"] = cnn_loss_report
        set_global_seed(seed)

    # 4. Train Unsupervised Autoencoder STRICTLY on BENIGN samples
    X_benign_train, benign_idx, benign_label = extract_benign_subset(
        dataset.X_train_scaled, dataset.y_train, dataset.label_encoder
    )
    X_benign_val, _, _ = extract_benign_subset(dataset.X_val_scaled, dataset.y_val, dataset.label_encoder)
    logger.info(
        "Training Autoencoder exclusively on '%s' (idx=%d): %d train / %d val samples",
        benign_label,
        benign_idx,
        len(X_benign_train),
        len(X_benign_val),
    )
    ae_epochs = epochs_override or model_cfg.autoencoder.epochs
    autoencoder, _, _ = build_autoencoder(num_features, cfg=model_cfg.autoencoder)
    ae_ckpt = models_dir / model_cfg.autoencoder.checkpoint_name
    ae_callbacks = build_keras_callbacks(
        checkpoint_path=ae_ckpt,
        early_stopping_patience=model_cfg.autoencoder.early_stopping_patience,
        reduce_lr_patience=model_cfg.autoencoder.reduce_lr_patience,
        reduce_lr_factor=model_cfg.autoencoder.reduce_lr_factor,
        min_lr=model_cfg.autoencoder.min_lr,
    )
    t0 = time.perf_counter()
    ae_hist = autoencoder.fit(
        X_benign_train,
        X_benign_train,
        validation_data=(X_benign_val, X_benign_val),
        epochs=ae_epochs,
        batch_size=model_cfg.autoencoder.batch_size,
        callbacks=ae_callbacks,
        verbose=2,
    )
    durations["Autoencoder"] = float(time.perf_counter() - t0)
    safe_save_keras_model(autoencoder, ae_ckpt)
    trained_models["Autoencoder"] = autoencoder
    histories["Autoencoder"] = _extract_history_dict(ae_hist)
    overfit_stats["Autoencoder"] = _compute_overfitting_gap(histories["Autoencoder"])

    benign_val_errors = compute_reconstruction_error(autoencoder, X_benign_val)
    anomaly_threshold, anomaly_stats = calibrate_threshold(
        benign_val_errors, percentile=model_cfg.autoencoder.threshold_percentile
    )
    registry.save_anomaly_threshold(
        threshold=anomaly_threshold,
        percentile=model_cfg.autoencoder.threshold_percentile,
        benign_errors_summary=anomaly_stats,
    )
    logger.info(
        "Autoencoder finished in %.2fs | Calibrated %.1fth pct threshold=%.6f",
        durations["Autoencoder"],
        model_cfg.autoencoder.threshold_percentile,
        anomaly_threshold,
    )

    anomaly_baselines: dict[str, Any] = {}
    if model_cfg.anomaly_baselines.enabled:
        anomaly_baselines = fit_classical_anomaly_baselines(
            X_benign_train=X_benign_train,
            X_benign_val=X_benign_val,
            cfg=model_cfg.anomaly_baselines,
            percentile=model_cfg.autoencoder.threshold_percentile,
            seed=seed,
        )
        for b_name, base_obj in anomaly_baselines.items():
            durations[b_name] = float(base_obj.fit_duration_sec)
        joblib.dump(anomaly_baselines, models_dir / model_cfg.anomaly_baselines.baselines_file)

    anomaly_variants_meta: dict[str, Any] = {}
    if model_cfg.autoencoder.try_variants and len(X_benign_train) >= 10 and len(X_benign_val) >= 5:
        var_epochs = max(1, min(int(ae_epochs), int(model_cfg.autoencoder.variant_epochs_cap)))
        deep_ae_raw = build_deep_regularized_autoencoder(num_features, cfg=model_cfg.autoencoder)
        deep_ae, h_deep, d_deep, t_deep, st_deep = _fit_benign_reconstruction_variant(
            "Autoencoder (Deep Reg)",
            deep_ae_raw,
            models_dir / model_cfg.autoencoder.deep_reg_checkpoint_name,
            X_benign_train,
            X_benign_val,
            epochs=var_epochs,
            ae_cfg=model_cfg.autoencoder,
        )
        trained_models["Autoencoder (Deep Reg)"] = deep_ae
        histories["Autoencoder (Deep Reg)"] = h_deep
        durations["Autoencoder (Deep Reg)"] = d_deep
        overfit_stats["Autoencoder (Deep Reg)"] = _compute_overfitting_gap(h_deep)
        anomaly_variants_meta["Autoencoder (Deep Reg)"] = {
            "threshold": float(t_deep),
            "stats": st_deep,
            "epochs_cap": var_epochs,
        }

        vae_raw = build_vae(num_features, cfg=model_cfg.autoencoder)
        vae_model, h_vae, d_vae, t_vae, st_vae = _fit_benign_reconstruction_variant(
            "VAE",
            vae_raw,
            models_dir / model_cfg.autoencoder.vae_checkpoint_name,
            X_benign_train,
            X_benign_val,
            epochs=var_epochs,
            ae_cfg=model_cfg.autoencoder,
        )
        trained_models["VAE"] = vae_model
        histories["VAE"] = h_vae
        durations["VAE"] = d_vae
        overfit_stats["VAE"] = _compute_overfitting_gap(h_vae)
        anomaly_variants_meta["VAE"] = {
            "threshold": float(t_vae),
            "stats": st_vae,
            "epochs_cap": var_epochs,
        }

    anomaly_fusion: AnomalyFusionDetector | None = None
    anomaly_validation_selection: dict[str, Any] = {}
    if model_cfg.autoencoder.fusion_enabled and len(dataset.X_val_scaled) > 0:
        val_anom_scores: dict[str, np.ndarray] = {
            "Autoencoder": compute_reconstruction_error(autoencoder, dataset.X_val_scaled)
        }
        for v_name in ("Autoencoder (Deep Reg)", "VAE"):
            if v_name in trained_models:
                val_anom_scores[v_name] = compute_reconstruction_error(
                    trained_models[v_name], dataset.X_val_scaled
                )
        for b_name, base_obj in anomaly_baselines.items():
            val_anom_scores[b_name] = base_obj.score(dataset.X_val_scaled)

        anomaly_fusion = fit_and_select_anomaly_fusion(
            val_scores_by_detector=val_anom_scores,
            y_val=dataset.y_val,
            benign_index=benign_idx,
            percentile=model_cfg.autoencoder.threshold_percentile,
            grid_step=model_cfg.autoencoder.fusion_grid_step,
        )
        joblib.dump(anomaly_fusion, models_dir / model_cfg.autoencoder.fusion_file)
        anomaly_validation_selection = {
            "ae_component_selected": anomaly_fusion.ae_component_name,
            "fusion_weights": anomaly_fusion.weights,
            "fusion_threshold": float(anomaly_fusion.threshold),
            "best_single_validation_detector": anomaly_fusion.best_single_validation_detector,
            "best_overall_validation_detector": anomaly_fusion.best_overall_validation_detector,
            "selection_table": anomaly_fusion.validation_selection_summary,
        }

    # 5. Train Baseline + Optuna-Tuned XGBoost
    xgb_cfg = model_cfg.xgboost.model_copy(deep=True)
    if optuna_trials_override is not None:
        xgb_cfg.optuna.n_trials = optuna_trials_override
        xgb_cfg.optuna.enabled = optuna_trials_override > 0

    smote_kw = {
        "smote_max_k_neighbors": app_cfg.data.imbalance.smote_max_k_neighbors,
        "smote_sampling_strategy": app_cfg.data.imbalance.smote_sampling_strategy,
        "smote_cap_per_class": app_cfg.data.imbalance.smote_cap_per_class,
        "smote_max_multiplier": app_cfg.data.imbalance.smote_max_multiplier,
        "smote_min_real_samples": app_cfg.data.imbalance.smote_min_real_samples,
    }
    unified_cv_mean: float | None = None
    unified_cv_std: float | None = None
    unified_cv_note = ""
    if (
        xgb_cfg.optuna.enabled
        and xgb_cfg.optuna.n_trials > 0
        and app_cfg.data.split_strategy in {"random", "time_block"}
        and app_cfg.evaluation.cv_n_splits >= 2
    ):
        baseline_cv_res = run_xgboost_stratified_cv(
            app_cfg=app_cfg,
            model_cfg=model_cfg,
            xgb_params=None,
            imbalance_strategy=xgb_imb,
            n_splits=app_cfg.evaluation.cv_n_splits,
            force_synthetic=dataset.source_info.is_synthetic,
        )
        set_global_seed(seed)
        if baseline_cv_res.get("executed"):
            unified_cv_mean = float(baseline_cv_res["mean_macro_f1"])
            unified_cv_std = float(baseline_cv_res["std_macro_f1"])
            unified_cv_note = (
                f"Unified {baseline_cv_res['n_splits']}-fold StratifiedKFold on the development pool "
                f"({baseline_cv_res['dev_pool_rows']:,} rows = train+val, held-out test excluded) with "
                f"per-fold preprocessing and '{xgb_imb}' imbalance handling (sample ddof=1 std across folds)."
            )

    t0 = time.perf_counter()
    xgb_opt = optimize_xgboost_with_optuna(
        X_train=X_xgb_train,
        y_train=y_xgb_train,
        X_val=dataset.X_val_scaled,
        y_val=dataset.y_val,
        num_classes=num_classes,
        cfg=xgb_cfg,
        xgb_device=hardware.xgb_device,
        seed=seed,
        sample_weight=xgb_sample_weight,
        cv_macro_f1_std=unified_cv_std,
        cv_macro_f1_mean=unified_cv_mean,
        cv_method_note=unified_cv_note,
        cv_raw_X=dataset.X_train_scaled,
        cv_raw_y=dataset.y_train,
        cv_n_splits=app_cfg.evaluation.cv_n_splits,
        use_cv_class_weight=(xgb_imb == "class_weight"),
        cv_imbalance_strategy=xgb_imb,
        smote_kwargs=smote_kw,
    )
    durations["XGBoost"] = float(time.perf_counter() - t0)
    xgb_model = xgb_opt.best_model
    xgb_model.save_model(str(models_dir / xgb_cfg.model_name))
    trained_models["XGBoost"] = xgb_model

    # 6. Probability calibration analysis & Temperature Scaling fit strictly on validation probabilities
    xgb_present_cls = np.unique(y_xgb_train).astype(np.int64)
    val_probas = {
        "MLP": mlp.predict(dataset.X_val_scaled, batch_size=1024, verbose=0),
        "CNN": cnn.predict(X_val_cnn, batch_size=1024, verbose=0),
        "XGBoost": expand_xgboost_probas(
            xgb_model.predict_proba(dataset.X_val_scaled), xgb_present_cls, num_classes
        ),
    }
    calibration_report: dict[str, dict[str, Any]] = {}
    calibrators: dict[str, TemperatureScaler] = {}
    for m_name, p_val in val_probas.items():
        ece_before = compute_ece(dataset.y_val, p_val)
        brier_before = compute_multiclass_brier(dataset.y_val, p_val)
        scaler_t = TemperatureScaler().fit(p_val, dataset.y_val)
        p_cal = scaler_t.transform(p_val)
        ece_after = compute_ece(dataset.y_val, p_cal)
        brier_after = compute_multiclass_brier(dataset.y_val, p_cal)
        if ece_after <= ece_before or brier_after < brier_before:
            calibrators[m_name] = scaler_t
        if abs(scaler_t.temperature - 1.0) <= 0.12 and (ece_before - ece_after) < 5e-4:
            cal_status = "Already well-calibrated near T=1.0"
        else:
            cal_status = (
                f"Recalibrated (T={scaler_t.temperature:.4f}, ECE delta={ece_after - ece_before:+.6f})"
            )
        calibration_report[m_name] = {
            "temperature": round(scaler_t.temperature, 4),
            "ece_before": round(ece_before, 6),
            "ece_after": round(ece_after, 6),
            "ece_delta": round(ece_after - ece_before, 6),
            "brier_before": round(brier_before, 6),
            "brier_after": round(brier_after, 6),
            "nll_before": round(scaler_t.nll_before, 6),
            "nll_after": round(scaler_t.nll_after, 6),
            "status": cal_status,
        }

    # Fit validation-only LogisticRegression stacking & logloss weights meta-learner
    stacker = StackingMetaLearner().fit(
        val_probas=val_probas,
        y_val=dataset.y_val,
        n_classes=num_classes,
        calibrators=calibrators,
        seed=seed,
    )
    best_sup_model = max(
        stacker.val_macro_f1.keys(),
        key=lambda k: stacker.val_macro_f1[k],
        default="XGBoost",
    )

    # Save training histories, calibrators, stacker, per-model metadata & optimization summary
    joblib.dump(calibrators, models_dir / "calibrators.pkl")
    joblib.dump(stacker, models_dir / model_cfg.ensemble.stacker_file)
    with (models_dir / "training_histories.json").open("w", encoding="utf-8") as fh:
        json.dump(histories, fh, indent=2)

    model_metadata: dict[str, dict[str, Any]] = {
        "MLP": {
            "imbalance_strategy": selected_strategies["MLP"],
            "loss_function": neural_losses_selected["MLP"],
            "train_samples": int(len(X_mlp_train)),
        },
        "CNN": {
            "imbalance_strategy": selected_strategies["CNN"],
            "loss_function": neural_losses_selected["CNN"],
            "train_samples": int(len(X_cnn_train)),
        },
        "XGBoost": {
            "imbalance_strategy": selected_strategies["XGBoost"],
            "active_profile": selected_strategies["XGBoost"],
            "profile_files": {selected_strategies["XGBoost"]: xgb_cfg.model_name},
            "available_profiles": {selected_strategies["XGBoost"]: xgb_cfg.model_name},
            "train_samples": int(len(X_xgb_train)),
            "selected_variant": xgb_opt.selected_variant,
        },
        "Autoencoder": {"imbalance_strategy": "benign_only", "train_samples": int(len(X_benign_train))},
    }
    registry.save_model_metadata(model_metadata)
    registry.save_artifact_manifest()

    diagnostics_payload = sanitize_machine_paths(
        {
            "training_durations_sec": durations,
            "overfitting_analysis": overfit_stats,
            "imbalance_experiment": imb_result.to_dict(),
            "imbalance_strategies_by_model": selected_strategies,
            "neural_loss_experiments": neural_loss_experiments,
            "model_metadata": model_metadata,
            "xgboost_optimization": xgb_opt.to_summary_dict(),
            "calibration_report": calibration_report,
            "stacking_summary": stacker.to_summary_dict(),
            "best_supervised_model": best_sup_model,
            "anomaly_baselines_meta": {
                k: {"threshold": float(v.threshold), "train_samples": int(v.train_samples_used)}
                for k, v in anomaly_baselines.items()
            },
            "anomaly_variants_meta": anomaly_variants_meta,
            "anomaly_validation_selection": anomaly_validation_selection,
            "hardware": {
                "tf_device": hardware.tf_device,
                "xgb_device": hardware.xgb_device,
                "cuda_driver_detected": hardware.cuda_driver_detected,
                "notes": hardware.notes,
            },
        }
    )
    with (artifacts_dir / "training_diagnostics.json").open("w", encoding="utf-8") as fh:
        json.dump(diagnostics_payload, fh, indent=2)

    return TrainingOutputs(
        models=trained_models,
        histories=histories,
        training_durations_sec=durations,
        overfitting_analysis=overfit_stats,
        anomaly_threshold=anomaly_threshold,
        anomaly_stats=anomaly_stats,
        imbalance_experiment=imb_result,
        xgb_optimization=xgb_opt,
        calibration_report=calibration_report,
        calibrators=calibrators,
        hardware_profile=hardware,
        stacker=stacker,
        best_supervised_model=best_sup_model,
        imbalance_strategies_by_model=selected_strategies,
        anomaly_fusion=anomaly_fusion,
        neural_loss_experiments=neural_loss_experiments,
    )


def train_xgboost_only_profiles(
    app_cfg: AppConfig,
    model_cfg: ModelConfig,
    profiles: list[str] | tuple[str, ...] = ("none", "class_weight"),
    optuna_trials_override: int | None = None,
    force_synthetic: bool = False,
) -> dict[str, Any]:
    """Train only XGBoost across deployment profiles on the identical split and seed.

    Preserves all existing MLP, 1D-CNN, Autoencoder, VAE, anomaly baseline, and scaler artifacts.
    """
    seed = int(app_cfg.project.random_seed)
    set_global_seed(seed)
    hardware = configure_hardware(
        use_gpu=model_cfg.hardware.use_gpu,
        memory_growth=model_cfg.hardware.memory_growth,
        mixed_precision=model_cfg.hardware.mixed_precision,
    )

    active_profile = validate_xgboost_profile(app_cfg.deployment.xgboost_imbalance_profile)
    ordered_profiles: list[str] = []
    for p in [*profiles, active_profile]:
        clean = validate_xgboost_profile(p)
        if clean not in ordered_profiles:
            ordered_profiles.append(clean)

    proc_dir = app_cfg.paths.resolve("processed_data_dir")
    models_dir = app_cfg.paths.resolve("models_dir")
    artifacts_dir = app_cfg.paths.resolve("artifacts_dir")
    registry = ModelRegistry(models_dir=models_dir, artifacts_dir=artifacts_dir)

    npz_path = proc_dir / "preprocessed_splits.npz"
    le_path = models_dir / "label_encoder.pkl"
    feat_path = models_dir / "feature_names.json"
    if npz_path.exists() and le_path.exists() and not force_synthetic:
        splits = np.load(npz_path)
        X_train_scaled = splits["X_train_scaled"]
        y_train = splits["y_train"]
        X_val_scaled = splits["X_val_scaled"]
        y_val = splits["y_val"]
        manifest_sha = registry._load_sha256_manifest() or {}
        registry.verify_artifact_file(le_path, manifest_payload=manifest_sha if manifest_sha else None)
        label_encoder = joblib.load(le_path)
        class_names = [str(c) for c in label_encoder.classes_]
        feature_names: list[str] = []
        if feat_path.exists():
            with feat_path.open("r", encoding="utf-8") as fh:
                feature_names = json.load(fh)
        audit_path = proc_dir / "preprocessing_audit.json"
        audit_report: dict[str, Any] = {}
        if audit_path.exists():
            with audit_path.open("r", encoding="utf-8") as fh:
                audit_report = json.load(fh)
        unique_y = np.unique(y_train).astype(np.int64)
        weights_arr = compute_class_weight(class_weight="balanced", classes=unique_y, y=y_train)
        class_weights = {int(cls): float(w) for cls, w in zip(unique_y, weights_arr, strict=False)}
    else:
        dataset = run_preprocessing_pipeline(app_cfg, force_synthetic=force_synthetic, save_outputs=True)
        X_train_scaled = dataset.X_train_scaled
        y_train = dataset.y_train
        X_val_scaled = dataset.X_val_scaled
        y_val = dataset.y_val
        label_encoder = dataset.label_encoder
        class_names = dataset.class_names
        feature_names = dataset.feature_names
        class_weights = dataset.class_weights
        audit_report = dataset.audit_report

    num_classes = len(class_names)
    benign_idx, _ = find_benign_class_index(label_encoder)
    xgb_cfg = model_cfg.xgboost.model_copy(deep=True)
    if optuna_trials_override is not None:
        xgb_cfg.optuna.n_trials = int(optuna_trials_override)
        xgb_cfg.optuna.enabled = int(optuna_trials_override) > 0
    else:
        xgb_cfg.optuna.enabled = False
        xgb_cfg.optuna.n_trials = 0

    meta_path = models_dir / "model_metadata.json"
    existing_model_metadata: dict[str, dict[str, Any]] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            existing_model_metadata = dict((json.load(fh).get("models", {})) or {})

    profile_files: dict[str, str] = {}
    profile_summaries: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Any] = {}
    total_xgb_duration = 0.0

    for prof in ordered_profiles:
        set_global_seed(seed)
        sample_w = (
            np.array([class_weights[int(c)] for c in y_train], dtype=np.float32)
            if prof == "class_weight"
            else None
        )
        t0 = time.perf_counter()
        if xgb_cfg.optuna.enabled and xgb_cfg.optuna.n_trials > 0:
            xgb_opt = optimize_xgboost_with_optuna(
                X_train=X_train_scaled,
                y_train=y_train,
                X_val=X_val_scaled,
                y_val=y_val,
                num_classes=num_classes,
                cfg=xgb_cfg,
                xgb_device=hardware.xgb_device,
                seed=seed,
                sample_weight=sample_w,
            )
            clf = xgb_opt.best_model
            present_cls = np.unique(y_train).astype(np.int64)
            variant_used = xgb_opt.selected_variant
        else:
            clf, present_cls = fit_xgboost_safe_multiclass(
                X_train=X_train_scaled,
                y_train=y_train,
                X_val=X_val_scaled,
                y_val=y_val,
                total_classes=num_classes,
                cfg=xgb_cfg,
                device=hardware.xgb_device,
                seed=seed,
                sample_weight=sample_w,
            )
            variant_used = "baseline"
        elapsed_prof = float(time.perf_counter() - t0)
        total_xgb_duration += elapsed_prof

        val_probs = expand_xgboost_probas(clf.predict_proba(X_val_scaled), present_cls, num_classes)
        val_scores = _eval_val_scores(y_val, val_probs, num_classes)
        val_preds = np.argmax(val_probs, axis=1)
        val_ben_mask = y_val == benign_idx
        val_ben_fpr = float(np.mean(val_preds[val_ben_mask] != benign_idx)) if np.any(val_ben_mask) else 0.0

        fname = xgb_cfg.model_name if prof == active_profile else f"xgb_model_{prof}.json"
        out_path = models_dir / fname
        clf.save_model(str(out_path))
        profile_files[prof] = fname
        trained_models[prof] = clf
        profile_summaries[prof] = {
            "profile": prof,
            "artifact_file": fname,
            "is_active": prof == active_profile,
            "selected_variant": variant_used,
            "train_samples": int(len(X_train_scaled)),
            "train_duration_sec": round(elapsed_prof, 2),
            "val_macro_f1": round(float(val_scores["val_macro_f1"]), 6),
            "val_weighted_f1": round(float(val_scores["val_weighted_f1"]), 6),
            "val_macro_recall": round(float(val_scores["val_macro_recall"]), 6),
            "val_mlogloss": round(float(val_scores["val_mlogloss"]), 6),
            "val_benign_fpr": round(val_ben_fpr, 6),
        }
        logger.info(
            "Trained XGBoost profile '%s' -> %s in %.2fs | val Macro-F1=%.4f | val Benign FPR=%.4f",
            prof,
            out_path.name,
            elapsed_prof,
            val_scores["val_macro_f1"],
            val_ben_fpr,
        )

    existing_model_metadata["XGBoost"] = {
        "imbalance_strategy": active_profile,
        "active_profile": active_profile,
        "profile_files": dict(profile_files),
        "available_profiles": dict(profile_files),
        "train_samples": int(len(X_train_scaled)),
        "selected_variant": profile_summaries[active_profile]["selected_variant"],
        "profiles": profile_summaries,
    }
    registry.save_model_metadata(existing_model_metadata)
    registry.save_artifact_manifest()

    diag_path = artifacts_dir / "training_diagnostics.json"
    diagnostics_payload: dict[str, Any] = {}
    if diag_path.exists():
        with diag_path.open("r", encoding="utf-8") as fh:
            diagnostics_payload = json.load(fh)
    durations_map = dict(diagnostics_payload.get("training_durations_sec", {}))
    durations_map["XGBoost"] = float(profile_summaries[active_profile]["train_duration_sec"])
    diagnostics_payload["training_durations_sec"] = durations_map
    imb_map = dict(diagnostics_payload.get("imbalance_strategies_by_model", {}))
    imb_map["XGBoost"] = active_profile
    diagnostics_payload["imbalance_strategies_by_model"] = imb_map
    diagnostics_payload["model_metadata"] = existing_model_metadata
    diagnostics_payload["xgboost_profiles"] = profile_summaries
    diagnostics_payload = sanitize_machine_paths(diagnostics_payload)
    with diag_path.open("w", encoding="utf-8") as fh:
        json.dump(diagnostics_payload, fh, indent=2)

    return {
        "active_profile": active_profile,
        "profile_files": profile_files,
        "profiles": profile_summaries,
        "models": trained_models,
        "class_names": class_names,
        "feature_names": feature_names,
        "total_duration_sec": round(total_xgb_duration, 2),
        "audit_split": audit_report.get("split_strategy"),
    }
