"""Tests for all four model architectures, saving/loading, ensemble, and inference."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    AnomalyBaselinesConfig,
    AppConfig,
    EnsembleConfig,
    ModelConfig,
    XGBoostConfig,
    XGBoostOptunaConfig,
)
from src.data.loader import generate_synthetic_cicids2017
from src.data.preprocessing import drop_features_from_prepared_dataset, run_preprocessing_pipeline
from src.detection.anomaly_detection import (
    AnomalyDetector,
    fit_and_select_anomaly_fusion,
    fit_classical_anomaly_baselines,
)
from src.detection.inference import UnifiedDetectionEngine
from src.evaluation.metrics import (
    bootstrap_macro_f1_ci,
    evaluate_anomaly_detector,
    evaluate_supervised_model,
    paired_bootstrap_macro_f1_test,
    wilson_score_interval,
)
from src.evaluation.protocols import (
    analyze_time_block_failures,
    assert_ablation_row_parity,
    evaluate_by_day_protocol,
    evaluate_feature_ablation,
    evaluate_mitigation_tradeoffs,
    evaluate_time_block_protocol,
    run_xgboost_stratified_cv,
    select_xgboost_profile_time_aware,
)
from src.evaluation.report_writer import format_ensemble_empirical_conclusion, write_academic_reports
from src.models.autoencoder import (
    build_autoencoder,
    build_deep_regularized_autoencoder,
    build_vae,
    calibrate_threshold,
    compute_reconstruction_error,
    extract_benign_subset,
)
from src.models.cnn import build_cnn1d
from src.models.ensemble import (
    HybridEnsembleEngine,
    StackingMetaLearner,
    TemperatureScaler,
    compute_ece,
)
from src.models.mlp import build_mlp
from src.models.xgboost_model import build_xgboost_classifier
from src.response.response_engine import ActiveResponseEngine
from src.training.optimization import (
    compare_imbalance_per_model,
    compare_smote_vs_class_weight,
    optimize_xgboost_with_optuna,
    select_imbalance_strategy,
)
from src.training.trainer import (
    refit_xgboost_downstream_artifacts,
    train_all_models,
    train_xgboost_only_profiles,
)
from src.utils.model_registry import ModelRegistry


def test_all_four_models_build_train_save_reload_and_infer(
    tmp_app_config: AppConfig, small_synthetic_df: pd.DataFrame, tmp_path: Path
) -> None:
    dataset = run_preprocessing_pipeline(tmp_app_config, force_synthetic=True, save_outputs=True)
    n_feat = dataset.X_train_scaled.shape[1]
    n_cls = len(dataset.class_names)
    models_dir = Path(tmp_app_config.paths.models_dir)

    # 1. MLP
    mlp = build_mlp(n_feat, n_cls)
    mlp.fit(dataset.X_train_sm[:128], dataset.y_train_sm[:128], epochs=1, batch_size=64, verbose=0)
    mlp.save(models_dir / "best_mlp.keras")

    # 2. 1D-CNN
    cnn = build_cnn1d(n_feat, n_cls)
    cnn.fit(
        dataset.X_train_sm[:128].reshape(-1, n_feat, 1),
        dataset.y_train_sm[:128],
        epochs=1,
        batch_size=64,
        verbose=0,
    )
    cnn.save(models_dir / "best_cnn.keras")

    # 3. Autoencoder trained strictly on BENIGN rows
    X_ben_tr, benign_idx, benign_name = extract_benign_subset(
        dataset.X_train_scaled, dataset.y_train, dataset.label_encoder
    )
    X_ben_val, _, _ = extract_benign_subset(dataset.X_val_scaled, dataset.y_val, dataset.label_encoder)
    assert benign_name == "BENIGN"
    assert len(X_ben_tr) < len(dataset.X_train_scaled)

    ae, _, _ = build_autoencoder(n_feat)
    ae.fit(X_ben_tr[:128], X_ben_tr[:128], epochs=1, batch_size=64, verbose=0)
    ae.save(models_dir / "best_autoencoder.keras")

    val_err = compute_reconstruction_error(ae, X_ben_val)
    thresh, stats = calibrate_threshold(val_err, percentile=95.0)
    assert thresh > 0.0
    assert stats["p99"] >= stats["p95"] >= stats["p90"]

    # 4. XGBoost
    xgb_clf = build_xgboost_classifier(
        n_cls,
        params_override={"n_estimators": 10, "max_depth": 3},
        device="cpu",
        early_stopping_rounds=None,
    )
    xgb_clf.fit(dataset.X_train_sm, dataset.y_train_sm)
    xgb_clf.save_model(str(models_dir / "xgb_model.json"))

    registry = ModelRegistry(models_dir=models_dir, artifacts_dir=Path(tmp_app_config.paths.artifacts_dir))
    registry.save_anomaly_threshold(thresh, 95.0, stats)

    # Independent reload from disk
    loaded = registry.load_all(load_keras=True)
    assert set(loaded.models.keys()) == {"MLP", "CNN", "Autoencoder", "XGBoost"}
    assert loaded.scaler is not None
    assert loaded.label_encoder is not None
    assert len(loaded.feature_names) == 77

    # Unified Detection Engine batch inference on raw CSV rows
    resp_eng = ActiveResponseEngine(
        dry_run=True,
        log_file=tmp_path / "test_response.log",
        history_file=tmp_path / "test_history.json",
    )
    engine = UnifiedDetectionEngine(loaded, response_engine=resp_eng)
    sample_raw = small_synthetic_df.head(6).copy()
    out_df, preds = engine.predict_batch(sample_raw, model_choice="Hybrid Ensemble", explain_top_k=5)
    assert len(out_df) == 6
    assert len(preds) == 6
    assert set(
        ["attack_class", "confidence", "anomaly_score", "is_anomaly", "threat_level", "action"]
    ).issubset(out_df.columns)
    assert len(preds[0].important_contributing_features) == 5
    assert abs(sum(preds[0].class_probabilities.values()) - 1.0) < 1e-2


def test_hybrid_ensemble_zero_day_override_and_missing_artifacts(tmp_path: Path) -> None:
    # Graceful load when models directory is empty
    empty_reg = ModelRegistry(models_dir=tmp_path / "empty_models")
    loaded = empty_reg.load_all(load_keras=True)
    assert loaded.models == {}

    # AnomalyDetector graceful fallback when AE is None
    anom_none = AnomalyDetector(None, threshold=2.0).score(np.zeros((3, 5), dtype=np.float32))
    assert np.all(~anom_none.is_anomaly)

    # HybridEnsembleEngine keeps AE separate and flags Zero-Day Anomaly when supervised predicts BENIGN
    classes = ["BENIGN", "DDoS", "PortScan"]
    ens = HybridEnsembleEngine(class_names=classes, benign_index=0)
    probas = {
        "XGBoost": np.array([[0.92, 0.05, 0.03], [0.05, 0.90, 0.05]]),
        "MLP": np.array([[0.88, 0.07, 0.05], [0.08, 0.85, 0.07]]),
    }
    ae_scores = np.array([4.5, 0.2])  # First row is strong structural anomaly despite BENIGN supervised pred
    decision = ens.decide(
        probas, anomaly_scores=ae_scores, anomaly_threshold=1.0, strategy="hybrid_anomaly_override"
    )
    assert decision.predicted_labels[0] == "Zero-Day Anomaly (Novel)"
    assert bool(decision.anomaly_override_flags[0]) is True
    assert decision.predicted_labels[1] == "DDoS"
    assert bool(decision.anomaly_override_flags[1]) is False


def test_by_day_protocol_feature_ablation_and_report_tables(tmp_app_config: AppConfig) -> None:
    """Secondary protocols run without class-mismatch crashes and emit separate report tables."""
    raw_dir = tmp_app_config.paths.resolve("raw_data_dir")
    generate_synthetic_cicids2017(
        n_samples=160, seed=21, class_weights={"BENIGN": 1.0}, save_path=raw_dir / "Monday-WorkingHours.csv"
    )
    generate_synthetic_cicids2017(
        n_samples=220,
        seed=22,
        class_weights={"BENIGN": 0.65, "DoS Hulk": 0.25, "FTP-Patator": 0.10},
        save_path=raw_dir / "Wednesday-workingHours.csv",
    )
    generate_synthetic_cicids2017(
        n_samples=160,
        seed=23,
        class_weights={"BENIGN": 0.70, "Web Attack": 0.30},
        save_path=raw_dir / "Thursday-WorkingHours.csv",
    )
    generate_synthetic_cicids2017(
        n_samples=200,
        seed=24,
        class_weights={"BENIGN": 0.55, "DDoS": 0.25, "PortScan": 0.20},
        save_path=raw_dir / "Friday-WorkingHours.csv",
    )

    model_cfg = ModelConfig()
    xgb_fast = {"n_estimators": 8, "max_depth": 3}

    by_day_res = evaluate_by_day_protocol(
        app_cfg=tmp_app_config,
        model_cfg=model_cfg,
        xgb_params=xgb_fast,
        imbalance_strategy="class_weight",
        keras_epochs=1,
        force_synthetic=False,
    )
    assert by_day_res["executed"] is True
    assert by_day_res["leakage_checks"]["passed"] is True
    assert set(["XGBoost", "MLP", "1D-CNN"]).issubset(by_day_res["multiclass_models"].keys())
    assert {"XGBoost (Binary)", "XGBoost (1 - P_Benign)", "Autoencoder"}.issubset(
        by_day_res["binary_cross_day_models"].keys()
    )
    assert {"DDoS", "PortScan", "Web Attack"}.issubset(set(by_day_res["unseen_test_classes"]))

    time_block_res = evaluate_time_block_protocol(
        app_cfg=tmp_app_config,
        model_cfg=model_cfg,
        xgb_params=xgb_fast,
        imbalance_strategy="none",
        keras_epochs=1,
        force_synthetic=False,
    )
    assert time_block_res["executed"] is True
    assert time_block_res["leakage_checks"]["passed"] is True
    assert set(["XGBoost", "MLP", "1D-CNN"]).issubset(time_block_res["models"].keys())
    assert time_block_res["models"]["XGBoost"]["f1_macro_ci95"]["n_bootstrap"] > 0
    assert len(time_block_res["per_class_metrics"]["XGBoost"]) > 0

    ablation_res = evaluate_feature_ablation(
        app_cfg=tmp_app_config,
        model_cfg=model_cfg,
        drop_features=["Destination Port"],
        xgb_params=xgb_fast,
        imbalance_strategy="class_weight",
        force_synthetic=False,
        compute_shap=True,
    )
    assert ablation_res["executed"] is True
    assert ablation_res["identical_test_rows_verified"] is True
    assert ablation_res["with_features"]["n_features"] == 77
    assert ablation_res["without_features"]["n_features"] == 76
    assert (
        ablation_res["with_features"]["split_sizes"]["test"]
        == ablation_res["without_features"]["split_sizes"]["test"]
    )
    assert ablation_res["with_features"]["test_rows_sha256"] != ""
    assert (
        ablation_res["with_features"]["test_rows_sha256"]
        == ablation_res["without_features"]["test_rows_sha256"]
    )
    assert "delta_test_f1_macro" in ablation_res
    wo_shap_names = {f["feature"] for f in ablation_res["without_features"]["top_shap_features"]}
    assert "Destination Port" not in wo_shap_names

    # Verify exact row indices & Test_Rows count survive flows that differ ONLY by Destination Port
    ds_base = run_preprocessing_pipeline(tmp_app_config, force_synthetic=False, save_outputs=False)
    ds_no_port = drop_features_from_prepared_dataset(ds_base, ["Destination Port"], tmp_app_config)
    assert_ablation_row_parity(ds_base, ds_no_port)
    assert ds_no_port.audit_report["leakage_checks"]["passed"] is True

    # Loud failure checks when test indices or test row counts diverge
    bad_idx = copy.copy(ds_no_port)
    bad_idx.test_row_indices = ds_no_port.test_row_indices[::-1].copy()
    with pytest.raises(AssertionError, match="test split row mismatch"):
        assert_ablation_row_parity(ds_base, bad_idx)

    bad_len = copy.copy(ds_no_port)
    bad_len.X_test_scaled = ds_no_port.X_test_scaled[:-1].copy()
    with pytest.raises(AssertionError, match="Test_Rows mismatch"):
        assert_ablation_row_parity(ds_base, bad_len)

    reports_dir = tmp_app_config.paths.resolve("reports_dir")
    cmp_df = pd.DataFrame([{"Model": "XGBoost", "Accuracy": 0.99, "F1_Weighted": 0.99, "F1_Macro": 0.95}])
    summary = {
        "dataset_source": {"is_synthetic": False},
        "split_strategy": "random",
        "time_block_protocol": time_block_res,
        "by_day_protocol": by_day_res,
        "feature_ablation": ablation_res,
        "preprocessing_audit": {"split_strategy": "random", "dropped_features": []},
        "training_diagnostics": {},
    }
    cmp_path, _ = write_academic_reports(summary, cmp_df, reports_dir)
    md_text = cmp_path.read_text(encoding="utf-8")
    assert "Chronological Drift Evaluation (`time_block`" in md_text
    assert "XGBoost Per-Class Metrics (`time_block` Test Partition)" in md_text
    assert "Open-Set Cross-Day Protocol (`by_day`" in md_text
    assert "Open-Set Cross-Day Binary Detection (`by_day`: Benign vs. Attack)" in md_text
    assert "Shortcut Feature Ablation (With vs. Without `Destination Port`)" in md_text
    assert "Delta Test Macro-F1 (without - with):" in md_text


def test_imbalance_selection_bootstrap_ci_per_class_and_xgb_cv(tmp_app_config: AppConfig) -> None:
    """Imbalance selection picks best val Macro-F1, plus bootstrap CI, per-class metrics, and CV."""
    rng = np.random.default_rng(123)
    X_tr = rng.normal(size=(180, 6)).astype(np.float32)
    y_tr = np.array([0] * 140 + [1] * 25 + [2] * 15, dtype=np.int64)
    X_tr[y_tr == 1] += 2.5
    X_tr[y_tr == 2] -= 2.5

    X_va = rng.normal(size=(90, 6)).astype(np.float32)
    y_va = np.array([0] * 70 + [1] * 12 + [2] * 8, dtype=np.int64)
    X_va[y_va == 1] += 2.5
    X_va[y_va == 2] -= 2.5

    # Corrupt SMOTE candidate deliberately so unweighted/class_weight cleanly outperform SMOTE
    X_sm_noisy = rng.normal(scale=4.0, size=(300, 6)).astype(np.float32)
    y_sm_noisy = np.array([0] * 140 + [1] * 80 + [2] * 80, dtype=np.int64)
    cw = {0: 0.43, 1: 2.4, 2: 4.0}
    imb_res = compare_smote_vs_class_weight(
        X_train_scaled=X_tr,
        y_train=y_tr,
        X_train_sm=X_sm_noisy,
        y_train_sm=y_sm_noisy,
        X_val_scaled=X_va,
        y_val=y_va,
        class_weights=cw,
        num_classes=3,
        xgb_device="cpu",
        seed=42,
    )
    scores = {
        "none": imb_res.baseline_unweighted_metrics["val_macro_f1"],
        "class_weight": imb_res.class_weight_metrics["val_macro_f1"],
        "smote": imb_res.smote_metrics["val_macro_f1"],
    }
    best_score = max(scores.values())
    within_tol = [k for k, v in scores.items() if (best_score - v) <= 0.002 + 1e-12]
    assert imb_res.selected_strategy in {"none", "class_weight"}
    if len(within_tol) > 1:
        assert "tie within tolerance, simplest strategy selected" in imb_res.justification
        assert "beating" not in imb_res.justification.lower()
    else:
        assert imb_res.selected_strategy == max(scores, key=lambda k: scores[k])
        assert "Data-driven selection chose" in imb_res.justification

    # Bootstrap CI + Per-class low-support reliability flag
    proba = np.array(
        [[0.9, 0.05, 0.05]] * 40 + [[0.1, 0.85, 0.05]] * 20 + [[0.05, 0.05, 0.9]] * 4,
        dtype=np.float64,
    )
    y_eval = np.array([0] * 40 + [1] * 20 + [2] * 4, dtype=np.int64)
    y_eval[63] = 0  # Flip 1 of 4 minority samples
    proba[63] = [0.8, 0.1, 0.1]
    ev = evaluate_supervised_model(
        y_eval,
        proba,
        ["BENIGN", "DDoS", "Heartbleed"],
        "XGBoost",
        bootstrap_n=250,
        low_support_threshold=30,
        seed=42,
    )
    ci = ev["f1_macro_ci95"]
    direct_ci = bootstrap_macro_f1_ci(y_eval, np.argmax(proba, axis=1), 3, n_bootstrap=250, seed=42)
    assert direct_ci == ci
    assert 0.0 <= ci["ci95_low"] <= ev["f1_macro"] <= ci["ci95_high"] <= 1.0
    assert ci["n_bootstrap"] == 250
    by_cls = {r["class_name"]: r for r in ev["per_class_metrics"]}
    assert by_cls["BENIGN"]["support"] == 41 and by_cls["BENIGN"]["statistically_unreliable"] is False
    assert by_cls["Heartbleed"]["support"] == 3 and by_cls["Heartbleed"]["statistically_unreliable"] is True

    # Leak-free XGBoost Stratified CV + Report formatting
    tmp_app_config.data.synthetic_fallback.n_samples = 900
    cv_res = run_xgboost_stratified_cv(
        app_cfg=tmp_app_config,
        model_cfg=ModelConfig(),
        xgb_params={"n_estimators": 8, "max_depth": 3},
        imbalance_strategy="class_weight",
        n_splits=3,
        force_synthetic=True,
    )
    assert cv_res["executed"] is True
    assert cv_res["n_splits"] >= 2
    assert len(cv_res["folds"]) == cv_res["n_splits"]
    assert 0.0 <= cv_res["mean_macro_f1"] <= 1.0 and cv_res["std_macro_f1"] >= 0.0

    reports_dir = tmp_app_config.paths.resolve("reports_dir")
    cmp_df = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "F1_Macro": ev["f1_macro"],
                "F1_Macro_CI95_Low": ci["ci95_low"],
                "F1_Macro_CI95_High": ci["ci95_high"],
            }
        ]
    )
    summary = {
        "dataset_source": {"is_synthetic": True},
        "supervised_evaluations": {"XGBoost": ev},
        "xgboost_cross_validation": cv_res,
        "preprocessing_audit": {"split_strategy": "random", "dropped_features": []},
        "training_diagnostics": {"imbalance_experiment": imb_res.to_dict()},
    }
    cmp_path, exp_path = write_academic_reports(summary, cmp_df, reports_dir)
    cmp_txt = cmp_path.read_text(encoding="utf-8")
    exp_txt = exp_path.read_text(encoding="utf-8")
    assert "Bootstrap CI" in cmp_txt
    assert "Statistically Unreliable" in cmp_txt
    assert f"XGBoost {cv_res['n_splits']}-Fold Stratified Cross-Validation" in cmp_txt
    assert "Unweighted ('none')" in exp_txt
    assert "Class Weighting ('class_weight')" in exp_txt


def test_temperature_scaler_handles_float32_overconfidence() -> None:
    """Float32 overconfident probabilities must calibrate away from T=1.0 and lower ECE/NLL."""
    rng = np.random.default_rng(7)
    n = 600
    true_logits = rng.normal(scale=1.2, size=(n, 4))
    exp_true = np.exp(true_logits - np.max(true_logits, axis=1, keepdims=True))
    p_true = exp_true / np.sum(exp_true, axis=1, keepdims=True)
    y_val = np.array([rng.choice(4, p=row) for row in p_true], dtype=np.int64)

    # Sharpen predictions (overconfident float32 inputs reproducing old L-BFGS-B flat step bug)
    sharp_logits = true_logits * 2.5
    exp_sharp = np.exp(sharp_logits - np.max(sharp_logits, axis=1, keepdims=True))
    probs_f32 = (exp_sharp / np.sum(exp_sharp, axis=1, keepdims=True)).astype(np.float32)

    ece_before = compute_ece(y_val, probs_f32)
    cal = TemperatureScaler().fit(probs_f32, y_val)
    calibrated = cal.transform(probs_f32)
    ece_after = compute_ece(y_val, calibrated)

    assert cal.temperature > 1.5
    assert cal.nll_after < cal.nll_before
    assert ece_after < ece_before


def test_optuna_cv_std_gating_and_stacking_meta_learner(tmp_path: Path) -> None:
    """Optuna requires > CV std improvement; stacking & val_weighted_voting train on val only."""
    rng = np.random.default_rng(42)
    X_tr = rng.normal(size=(150, 6)).astype(np.float32)
    y_tr = np.array([0] * 100 + [1] * 30 + [2] * 20, dtype=np.int64)
    X_tr[y_tr == 1] += 2.2
    X_tr[y_tr == 2] -= 2.2

    X_va = rng.normal(size=(60, 6)).astype(np.float32)
    y_va = np.array([0] * 40 + [1] * 12 + [2] * 8, dtype=np.int64)
    X_va[y_va == 1] += 2.2
    X_va[y_va == 2] -= 2.2

    xgb_cfg = XGBoostConfig(
        baseline={"n_estimators": 20, "max_depth": 4, "learning_rate": 0.1},
        optuna=XGBoostOptunaConfig(enabled=True, n_trials=2, timeout_seconds=30),
    )
    opt_res = optimize_xgboost_with_optuna(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        num_classes=3,
        cfg=xgb_cfg,
        xgb_device="cpu",
        seed=42,
        cv_macro_f1_mean=0.9123,
        cv_macro_f1_std=0.25,  # Larger than plausible 2-trial fluctuation
        cv_method_note="Unified 5-fold CV on dev pool (ddof=1).",
    )
    assert opt_res.selected_variant == "baseline"
    assert opt_res.meaningful_improvement is False
    assert "No meaningful improvement" in opt_res.selection_note
    assert opt_res.to_summary_dict()["cv_macro_f1_std"] == 0.25
    assert opt_res.to_summary_dict()["cv_macro_f1_mean"] == 0.9123
    assert "Unified 5-fold CV" in opt_res.to_summary_dict()["cv_method_note"]

    # Fallback CV estimation honors imbalance strategy without crashing
    opt_fb = optimize_xgboost_with_optuna(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        num_classes=3,
        cfg=XGBoostConfig(
            baseline={"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1},
            optuna=XGBoostOptunaConfig(enabled=False, n_trials=0),
        ),
        xgb_device="cpu",
        seed=42,
        cv_n_splits=3,
        cv_imbalance_strategy="class_weight",
    )
    assert opt_fb.cv_macro_f1_std >= 0.0
    assert "class_weight" in opt_fb.cv_method_note

    rep_cmp, _ = write_academic_reports(
        summary={
            "dataset_source": {"is_synthetic": True},
            "preprocessing_audit": {"split_strategy": "random", "dropped_features": []},
            "training_diagnostics": {"xgboost_optimization": opt_res.to_summary_dict()},
        },
        comparison_df=pd.DataFrame([{"Model": "XGBoost", "F1_Macro": 0.95}]),
        reports_dir=tmp_path / "opt_reports",
    )
    assert "How CV Std is Computed:" in rep_cmp.read_text(encoding="utf-8")

    # Stacking meta-learner trained strictly on validation probabilities learns to downweight noisy model
    val_probas = {
        "XGBoost": np.where(
            np.eye(3, dtype=bool)[y_va],
            0.92,
            0.04,
        ),
        "CNN": np.full((len(y_va), 3), 1.0 / 3.0),
    }
    stacker = StackingMetaLearner().fit(val_probas=val_probas, y_val=y_va, n_classes=3, seed=42)
    assert stacker.is_fitted is True
    assert stacker.val_weights["XGBoost"] > stacker.val_weights["CNN"]

    test_y = np.array([0, 1, 2], dtype=np.int64)
    test_probas = {
        "XGBoost": np.array([[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]]),
        "CNN": np.array([[0.10, 0.80, 0.10], [0.80, 0.10, 0.10], [0.10, 0.80, 0.10]]),
    }
    ens = HybridEnsembleEngine(
        class_names=["BENIGN", "DDoS", "PortScan"],
        cfg=EnsembleConfig(strategy="hybrid_anomaly_override", base_supervised_strategy="best_single"),
        best_model_name="XGBoost",
        benign_index=0,
        stacker=stacker,
    )
    p_stack, strat_s = ens.combine_probabilities(test_probas, strategy="stacking")
    p_valw, strat_v = ens.combine_probabilities(test_probas, strategy="val_weighted_voting")
    dec_ov = ens.decide(test_probas, anomaly_scores=np.array([3.0, 0.1, 0.1]), anomaly_threshold=1.0)
    assert strat_s == "stacking" and np.array_equal(np.argmax(p_stack, axis=1), test_y)
    assert strat_v == "val_weighted_voting" and np.array_equal(np.argmax(p_valw, axis=1), test_y)
    assert dec_ov.predicted_labels[0] == "Zero-Day Anomaly (Novel)"
    assert dec_ov.predicted_labels[1] == "DDoS"
    assert dec_ov.predicted_labels[2] == "PortScan"


def test_classical_anomaly_baselines_and_unified_evaluation(tmp_app_config: AppConfig) -> None:
    """Isolation Forest, One-Class SVM, and LOF fit on benign-only data and match AE protocol."""
    rng = np.random.default_rng(101)
    X_benign_tr = rng.normal(loc=0.0, scale=1.0, size=(300, 10)).astype(np.float32)
    X_benign_va = rng.normal(loc=0.0, scale=1.0, size=(120, 10)).astype(np.float32)

    X_test = np.vstack(
        [
            rng.normal(loc=0.0, scale=1.0, size=(100, 10)),
            rng.normal(loc=4.5, scale=1.0, size=(40, 10)),
            rng.normal(loc=0.4, scale=1.1, size=(20, 10)),
        ]
    ).astype(np.float32)
    y_test = np.array([0] * 100 + [1] * 40 + [2] * 20, dtype=np.int64)
    cls_names = ["BENIGN", "DoS Hulk", "Web Attack"]

    cfg = AnomalyBaselinesConfig(
        isolation_forest_n_estimators=50,
        isolation_forest_max_samples=128,
        ocsvm_max_train_samples=300,
        lof_max_train_samples=300,
    )
    baselines = fit_classical_anomaly_baselines(
        X_benign_train=X_benign_tr,
        X_benign_val=X_benign_va,
        cfg=cfg,
        percentile=95.0,
        seed=42,
    )
    assert set(baselines.keys()) == {"Isolation Forest", "One-Class SVM", "LOF"}

    models_dir = tmp_app_config.paths.resolve("models_dir")
    joblib.dump(baselines, models_dir / cfg.baselines_file)
    loaded = ModelRegistry(models_dir=models_dir).load_all(load_keras=False)
    assert loaded.anomaly_baselines is not None
    assert set(loaded.anomaly_baselines.keys()) == {"Isolation Forest", "One-Class SVM", "LOF"}

    anom_evals = {}
    det_rows = []
    for b_name, b_obj in loaded.anomaly_baselines.items():
        res_b = b_obj.predict_result(X_test)
        assert len(res_b.scores) == len(X_test)
        assert float(np.mean(res_b.scores[y_test == 1])) > float(np.mean(res_b.scores[y_test == 0]))
        ev = evaluate_anomaly_detector(
            model_name=b_name,
            y_test_multiclass=y_test,
            anomaly_scores_test=res_b.scores,
            benign_val_scores=b_obj.score(X_benign_va),
            threshold=b_obj.threshold,
            class_names=cls_names,
            benign_index=0,
            percentiles=[90.0, 95.0, 97.0, 99.0],
            bootstrap_n=100,
            seed=42,
        )
        assert len(ev["percentile_sweep"]) == 4
        assert "Web Attack" in ev["per_class_anomaly_detection_rate"]
        assert ev["roc_auc"] > 0.75
        anom_evals[b_name] = ev
        det_rows.append(
            {
                "Detector": b_name,
                "Accuracy": ev["accuracy"],
                "Precision": ev["precision"],
                "Recall": ev["recall"],
                "F1": ev["f1"],
                "ROC_AUC": ev["roc_auc"],
                "PR_AUC": ev["pr_auc"],
                "FPR": ev["fpr"],
                "Holdout_Class_Recall": ev["per_class_anomaly_detection_rate"]["Web Attack"]["flagged_rate"],
                "Holdout_Class_ROC_AUC": ev["roc_auc"],
            }
        )

    best_f1 = max(det_rows, key=lambda r: r["F1"])["Detector"]
    best_auc = max(det_rows, key=lambda r: r["ROC_AUC"])["Detector"]
    reports_dir = tmp_app_config.paths.resolve("reports_dir")
    cmp_path, exp_path = write_academic_reports(
        summary={
            "dataset_source": {"is_synthetic": True},
            "anomaly_evaluations": anom_evals,
            "anomaly_detector_comparison": {
                "best_by_f1": best_f1,
                "best_by_roc_auc": best_auc,
                "detectors": det_rows,
                "verdict": f"Best F1={best_f1}, best ROC-AUC={best_auc}.",
            },
            "preprocessing_audit": {"split_strategy": "random", "dropped_features": []},
            "training_diagnostics": {},
        },
        comparison_df=pd.DataFrame([{"Model": r["Detector"], "F1_Macro": r["F1"]} for r in det_rows]),
        reports_dir=reports_dir,
    )
    assert "Unsupervised Anomaly Detector Baselines" in cmp_path.read_text(encoding="utf-8")
    assert "Unsupervised Anomaly Detection & Classical Baselines" in exp_path.read_text(encoding="utf-8")


def test_paired_bootstrap_significance_and_report_conclusion(tmp_path: Path) -> None:
    """Paired bootstrap detects significant vs overlapping differences and drives report text cleanly."""
    y_true = np.array(([0] * 60) + ([1] * 30) + ([2] * 30), dtype=np.int64)
    y_base = y_true.copy()
    # Baseline misclassifies 18 minority-class rows consistently -> clearly weaker Macro-F1
    y_base[60:78] = 0
    y_cand_sig = y_true.copy()

    sig_res = paired_bootstrap_macro_f1_test(
        y_true=y_true,
        y_pred_candidate=y_cand_sig,
        y_pred_baseline=y_base,
        num_classes=3,
        n_bootstrap=400,
        seed=42,
    )
    assert sig_res["observed_diff"] > 0.05
    assert sig_res["ci95_low"] > 0.0
    assert sig_res["statistically_significant"] is True

    # Overlapping / tiny opposing errors yield CI spanning zero (not significant)
    y_cand_tie = y_true.copy()
    y_cand_tie[60:64] = 2
    y_base_tie = y_true.copy()
    y_base_tie[90:94] = 1
    tie_res = paired_bootstrap_macro_f1_test(
        y_true=y_true,
        y_pred_candidate=y_cand_tie,
        y_pred_baseline=y_base_tie,
        num_classes=3,
        n_bootstrap=400,
        seed=42,
    )
    assert tie_res["ci95_low"] <= 0.0 <= tie_res["ci95_high"]
    assert tie_res["statistically_significant"] is False

    # Report generator uses paired bootstrap significance to enforce honest Empirical Conclusion wording
    ns_analysis = {
        "best_single_model": "XGBoost",
        "best_single_f1_macro": 0.9594,
        "best_ensemble_strategy": "stacking",
        "best_ensemble_f1_macro": 0.9661,
        "delta_vs_best_single": 0.0067,
        "paired_bootstrap_vs_best_single": {"stacking": tie_res},
        "paired_bootstrap_stacking_vs_all": {"stacking_vs_soft_voting": sig_res},
    }
    ns_text = format_ensemble_empirical_conclusion(ns_analysis)
    assert "comparable to XGBoost, difference not statistically significant" in ns_text
    assert "outperforming" not in ns_text.lower()

    cmp_path, _ = write_academic_reports(
        summary={
            "dataset_source": {"is_synthetic": True},
            "ensemble_comparison": {
                "best_single": {"f1_macro": 0.9594},
                "stacking": {"f1_macro": 0.9661},
            },
            "ensemble_analysis": ns_analysis,
            "preprocessing_audit": {"split_strategy": "random", "dropped_features": []},
            "training_diagnostics": {},
        },
        comparison_df=pd.DataFrame([{"Model": "XGBoost", "F1_Macro": 0.9594}]),
        reports_dir=tmp_path / "rep_ns",
    )
    md_ns = cmp_path.read_text(encoding="utf-8")
    assert "Paired Bootstrap Macro-F1 Significance" in md_ns
    assert "comparable to XGBoost, difference not statistically significant" in md_ns
    assert "outperforming" not in md_ns.lower()

    sig_analysis = {
        "best_single_model": "XGBoost",
        "best_single_f1_macro": 0.85,
        "best_ensemble_strategy": "stacking",
        "best_ensemble_f1_macro": 0.98,
        "delta_vs_best_single": 0.13,
        "paired_bootstrap_vs_best_single": {"stacking": sig_res},
    }
    sig_text = format_ensemble_empirical_conclusion(sig_analysis)
    assert "statistically significant, CI excludes 0" in sig_text
    assert "outperforming" not in sig_text.lower()


def test_train_all_models_unified_cv_regression(tmp_app_config: AppConfig) -> None:
    """train_all_models invokes run_xgboost_stratified_cv with ModelConfig cleanly."""
    dataset = run_preprocessing_pipeline(tmp_app_config, force_synthetic=True, save_outputs=True)
    m_cfg = ModelConfig()
    m_cfg.mlp.epochs = 1
    m_cfg.cnn.epochs = 1
    m_cfg.autoencoder.epochs = 1
    m_cfg.xgboost.optuna.n_trials = 1
    m_cfg.xgboost.baseline["n_estimators"] = 10
    m_cfg.anomaly_baselines.enabled = False
    tmp_app_config.evaluation.cv_n_splits = 2

    out = train_all_models(
        dataset=dataset,
        app_cfg=tmp_app_config,
        model_cfg=m_cfg,
        epochs_override=1,
        optuna_trials_override=1,
    )
    assert out.xgb_optimization.cv_macro_f1_std >= 0.0
    assert "StratifiedKFold" in out.xgb_optimization.cv_method_note
    assert set(out.imbalance_strategies_by_model.keys()) == {"MLP", "CNN", "XGBoost"}
    assert out.imbalance_experiment.selection_mode == "per_model"
    loaded_meta = ModelRegistry(
        models_dir=tmp_app_config.paths.resolve("models_dir"),
        artifacts_dir=tmp_app_config.paths.resolve("artifacts_dir"),
    ).load_all(load_keras=False)
    for model_key, expected_strat in out.imbalance_strategies_by_model.items():
        assert loaded_meta.model_imbalance_strategies[model_key] == expected_strat
    assert loaded_meta.model_imbalance_strategies["Autoencoder"] == "benign_only"
    assert loaded_meta.model_metadata["MLP"]["imbalance_strategy"] == out.imbalance_strategies_by_model["MLP"]


def test_imbalance_tie_breaking_with_synthetic_metric_dicts() -> None:
    """Tie-breaker prefers none > class_weight > smote within tolerance and never says beating on ties."""
    # 1. Near-tie between 'smote' and 'none' (real CICIDS2017 pattern) -> picks simplest 'none'
    tie_smote_none = {
        "none": {
            "val_macro_f1": 0.951797,
            "val_weighted_f1": 0.998852,
            "val_macro_recall": 0.9323,
            "val_mlogloss": 0.0046,
        },
        "class_weight": {
            "val_macro_f1": 0.944607,
            "val_weighted_f1": 0.998069,
            "val_macro_recall": 0.9583,
            "val_mlogloss": 0.0092,
        },
        "smote": {
            "val_macro_f1": 0.951808,
            "val_weighted_f1": 0.998853,
            "val_macro_recall": 0.9439,
            "val_mlogloss": 0.0044,
        },
    }
    sel1, just1 = select_imbalance_strategy(tie_smote_none, tie_tolerance=0.002)
    assert sel1 == "none"
    assert "tie within tolerance, simplest strategy selected" in just1
    assert "beating" not in just1.lower()

    # 2. 'smote' & 'class_weight' tied within 0.002 while 'none' trails -> prefers 'class_weight' > 'smote'
    tie_cw_smote = {
        "none": {
            "val_macro_f1": 0.9400,
            "val_weighted_f1": 0.9970,
            "val_macro_recall": 0.9200,
            "val_mlogloss": 0.0060,
        },
        "class_weight": {
            "val_macro_f1": 0.9515,
            "val_weighted_f1": 0.9985,
            "val_macro_recall": 0.9500,
            "val_mlogloss": 0.0050,
        },
        "smote": {
            "val_macro_f1": 0.9530,
            "val_weighted_f1": 0.9989,
            "val_macro_recall": 0.9490,
            "val_mlogloss": 0.0045,
        },
    }
    sel2, just2 = select_imbalance_strategy(tie_cw_smote, tie_tolerance=0.002)
    assert sel2 == "class_weight"
    assert "tie within tolerance, simplest strategy selected" in just2
    assert "beating" not in just2.lower()

    # 3. Clear margin > tolerance -> selects genuine winner
    clear_smote = {
        "none": {
            "val_macro_f1": 0.9400,
            "val_weighted_f1": 0.9970,
            "val_macro_recall": 0.9200,
            "val_mlogloss": 0.0060,
        },
        "class_weight": {
            "val_macro_f1": 0.9480,
            "val_weighted_f1": 0.9980,
            "val_macro_recall": 0.9400,
            "val_mlogloss": 0.0050,
        },
        "smote": {
            "val_macro_f1": 0.9600,
            "val_weighted_f1": 0.9990,
            "val_macro_recall": 0.9550,
            "val_mlogloss": 0.0040,
        },
    }
    sel3, just3 = select_imbalance_strategy(clear_smote, tie_tolerance=0.002)
    assert sel3 == "smote"
    assert "beating runner-up" in just3
    assert "tie within tolerance" not in just3.lower()


def test_per_model_imbalance_selection_caps_epochs_and_reports(
    tmp_app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-model imbalance selection evaluates each model independently, caps epochs, and renders matrix."""
    rng = np.random.default_rng(77)
    X_tr = rng.normal(size=(150, 8)).astype(np.float32)
    y_tr = np.array([0] * 110 + [1] * 25 + [2] * 15, dtype=np.int64)
    X_tr[y_tr == 1] += 2.2
    X_tr[y_tr == 2] -= 2.2

    X_sm = rng.normal(size=(240, 8)).astype(np.float32)
    y_sm = np.array([0] * 110 + [1] * 70 + [2] * 60, dtype=np.int64)
    X_sm[y_sm == 1] += 2.2
    X_sm[y_sm == 2] -= 2.2

    X_va = rng.normal(size=(75, 8)).astype(np.float32)
    y_va = np.array([0] * 55 + [1] * 12 + [2] * 8, dtype=np.int64)
    X_va[y_va == 1] += 2.2
    X_va[y_va == 2] -= 2.2

    cw = {0: 0.45, 1: 2.0, 2: 3.3}
    m_cfg = ModelConfig()
    m_cfg.mlp.epochs = 25
    m_cfg.cnn.epochs = 25

    res = compare_imbalance_per_model(
        X_train_scaled=X_tr,
        y_train=y_tr,
        X_train_sm=X_sm,
        y_train_sm=y_sm,
        X_val_scaled=X_va,
        y_val=y_va,
        class_weights=cw,
        num_classes=3,
        model_cfg=m_cfg,
        smote_class_weights={0: 1.0, 1: 1.0, 2: 1.0},
        xgb_device="cpu",
        seed=42,
        tie_tolerance=0.002,
        neural_epochs_cap=2,
    )
    assert res.selection_mode == "per_model"
    assert res.neural_epochs_cap == 2
    assert set(res.per_model.keys()) == {"MLP", "CNN", "XGBoost"}
    assert set(res.selected_strategies_by_model.keys()) == {"MLP", "CNN", "XGBoost"}
    assert res.per_model["MLP"]["epochs_cap_applied"] == 2
    assert res.per_model["CNN"]["epochs_cap_applied"] == 2
    for model_key in ("MLP", "CNN", "XGBoost"):
        assert set(res.per_model[model_key]["metrics_by_strategy"].keys()) == {
            "none",
            "class_weight",
            "smote",
        }

    # Verify independent winners are possible per model using mocked strategy evaluators
    import src.training.optimization as opt_mod

    fake_neural_map = {
        "MLP": {
            "none": {
                "val_macro_f1": 0.67,
                "val_weighted_f1": 0.96,
                "val_macro_recall": 0.65,
                "val_mlogloss": 0.2,
            },
            "class_weight": {
                "val_macro_f1": 0.76,
                "val_weighted_f1": 0.97,
                "val_macro_recall": 0.80,
                "val_mlogloss": 0.15,
            },
            "smote": {
                "val_macro_f1": 0.72,
                "val_weighted_f1": 0.96,
                "val_macro_recall": 0.75,
                "val_mlogloss": 0.18,
            },
        },
        "CNN": {
            "none": {
                "val_macro_f1": 0.70,
                "val_weighted_f1": 0.97,
                "val_macro_recall": 0.68,
                "val_mlogloss": 0.18,
            },
            "class_weight": {
                "val_macro_f1": 0.71,
                "val_weighted_f1": 0.97,
                "val_macro_recall": 0.70,
                "val_mlogloss": 0.17,
            },
            "smote": {
                "val_macro_f1": 0.79,
                "val_weighted_f1": 0.98,
                "val_macro_recall": 0.78,
                "val_mlogloss": 0.12,
            },
        },
    }
    monkeypatch.setattr(
        opt_mod,
        "_evaluate_neural_imbalance_strategies",
        lambda **kw: (fake_neural_map[kw["model_name"]], int(kw["neural_epochs_cap"])),
    )
    monkeypatch.setattr(
        opt_mod,
        "_evaluate_xgb_imbalance_strategies",
        lambda **kw: {
            "none": {
                "val_macro_f1": 0.951,
                "val_weighted_f1": 0.999,
                "val_macro_recall": 0.94,
                "val_mlogloss": 0.01,
            },
            "class_weight": {
                "val_macro_f1": 0.945,
                "val_weighted_f1": 0.998,
                "val_macro_recall": 0.95,
                "val_mlogloss": 0.02,
            },
            "smote": {
                "val_macro_f1": 0.952,
                "val_weighted_f1": 0.999,
                "val_macro_recall": 0.94,
                "val_mlogloss": 0.01,
            },
        },
    )
    diverged = compare_imbalance_per_model(
        X_train_scaled=X_tr,
        y_train=y_tr,
        X_train_sm=X_sm,
        y_train_sm=y_sm,
        X_val_scaled=X_va,
        y_val=y_va,
        class_weights=cw,
        num_classes=3,
        model_cfg=m_cfg,
        tie_tolerance=0.002,
        neural_epochs_cap=5,
    )
    assert diverged.selected_strategies_by_model == {
        "MLP": "class_weight",
        "CNN": "smote",
        "XGBoost": "none",
    }

    # Verify report writer renders all Strategy x Model rows and per-model justifications
    reports_dir = tmp_app_config.paths.resolve("reports_dir")
    cmp_df = pd.DataFrame(
        [
            {
                "Model": "MLP",
                "Imbalance_Strategy": "class_weight",
                "Accuracy": 0.98,
                "F1_Weighted": 0.98,
                "F1_Macro": 0.76,
            },
            {
                "Model": "1D-CNN",
                "Imbalance_Strategy": "smote",
                "Accuracy": 0.98,
                "F1_Weighted": 0.98,
                "F1_Macro": 0.79,
            },
            {
                "Model": "XGBoost",
                "Imbalance_Strategy": "none",
                "Accuracy": 0.99,
                "F1_Weighted": 0.99,
                "F1_Macro": 0.95,
            },
        ]
    )
    summary = {
        "dataset_source": {"is_synthetic": False},
        "split_strategy": "random",
        "preprocessing_audit": {"split_strategy": "random", "dropped_features": []},
        "training_diagnostics": {
            "imbalance_experiment": diverged.to_dict(),
            "imbalance_strategies_by_model": diverged.selected_strategies_by_model,
        },
    }
    _, exp_path = write_academic_reports(summary, cmp_df, reports_dir)
    exp_text = exp_path.read_text(encoding="utf-8")
    assert "MLP (`class_weight`)" in exp_text
    assert "1D-CNN (`smote`)" in exp_text
    assert "XGBoost (`none`)" in exp_text
    assert "Val_Macro_F1" in exp_text


def test_small_class_reliability_oof_wilson_and_reliable_macro_f1(tmp_app_config: AppConfig) -> None:
    """Reliable-class Macro-F1, Wilson 95% recall CIs, and pooled OOF CV evaluation work end-to-end."""
    # 1. Wilson interval bounds and shrinking width with sample size
    lo_0, hi_0 = wilson_score_interval(0, 0)
    assert lo_0 == 0.0 and hi_0 == 0.0
    lo_small, hi_small = wilson_score_interval(1, 2)
    lo_large, hi_large = wilson_score_interval(50, 100)
    assert 0.0 <= lo_small < 0.5 < hi_small <= 1.0
    assert (hi_large - lo_large) < (hi_small - lo_small)

    # 2. Reliable Macro-F1 alongside all-class Macro-F1
    # 2 adequate classes (>= 30 samples, perfect F1=1.0) + 1 rare class (support=4, F1=0.0)
    y_true = np.array([0] * 40 + [1] * 35 + [2] * 4, dtype=np.int64)
    proba = np.vstack(
        [
            np.tile([0.92, 0.04, 0.04], (40, 1)),
            np.tile([0.04, 0.92, 0.04], (35, 1)),
            np.tile([0.90, 0.05, 0.05], (4, 1)),  # rare class misclassified as class 0
        ]
    )
    ev = evaluate_supervised_model(
        y_true,
        proba,
        ["BENIGN", "DDoS", "Heartbleed"],
        "XGBoost",
        bootstrap_n=0,
        low_support_threshold=30,
        seed=42,
    )
    assert ev["f1_macro"] < 0.75
    assert ev["f1_macro_reliable"] > 0.92
    assert ev["reliable_class_names"] == ["BENIGN", "DDoS"]
    assert ev["reliable_support_threshold"] == 30
    by_cls = {r["class_name"]: r for r in ev["per_class_metrics"]}
    assert by_cls["Heartbleed"]["statistically_unreliable"] is True
    assert by_cls["Heartbleed"]["recall"] == 0.0
    assert by_cls["Heartbleed"]["recall_ci95_low"] == 0.0
    assert 0.40 <= by_cls["Heartbleed"]["recall_ci95_high"] <= 0.65
    assert by_cls["DDoS"]["recall_ci95_low"] <= by_cls["DDoS"]["recall"] <= by_cls["DDoS"]["recall_ci95_high"]

    # 3. Pooled 5-fold OOF evaluation in run_xgboost_stratified_cv + report rendering
    tmp_app_config.data.synthetic_fallback.n_samples = 900
    tmp_app_config.evaluation.reliable_support_threshold = 25
    m_cfg = ModelConfig()
    m_cfg.xgboost.baseline["n_estimators"] = 12
    cv_res = run_xgboost_stratified_cv(
        app_cfg=tmp_app_config,
        model_cfg=m_cfg,
        imbalance_strategy="none",
        n_splits=3,
        force_synthetic=True,
    )
    assert cv_res["executed"] is True
    assert cv_res["oof_total_rows"] > 0
    assert cv_res["oof_total_rows"] <= cv_res["dev_pool_rows"]
    assert len(cv_res["oof_per_class"]) >= 2
    first_oof = cv_res["oof_per_class"][0]
    assert "recall_ci95_low" in first_oof and "recall_ci95_high" in first_oof

    reports_dir = tmp_app_config.paths.resolve("reports_dir")
    cmp_df = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "Imbalance_Strategy": "none",
                "Accuracy": 0.99,
                "F1_Weighted": 0.99,
                "F1_Macro": ev["f1_macro"],
                "F1_Macro_Reliable": ev["f1_macro_reliable"],
            }
        ]
    )
    summary = {
        "dataset_source": {"is_synthetic": False},
        "split_strategy": "random",
        "preprocessing_audit": {"split_strategy": "random", "dropped_features": []},
        "training_diagnostics": {},
        "supervised_evaluations": {"XGBoost": ev},
        "xgboost_cross_validation": cv_res,
        "time_block_protocol": {
            "executed": True,
            "ordering_source": "Capture_Order",
            "ordering_warning": "time_block ordering = CSV row order (no real Timestamp)",
            "split_sizes": {"train_raw": 100, "validation": 30, "test": 30},
            "leakage_checks": {"passed": True},
            "models": {
                "XGBoost": {
                    "accuracy": 0.9,
                    "f1_weighted": 0.9,
                    "f1_macro": 0.7,
                    "f1_macro_reliable": 0.92,
                    "reliable_support_threshold": 30,
                    "unreliable_class_names": ["Bot", "Heartbleed"],
                }
            },
        },
        "by_day_protocol": {
            "executed": True,
            "split_sizes": {"train_raw": 100, "validation": 30, "test": 30},
            "leakage_checks": {"passed": True},
            "unseen_test_classes": ["DDoS"],
            "binary_cross_day_models": {"XGBoost (1 - P_Benign)": {"f1": 0.45}},
            "multiclass_models": {
                "XGBoost": {
                    "accuracy": 0.7,
                    "precision_weighted": 0.7,
                    "recall_weighted": 0.7,
                    "f1_weighted": 0.7,
                    "f1_macro": 0.25,
                    "f1_macro_reliable": 0.80,
                    "f1_macro_present_test_classes": 0.30,
                    "reliable_support_threshold": 30,
                    "unreliable_class_names": ["DDoS"],
                    "roc_auc_macro_ovr": 0.75,
                }
            },
        },
    }
    cmp_path, _ = write_academic_reports(summary, cmp_df, reports_dir)
    md_text = cmp_path.read_text(encoding="utf-8")
    assert "F1_Macro_Reliable" in md_text
    assert (
        md_text.count("F1_Macro_Reliable excludes classes with support below 30 on the evaluated partition")
        == 3
    )
    assert (
        md_text.count("F1_Macro_Reliable excludes classes with support below 25 on the evaluated partition")
        == 1
    )
    assert "Excluded classes: Heartbleed." in md_text
    assert "Excluded classes: Bot, Heartbleed." in md_text
    assert "Excluded classes: DDoS." in md_text
    assert "DDoS, DoS GoldenEye, DoS Hulk, FTP-Patator, PortScan, SSH-Patator, Web Attack." in md_text
    assert "Pooled Out-of-Fold (OOF) Per-Class Evaluation" in md_text
    assert "Recall_CI95_Low" in md_text


def test_time_block_failure_analysis_ks_drift_and_mitigation_tradeoffs(tmp_app_config: AppConfig) -> None:
    """Failure analysis identifies shifted KS features/confusion targets and compares Recall vs FPR mitigations."""
    rng = np.random.default_rng(2025)
    # Create 3 classes: BENIGN (0), DDoS (1), Bot (2) where Bot shifts sharply in Flow_IAT_Mean train->test
    X_tr = rng.normal(0.0, 1.0, size=(120, 4)).astype(np.float32)
    y_tr = np.array([0] * 60 + [1] * 30 + [2] * 30, dtype=np.int64)
    X_te = rng.normal(0.0, 1.0, size=(90, 4)).astype(np.float32)
    y_te = np.array([0] * 40 + [1] * 25 + [2] * 25, dtype=np.int64)
    # Strong temporal shift on feature 2 ("Flow IAT Mean") for failing class Bot (2)
    X_te[y_te == 2, 2] += 5.0

    # Probabilities: Bot is mostly misclassified as BENIGN
    p_base = np.zeros((len(y_te), 3), dtype=np.float64)
    p_base[y_te == 0] = [0.95, 0.03, 0.02]
    p_base[y_te == 1] = [0.02, 0.95, 0.03]
    p_base[y_te == 2] = [0.85, 0.05, 0.10]  # predicted BENIGN -> Bot recall = 0.0
    bot_first_idx = int(np.where(y_te == 2)[0][0])
    p_base[bot_first_idx] = [0.10, 0.80, 0.10]  # 1 flow predicted as DDoS

    class_names = ["BENIGN", "DDoS", "Bot"]
    feat_names = ["Destination Port", "Flow Duration", "Flow IAT Mean", "SYN Flag Count"]
    ev = evaluate_supervised_model(y_te, p_base, class_names, "XGBoost", bootstrap_n=0, seed=42)
    fail_res = analyze_time_block_failures(
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        y_proba=p_base,
        feature_names=feat_names,
        class_names=class_names,
        per_class_metrics=ev["per_class_metrics"],
        recall_threshold=0.85,
        top_k_features=2,
    )
    assert fail_res["failing_class_names"] == ["Bot"]
    assert fail_res["confusion_breakdowns"][0]["Class"] == "Bot"
    assert "BENIGN: 24" in fail_res["confusion_breakdowns"][0]["Predicted_As_Breakdown"]
    assert fail_res["feature_drift_top_features"][0]["Feature"] == "Flow IAT Mean"
    assert fail_res["feature_drift_top_features"][0]["KS_Statistic"] > 0.8

    # Mitigation comparison across class_weight, hybrid anomaly override, and P(benign) gates
    y_val = np.array([0] * 40 + [1] * 10 + [2] * 10, dtype=np.int64)
    p_val = np.tile([0.96, 0.02, 0.02], (len(y_val), 1))
    p_alt_val = p_val.copy()
    p_alt_val[y_val == 2] = [0.10, 0.10, 0.80]
    ae_val_ben = rng.uniform(0.0, 1.0, size=40)
    ae_val_all = np.concatenate([ae_val_ben, rng.uniform(0.0, 0.9, size=10), np.full(10, 2.5)])
    ae_test = np.where(y_te == 2, 2.5, rng.uniform(0.0, 1.0, size=len(y_te)))
    p_alt = p_base.copy()
    p_alt[y_te == 2] = [0.10, 0.10, 0.80]  # class_weight recovers Bot in this synthetic case

    mit = evaluate_mitigation_tradeoffs(
        y_val=y_val,
        benign_idx=0,
        p_sup_base_val=p_val,
        ae_val_benign_errors=ae_val_ben,
        y_test=y_te,
        p_sup_base_test=p_base,
        ae_test_errors=ae_test,
        class_names=class_names,
        focus_class_indices=[2],
        p_sup_alt_test=p_alt,
        percentiles=[90.0, 95.0],
        ae_val_all_errors=ae_val_all,
        p_sup_alt_val=p_alt_val,
    )
    options = {r["Option"] for r in mit["rows"]}
    assert "Baseline XGBoost" in options
    assert "XGBoost ('class_weight' balanced)" in options
    assert "Hybrid Override (XGBoost OR AE)" in options
    assert "Lower P(Benign) Gate (1 - P_Benign > tau)" in options
    assert len(mit["validation_rows"]) == len(mit["rows"])
    assert "summary" in mit["validation_vs_test_agreement"]
    assert mit["rows"][0]["Focus_Mean_Recall"] == pytest.approx(0.04)
    assert any(
        r["Option"] == "Hybrid Override (XGBoost OR AE)" and r["Focus_Mean_Recall"] == 1.0
        for r in mit["rows"]
    )

    reports_dir = tmp_app_config.paths.resolve("reports_dir")
    cmp_df = pd.DataFrame([{"Model": "XGBoost", "Accuracy": 0.9, "F1_Weighted": 0.9, "F1_Macro": 0.65}])
    summary = {
        "dataset_source": {"is_synthetic": False},
        "split_strategy": "random",
        "preprocessing_audit": {"split_strategy": "random", "dropped_features": []},
        "training_diagnostics": {},
        "time_block_protocol": {
            "executed": True,
            "ordering_source": "Capture_Order",
            "ordering_warning": "time_block ordering = CSV row order (no real Timestamp)",
            "split_sizes": {"train_raw": 120, "validation": 60, "test": 90},
            "leakage_checks": {"passed": True},
            "models": {
                "XGBoost": {"accuracy": 0.9, "f1_weighted": 0.9, "f1_macro": 0.65, "f1_macro_reliable": 0.65}
            },
            "failure_analysis": fail_res,
            "mitigation_experiments": {"executed": True, **mit},
        },
    }
    cmp_path, _ = write_academic_reports(summary, cmp_df, reports_dir)
    md = cmp_path.read_text(encoding="utf-8")
    assert "Time-Block Error Analysis" in md
    assert "Validation Partition Screening" in md
    assert (
        "Evaluated on the held-out test partition for description only; options were NOT selected on test data."
        in md
    )
    assert "Validation vs. Test Ranking Agreement" in md
    assert "Confusion-Matrix Breakdown for Failing Classes" in md
    assert "Top Shifted Features Train-vs-Test per Failing Class (KS Statistic)" in md
    assert "Time-Block Recall-vs-FPR Mitigation Experiment" in md


def test_deep_ae_vae_and_leave_attack_out_score_fusion(tmp_app_config: AppConfig) -> None:
    """Deep regularized AE, VAE, and non-test leave-attack-out score fusion evaluate and reload cleanly."""
    from src.config import AutoencoderConfig
    from src.training.callbacks import safe_save_keras_model

    rng = np.random.default_rng(314)
    X_ben_tr = rng.normal(0.0, 1.0, size=(240, 8)).astype(np.float32)
    X_val = np.vstack(
        [
            rng.normal(0.0, 1.0, size=(100, 8)),
            rng.normal(3.8, 1.0, size=(25, 8)),
            rng.normal(-3.5, 1.0, size=(20, 8)),
        ]
    ).astype(np.float32)
    y_val = np.array([0] * 100 + [1] * 25 + [2] * 20, dtype=np.int64)
    X_test = np.vstack(
        [
            rng.normal(0.0, 1.0, size=(90, 8)),
            rng.normal(4.0, 1.0, size=(30, 8)),
            rng.normal(-3.8, 1.0, size=(20, 8)),
        ]
    ).astype(np.float32)
    y_test = np.array([0] * 90 + [1] * 30 + [2] * 20, dtype=np.int64)
    cls_names = ["BENIGN", "DDoS", "Web Attack"]

    ae_cfg = AutoencoderConfig(
        encoder_units=[32, 16],
        latent_dim=8,
        decoder_units=[16, 32],
        deep_reg_encoder_units=[48, 24],
        deep_reg_latent_dim=10,
        vae_encoder_units=[32, 16],
        vae_latent_dim=6,
        epochs=2,
        variant_epochs_cap=2,
        batch_size=64,
    )
    models_dir = tmp_app_config.paths.resolve("models_dir")

    ae_base, _, _ = build_autoencoder(8, cfg=ae_cfg)
    ae_base.fit(X_ben_tr, X_ben_tr, epochs=2, batch_size=64, verbose=0)
    safe_save_keras_model(ae_base, models_dir / ae_cfg.checkpoint_name)

    deep_ae = build_deep_regularized_autoencoder(8, cfg=ae_cfg)
    deep_ae.fit(X_ben_tr, X_ben_tr, epochs=2, batch_size=64, verbose=0)
    safe_save_keras_model(deep_ae, models_dir / ae_cfg.deep_reg_checkpoint_name)

    vae = build_vae(8, cfg=ae_cfg)
    vae.fit(X_ben_tr, X_ben_tr, epochs=2, batch_size=64, verbose=0)
    safe_save_keras_model(vae, models_dir / ae_cfg.vae_checkpoint_name)

    base_cls = fit_classical_anomaly_baselines(
        X_ben_tr,
        X_val[y_val == 0],
        AnomalyBaselinesConfig(
            isolation_forest_n_estimators=25,
            isolation_forest_max_samples=128,
            ocsvm_max_train_samples=240,
            lof_max_train_samples=240,
        ),
        percentile=95.0,
        seed=42,
    )
    joblib.dump(base_cls, models_dir / "anomaly_baselines.pkl")

    val_scores = {
        "Autoencoder": compute_reconstruction_error(ae_base, X_val),
        "Autoencoder (Deep Reg)": compute_reconstruction_error(deep_ae, X_val),
        "VAE": compute_reconstruction_error(vae, X_val),
        "One-Class SVM": base_cls["One-Class SVM"].score(X_val),
        "Isolation Forest": base_cls["Isolation Forest"].score(X_val),
        "LOF": base_cls["LOF"].score(X_val),
    }
    fusion = fit_and_select_anomaly_fusion(
        val_scores_by_detector=val_scores,
        y_val=y_val,
        benign_index=0,
        percentile=95.0,
        grid_step=0.25,
    )
    assert abs(sum(fusion.weights.values()) - 1.0) < 1e-6
    assert len(fusion.validation_selection_summary) == 7
    joblib.dump(fusion, models_dir / ae_cfg.fusion_file)

    reloaded = ModelRegistry(models_dir=models_dir).load_all(load_keras=True)
    assert "Autoencoder (Deep Reg)" in reloaded.models
    assert "VAE" in reloaded.models
    assert reloaded.anomaly_fusion is not None

    test_scores = {
        "Autoencoder": compute_reconstruction_error(reloaded.models["Autoencoder"], X_test),
        "Autoencoder (Deep Reg)": compute_reconstruction_error(
            reloaded.models["Autoencoder (Deep Reg)"], X_test
        ),
        "VAE": compute_reconstruction_error(reloaded.models["VAE"], X_test),
        "One-Class SVM": base_cls["One-Class SVM"].score(X_test),
        "Isolation Forest": base_cls["Isolation Forest"].score(X_test),
        "LOF": base_cls["LOF"].score(X_test),
    }
    fused_val = reloaded.anomaly_fusion.score_from_components(
        {k: v[y_val == 0] for k, v in val_scores.items()}
    )
    fused_test = reloaded.anomaly_fusion.score_from_components(test_scores)
    fused_ev = evaluate_anomaly_detector(
        "Score-Level Fusion (AE+OCSVM+IF)",
        y_test,
        fused_test,
        fused_val,
        float(reloaded.anomaly_fusion.threshold),
        cls_names,
        benign_index=0,
        percentiles=[90.0, 95.0],
        bootstrap_n=50,
        seed=42,
    )
    assert fused_ev["roc_auc"] > 0.80
    assert fused_ev["f1_ci95"]["ci95_low"] <= fused_ev["f1"] <= fused_ev["f1_ci95"]["ci95_high"]

    reports_dir = tmp_app_config.paths.resolve("reports_dir")
    cmp_df = pd.DataFrame([{"Model": "XGBoost", "Accuracy": 0.99, "F1_Weighted": 0.99, "F1_Macro": 0.95}])
    summary = {
        "dataset_source": {"is_synthetic": False},
        "split_strategy": "random",
        "preprocessing_audit": {"split_strategy": "random", "dropped_features": []},
        "training_diagnostics": {
            "anomaly_validation_selection": {
                "selection_table": fusion.validation_selection_summary,
                "fusion_weights": fusion.weights,
                "fusion_threshold": fusion.threshold,
            }
        },
        "anomaly_detector_comparison": {
            "detectors": [
                {
                    "Detector": "Score-Level Fusion (AE+OCSVM+IF)",
                    "F1": fused_ev["f1"],
                    "F1_CI95_Low": fused_ev["f1_ci95"]["ci95_low"],
                    "F1_CI95_High": fused_ev["f1_ci95"]["ci95_high"],
                    "ROC_AUC": fused_ev["roc_auc"],
                }
            ],
            "verdict": "Supplementary zero-day signals only.",
        },
    }
    cmp_path, _ = write_academic_reports(summary, cmp_df, reports_dir)
    md = cmp_path.read_text(encoding="utf-8")
    assert "supplementary zero-day signals" in md
    assert "Anomaly Validation Selection Summary (Non-Test Leave-Attack-Out)" in md


def test_optuna_budget_and_bounded_neural_loss_keep_revert(
    tmp_app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Optuna defaults to 40 trials with timeout_minutes; bounded neural trials keep only if > tolerance."""
    import time as time_mod

    from src.training.callbacks import safe_save_keras_model
    from src.training.trainer import _run_bounded_neural_loss_experiment

    default_mcfg = ModelConfig()
    assert default_mcfg.optuna.n_trials == 40
    assert default_mcfg.optuna.timeout_minutes == pytest.approx(6.0)
    assert default_mcfg.optuna.timeout_seconds == 360

    coerced_mcfg = ModelConfig(optuna={"n_trials": 25, "timeout_minutes": 2.5})
    assert coerced_mcfg.xgboost.optuna.n_trials == 25
    assert coerced_mcfg.xgboost.optuna.timeout_seconds == 150

    rng = np.random.default_rng(19)
    X_tr = rng.normal(size=(120, 6)).astype(np.float32)
    y_tr = np.array([0] * 60 + [1] * 30 + [2] * 30, dtype=np.int64)
    X_tr[y_tr == 1] += 2.0
    X_tr[y_tr == 2] -= 2.0
    X_va = rng.normal(size=(60, 6)).astype(np.float32)
    y_va = np.array([0] * 30 + [1] * 15 + [2] * 15, dtype=np.int64)
    X_va[y_va == 1] += 2.0
    X_va[y_va == 2] -= 2.0

    m_cfg = ModelConfig()
    m_cfg.mlp.hidden_units = [16, 8]
    m_cfg.mlp.epochs = 2
    m_cfg.mlp.batch_size = 32
    m_cfg.neural.epochs_cap = 2
    m_cfg.neural.improvement_tolerance = 0.01

    models_dir = tmp_app_config.paths.resolve("models_dir")
    mlp_ckpt = models_dir / "best_mlp.keras"

    # Build + save custom focal-loss MLP directly and verify reload succeeds
    focal_mlp = build_mlp(6, 3, cfg=m_cfg.mlp, loss_type="focal_loss", focal_gamma=2.0)
    focal_mlp.fit(X_tr, y_tr, epochs=1, batch_size=32, verbose=0)
    safe_save_keras_model(focal_mlp, mlp_ckpt)
    reloaded = ModelRegistry(models_dir=models_dir).load_all(load_keras=True)
    assert "MLP" in reloaded.models

    # Verify KEEP path vs REVERT path via deterministic _eval_val_scores scoring
    base_mlp = build_mlp(6, 3, cfg=m_cfg.mlp, loss_type="cross_entropy")
    base_mlp.fit(X_tr, y_tr, epochs=1, batch_size=32, verbose=0)
    safe_save_keras_model(base_mlp, mlp_ckpt)

    import src.training.trainer as trainer_mod

    scores_seq = [
        {"val_macro_f1": 0.70, "val_weighted_f1": 0.96, "val_macro_recall": 0.70, "val_mlogloss": 0.25},
        {
            "val_macro_f1": 0.705,
            "val_weighted_f1": 0.96,
            "val_macro_recall": 0.70,
            "val_mlogloss": 0.24,
        },  # < +0.01 tol
        {"val_macro_f1": 0.69, "val_weighted_f1": 0.95, "val_macro_recall": 0.69, "val_mlogloss": 0.28},
    ]
    it_rev = iter(scores_seq)
    monkeypatch.setattr(trainer_mod, "_eval_val_scores", lambda *a, **kw: next(it_rev))
    kept_model_rev, _, loss_rev, rep_rev = _run_bounded_neural_loss_experiment(
        model_name="MLP",
        base_model=base_mlp,
        base_history={"loss": [0.5], "val_loss": [0.4]},
        ckpt_path=mlp_ckpt,
        X_fit=X_tr,
        y_fit=y_tr,
        X_val=X_va,
        y_val=y_va,
        num_features=6,
        num_classes=3,
        class_weight=None,
        base_epochs=2,
        model_cfg=m_cfg,
        deadline_perf=time_mod.perf_counter() + 60.0,
        seed=42,
    )
    assert loss_rev == "cross_entropy"
    assert kept_model_rev is base_mlp
    assert "Reverted" in rep_rev["decision"]

    scores_keep = [
        {"val_macro_f1": 0.70, "val_weighted_f1": 0.96, "val_macro_recall": 0.70, "val_mlogloss": 0.25},
        {
            "val_macro_f1": 0.75,
            "val_weighted_f1": 0.97,
            "val_macro_recall": 0.76,
            "val_mlogloss": 0.18,
        },  # beats +0.01
        {"val_macro_f1": 0.72, "val_weighted_f1": 0.96, "val_macro_recall": 0.72, "val_mlogloss": 0.22},
    ]
    it_keep = iter(scores_keep)
    monkeypatch.setattr(trainer_mod, "_eval_val_scores", lambda *a, **kw: next(it_keep))
    kept_model_win, _, loss_win, rep_win = _run_bounded_neural_loss_experiment(
        model_name="MLP",
        base_model=base_mlp,
        base_history={"loss": [0.5], "val_loss": [0.4]},
        ckpt_path=mlp_ckpt,
        X_fit=X_tr,
        y_fit=y_tr,
        X_val=X_va,
        y_val=y_va,
        num_features=6,
        num_classes=3,
        class_weight=None,
        base_epochs=2,
        model_cfg=m_cfg,
        deadline_perf=time_mod.perf_counter() + 60.0,
        seed=42,
    )
    assert loss_win == "label_smoothing"
    assert kept_model_win is not base_mlp
    assert "Kept" in rep_win["decision"]

    # Expired budget skips candidate runs cleanly without crashing
    monkeypatch.undo()
    _, _, loss_skip, rep_skip = _run_bounded_neural_loss_experiment(
        model_name="MLP",
        base_model=base_mlp,
        base_history={"loss": [0.5], "val_loss": [0.4]},
        ckpt_path=mlp_ckpt,
        X_fit=X_tr,
        y_fit=y_tr,
        X_val=X_va,
        y_val=y_va,
        num_features=6,
        num_classes=3,
        class_weight=None,
        base_epochs=2,
        model_cfg=m_cfg,
        deadline_perf=time_mod.perf_counter() - 1.0,
        seed=42,
    )
    assert loss_skip == "cross_entropy"
    assert all(c["status"] == "skipped_time_budget" for c in rep_skip["candidates"][1:])

    reports_dir = tmp_app_config.paths.resolve("reports_dir")
    cmp_df = pd.DataFrame([{"Model": "XGBoost", "Accuracy": 0.99, "F1_Weighted": 0.99, "F1_Macro": 0.95}])
    cmp_path, _ = write_academic_reports(
        {
            "dataset_source": {"is_synthetic": False},
            "split_strategy": "random",
            "preprocessing_audit": {"split_strategy": "random", "dropped_features": []},
            "training_diagnostics": {"neural_loss_experiments": {"MLP": rep_rev}},
        },
        cmp_df,
        reports_dir,
    )
    md_text = cmp_path.read_text(encoding="utf-8")
    assert "Neural Baselines vs. Recommended Production Model" in md_text
    assert "Bounded Neural Improvement Trials" in md_text


def test_degenerate_fusion_labeling_and_na_unmeasured_values(tmp_path: Path) -> None:
    """Single-detector fusion is labeled degenerate and unmeasured cells render as N/A (not 0.0)."""
    from dashboard.pages.model_performance import _display_df_with_na
    from src.detection.anomaly_detection import describe_score_level_fusion
    from src.evaluation.report_writer import normalize_fusion_display_df

    deg_label, is_deg, deg_note = describe_score_level_fusion(
        {"Autoencoder": 1.0, "One-Class SVM": 0.0, "Isolation Forest": 0.0}
    )
    assert deg_label == "Score-Level Fusion (degenerate: equals Autoencoder)"
    assert is_deg is True
    assert (
        deg_note
        == "Validation selected a single-detector fusion; no gain from fusing detectors on this data."
    )

    multi_label, is_multi_deg, multi_note = describe_score_level_fusion(
        {"Autoencoder": 0.6, "One-Class SVM": 0.4}
    )
    assert is_multi_deg is False
    assert multi_note is None

    cmp_df = pd.DataFrame(
        [
            {
                "Model": "Autoencoder",
                "Accuracy": 0.8934,
                "F1_Weighted": 0.6926,
                "F1_Macro": 0.6926,
                "F1_Macro_Reliable": None,
                "Latency_Median_us": 198.7,
                "Training_Duration_s": 68.8,
            },
            {
                "Model": "Score-Level Fusion (AE+OCSVM+IF)",
                "Accuracy": 0.8934,
                "F1_Weighted": 0.6926,
                "F1_Macro": 0.6926,
                "F1_Macro_Reliable": None,
                "Latency_Median_us": None,
                "Training_Duration_s": None,
            },
        ]
    )
    csv_path = tmp_path / "model_comparison.csv"
    cmp_df.to_csv(csv_path, index=False)
    csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
    # Empty CSV fields (",," or trailing ",") rather than "0.0" for unmeasured values
    assert csv_lines[2].endswith(",,")

    summary = {
        "dataset_source": {"is_synthetic": False},
        "split_strategy": "random",
        "preprocessing_audit": {"split_strategy": "random", "dropped_features": []},
        "training_diagnostics": {
            "neural_loss_experiments": {
                "CNN": {
                    "model": "CNN",
                    "kept_variant": "cross_entropy",
                    "decision": "Reverted",
                    "candidates": [
                        {
                            "loss_variant": "focal_loss",
                            "status": "skipped_time_budget",
                            "val_macro_f1": None,
                            "elapsed_sec": 0.0,
                        }
                    ],
                }
            }
        },
        "anomaly_detector_comparison": {
            "val_selection": {"fusion_weights": {"Autoencoder": 1.0, "One-Class SVM": 0.0}},
            "detectors": [
                {"Detector": "Autoencoder", "Accuracy": 0.8934, "F1": 0.6926, "ROC_AUC": 0.8676},
                {
                    "Detector": "Score-Level Fusion (AE+OCSVM+IF)",
                    "Accuracy": 0.8934,
                    "F1": 0.6926,
                    "ROC_AUC": 0.8676,
                },
            ],
            "verdict": "Supplementary zero-day signals only.",
        },
    }
    cmp_md_path, exp_md_path = write_academic_reports(summary, cmp_df, tmp_path)
    cmp_md = cmp_md_path.read_text(encoding="utf-8")
    exp_md = exp_md_path.read_text(encoding="utf-8")
    assert "Score-Level Fusion (degenerate: equals Autoencoder)" in cmp_md
    assert "Score-Level Fusion (degenerate: equals Autoencoder)" in exp_md
    assert (
        "Validation selected a single-detector fusion; no gain from fusing detectors on this data." in cmp_md
    )
    assert (
        "Validation selected a single-detector fusion; no gain from fusing detectors on this data." in exp_md
    )
    assert "| 0.8934 | 0.6926 | 0.6926 | N/A | N/A | N/A |" in cmp_md
    assert "focal_loss | skipped_time_budget | N/A | N/A |" in cmp_md

    ui_df = _display_df_with_na(normalize_fusion_display_df(cmp_df, summary, name_col="Model")[0])
    assert ui_df.loc[1, "Model"] == "Score-Level Fusion (degenerate: equals Autoencoder)"
    assert ui_df.loc[1, "Latency_Median_us"] == "N/A"
    assert ui_df.loc[1, "Training_Duration_s"] == "N/A"


def test_time_aware_xgboost_profile_selection_validation_only_and_reporting(
    tmp_app_config: AppConfig,
    tmp_path: Path,
) -> None:
    """select_xgboost_profile_time_aware never touches test data, tracks Benign FPR, and renders reports."""
    tb_cfg = copy.deepcopy(tmp_app_config)
    tb_cfg.data.split_strategy = "time_block"
    ds = run_preprocessing_pipeline(tb_cfg, force_synthetic=True, save_outputs=False)

    model_cfg = ModelConfig()
    model_cfg.hardware.use_gpu = False
    fast_xgb = {"n_estimators": 12, "max_depth": 3}

    rand_ctx = {
        "none": {"val_macro_f1": 0.9518},
        "class_weight": {"val_macro_f1": 0.9446},
    }
    res_clean = select_xgboost_profile_time_aware(
        app_cfg=tb_cfg,
        model_cfg=model_cfg,
        dataset=ds,
        xgb_params=fast_xgb,
        random_context=rand_ctx,
    )

    ds_corrupt = copy.deepcopy(ds)
    ds_corrupt.y_test = (ds_corrupt.y_test + 1) % max(len(ds_corrupt.class_names), 2)
    ds_corrupt.X_test_scaled = np.zeros_like(ds_corrupt.X_test_scaled)
    res_corrupt = select_xgboost_profile_time_aware(
        app_cfg=tb_cfg,
        model_cfg=model_cfg,
        dataset=ds_corrupt,
        xgb_params=fast_xgb,
        random_context=rand_ctx,
    )

    assert res_clean["executed"] is True
    assert res_clean["used_test_data"] is False
    assert res_clean["selection_split"] == "time_block"
    assert res_clean["selected_profile"] in {"none", "class_weight"}
    assert res_clean["selected_profile"] == res_corrupt["selected_profile"]
    assert res_clean["metrics_by_profile"] == res_corrupt["metrics_by_profile"]
    for prof in ("none", "class_weight"):
        assert "val_macro_f1" in res_clean["metrics_by_profile"][prof]
        assert "val_benign_fpr" in res_clean["metrics_by_profile"][prof]
    assert "Validation Benign FPR (time_block):" in res_clean["justification"]
    assert "Random-split validation context:" in res_clean["justification"]

    cmp_df = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "Accuracy": 0.9,
                "F1_Weighted": 0.9,
                "F1_Macro": 0.85,
            }
        ]
    )
    summary = {
        "dataset_source": {"is_synthetic": True},
        "split_strategy": "random",
        "preprocessing_audit": {"split_strategy": "random", "dropped_features": []},
        "training_diagnostics": {},
        "time_block_profile_selection": res_clean,
        "time_block_protocol": {
            "executed": True,
            "split_sizes": ds.audit_report.get("split_sizes", {}),
            "leakage_checks": {"passed": True},
            "ordering_source": ds.audit_report.get("time_block_details", {}).get(
                "ordering_source", "CSV_Row_Order"
            ),
            "models": {},
            "profile_selection": res_clean,
        },
    }
    cmp_md_path, exp_md_path = write_academic_reports(summary, cmp_df, tmp_path / "rep")
    cmp_md = cmp_md_path.read_text(encoding="utf-8")
    exp_md = exp_md_path.read_text(encoding="utf-8")
    for md_text in (cmp_md, exp_md):
        assert "Time-Aware XGBoost" in md_text
        assert "Val_Benign_FPR" in md_text
        assert "used_test_data=False" in md_text
        assert "Random-split validation context" in md_text


def test_xgboost_downstream_refit_validation_only_keeps_mlp_cnn_calibrators(
    tmp_app_config: AppConfig,
) -> None:
    """Refit touches only the XGBoost calibrator/stacker/SHAP entries, using validation data only."""
    dataset = run_preprocessing_pipeline(tmp_app_config, force_synthetic=True, save_outputs=True)
    m_cfg = ModelConfig()
    m_cfg.mlp.epochs = 1
    m_cfg.cnn.epochs = 1
    m_cfg.autoencoder.epochs = 1
    m_cfg.xgboost.optuna.n_trials = 1
    m_cfg.xgboost.baseline["n_estimators"] = 10
    m_cfg.anomaly_baselines.enabled = False
    tmp_app_config.evaluation.cv_n_splits = 2
    train_all_models(
        dataset=dataset,
        app_cfg=tmp_app_config,
        model_cfg=m_cfg,
        epochs_override=1,
        optuna_trials_override=1,
    )
    train_xgboost_only_profiles(
        app_cfg=tmp_app_config,
        model_cfg=m_cfg,
        profiles=("none", "class_weight"),
        optuna_trials_override=0,
    )

    models_dir = tmp_app_config.paths.resolve("models_dir")
    artifacts_dir = tmp_app_config.paths.resolve("artifacts_dir")
    cal_before = joblib.load(models_dir / "calibrators.pkl")
    mlp_t_before = cal_before["MLP"].temperature
    cnn_t_before = cal_before["CNN"].temperature
    mlp_ckpt_bytes = (models_dir / "best_mlp.keras").read_bytes()

    shap_path = artifacts_dir / "shap_summary.json"
    shap_path.parent.mkdir(parents=True, exist_ok=True)
    shap_path.write_text('{"MLP": {"explainer_type": "sentinel"}}', encoding="utf-8")

    res_a = refit_xgboost_downstream_artifacts(tmp_app_config, m_cfg, compute_shap=False)
    cal_a = joblib.load(models_dir / "calibrators.pkl")
    assert res_a["active_profile"] == "class_weight"
    assert cal_a["MLP"].temperature == mlp_t_before
    assert cal_a["CNN"].temperature == cnn_t_before
    assert res_a["untouched_calibrators_after"] == res_a["untouched_calibrators_before"]
    assert res_a["shap_refreshed"] is False
    assert (models_dir / "best_mlp.keras").read_bytes() == mlp_ckpt_bytes

    # Corrupt the held-out test partition: validation-only fitting must be invariant to it.
    npz_path = tmp_app_config.paths.resolve("processed_data_dir") / "preprocessed_splits.npz"
    arrays = dict(np.load(npz_path))
    arrays["y_test"] = (arrays["y_test"] + 1) % int(arrays["y_test"].max() + 1)
    arrays["X_test_scaled"] = np.zeros_like(arrays["X_test_scaled"])
    np.savez(npz_path, **arrays)
    res_b = refit_xgboost_downstream_artifacts(tmp_app_config, m_cfg, compute_shap=False)
    assert res_b["xgboost_temperature"] == res_a["xgboost_temperature"]
    assert res_b["stacking_summary"] == res_a["stacking_summary"]

    res_c = refit_xgboost_downstream_artifacts(tmp_app_config, m_cfg, compute_shap=True)
    assert res_c["shap_refreshed"] is True
    shap_after = json.loads(shap_path.read_text(encoding="utf-8"))
    assert shap_after["MLP"] == {"explainer_type": "sentinel"}
    assert shap_after["XGBoost"]["profile"] == "class_weight"
    assert shap_after["XGBoost"]["top_features"]
    assert "local_example" in shap_after

    diag = json.loads((artifacts_dir / "training_diagnostics.json").read_text(encoding="utf-8"))
    assert diag["downstream_refit"]["used_test_data_for_fitting"] is False
    assert "XGBoost" in diag["calibration_report"]
    manifest = json.loads((models_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert "calibrators.pkl" in manifest["artifacts"]
    assert "stacking_meta.pkl" in manifest["artifacts"]
