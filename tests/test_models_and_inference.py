"""Tests for all four model architectures, saving/loading, ensemble, and inference."""

from __future__ import annotations

import copy
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
from src.detection.anomaly_detection import AnomalyDetector, fit_classical_anomaly_baselines
from src.detection.inference import UnifiedDetectionEngine
from src.evaluation.metrics import (
    bootstrap_macro_f1_ci,
    evaluate_anomaly_detector,
    evaluate_supervised_model,
    paired_bootstrap_macro_f1_test,
)
from src.evaluation.protocols import (
    assert_ablation_row_parity,
    evaluate_by_day_protocol,
    evaluate_feature_ablation,
    evaluate_time_block_protocol,
    run_xgboost_stratified_cv,
)
from src.evaluation.report_writer import format_ensemble_empirical_conclusion, write_academic_reports
from src.models.autoencoder import (
    build_autoencoder,
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
from src.training.trainer import train_all_models
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
