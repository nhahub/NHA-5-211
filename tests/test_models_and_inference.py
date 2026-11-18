"""Tests for all four model architectures, saving/loading, ensemble, and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AppConfig, ModelConfig
from src.data.loader import generate_synthetic_cicids2017
from src.data.preprocessing import run_preprocessing_pipeline
from src.detection.anomaly_detection import AnomalyDetector
from src.detection.inference import UnifiedDetectionEngine
from src.evaluation.protocols import evaluate_by_day_protocol, evaluate_feature_ablation
from src.evaluation.report_writer import write_academic_reports
from src.models.autoencoder import (
    build_autoencoder,
    calibrate_threshold,
    compute_reconstruction_error,
    extract_benign_subset,
)
from src.models.cnn import build_cnn1d
from src.models.ensemble import HybridEnsembleEngine
from src.models.mlp import build_mlp
from src.models.xgboost_model import build_xgboost_classifier
from src.response.response_engine import ActiveResponseEngine
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
    assert ablation_res["with_features"]["n_features"] == 77
    assert ablation_res["without_features"]["n_features"] == 76
    assert "delta_test_f1_macro" in ablation_res
    wo_shap_names = {f["feature"] for f in ablation_res["without_features"]["top_shap_features"]}
    assert "Destination Port" not in wo_shap_names

    reports_dir = tmp_app_config.paths.resolve("reports_dir")
    cmp_df = pd.DataFrame([{"Model": "XGBoost", "Accuracy": 0.99, "F1_Weighted": 0.99, "F1_Macro": 0.95}])
    summary = {
        "dataset_source": {"is_synthetic": False},
        "split_strategy": "random",
        "by_day_protocol": by_day_res,
        "feature_ablation": ablation_res,
        "preprocessing_audit": {"split_strategy": "random", "dropped_features": []},
        "training_diagnostics": {},
    }
    cmp_path, _ = write_academic_reports(summary, cmp_df, reports_dir)
    md_text = cmp_path.read_text(encoding="utf-8")
    assert "Cross-Day Temporal Split Protocol (`by_day`" in md_text
    assert "Cross-Day Binary Intrusion Detection (`by_day`: Benign vs. Attack)" in md_text
    assert "Shortcut Feature Ablation (With vs. Without `Destination Port`)" in md_text
    assert "Delta Test Macro-F1 (without - with):" in md_text
