"""Tests for all four model architectures, saving/loading, ensemble, and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.data.preprocessing import run_preprocessing_pipeline
from src.detection.anomaly_detection import AnomalyDetector
from src.detection.inference import UnifiedDetectionEngine
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
