"""Phase 6 tests for partial registry loading, batch CSV validation, split strategies, direct latency, and ONNX checks."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scripts.export_onnx import check_onnx_environment
from scripts.export_onnx import main as export_onnx_main
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import AppConfig, DataConfig
from src.data.feature_engineering import SchemaValidationError
from src.data.loader import CICIDS2017_FEATURES, generate_synthetic_cicids2017
from src.data.preprocessing import (
    _classify_day_bucket,
    _split_indices_by_day,
    run_preprocessing_pipeline,
)
from src.detection.inference import UnifiedDetectionEngine
from src.evaluation.benchmarking import benchmark_model_inference
from src.models.mlp import build_mlp
from src.models.xgboost_model import build_xgboost_classifier
from src.response.response_engine import ActiveResponseEngine
from src.utils.model_registry import ModelRegistry


def test_model_registry_partial_and_corrupt_artifacts(
    tmp_app_config: AppConfig, small_synthetic_df: pd.DataFrame
) -> None:
    """ModelRegistry loads surviving artifacts cleanly when Keras files are missing or corrupt."""
    dataset = run_preprocessing_pipeline(tmp_app_config, force_synthetic=True, save_outputs=True)
    models_dir = Path(tmp_app_config.paths.models_dir)
    artifacts_dir = Path(tmp_app_config.paths.artifacts_dir)

    # Train and save only XGBoost; omit calibrators, stacking_meta, anomaly_baselines, threshold JSON
    xgb_clf = build_xgboost_classifier(
        len(dataset.class_names),
        params_override={"n_estimators": 8, "max_depth": 3},
        device="cpu",
        early_stopping_rounds=None,
    )
    xgb_clf.fit(dataset.X_train_sm, dataset.y_train_sm)
    xgb_clf.save_model(str(models_dir / "xgb_model.json"))

    # Write a corrupt Keras checkpoint alongside missing CNN/AE files
    (models_dir / "best_mlp.keras").write_text("not-a-valid-keras-archive", encoding="utf-8")

    registry = ModelRegistry(models_dir=models_dir, artifacts_dir=artifacts_dir)
    loaded = registry.load_all(load_keras=True)

    assert "XGBoost" in loaded.models
    assert "MLP" not in loaded.models
    assert "CNN" not in loaded.models
    assert "Autoencoder" not in loaded.models
    assert loaded.scaler is not None
    assert loaded.label_encoder is not None
    assert len(loaded.feature_names) == len(dataset.feature_names)
    assert loaded.anomaly_threshold == 1.0
    assert loaded.calibrators is None
    assert loaded.stacker is None
    assert loaded.anomaly_baselines is None

    # Disabling Keras loading also works with partial directory
    loaded_no_keras = registry.load_all(load_keras=False)
    assert list(loaded_no_keras.models.keys()) == ["XGBoost"]

    # Missing preprocessing metadata files fall back gracefully on load_all
    bare_dir = models_dir / "bare_partial"
    bare_reg = ModelRegistry(models_dir=bare_dir, artifacts_dir=bare_dir)
    joblib.dump(StandardScaler(), bare_reg.models_dir / "scaler.pkl")
    joblib.dump(LabelEncoder(), bare_reg.models_dir / "label_encoder.pkl")
    bare_loaded = bare_reg.load_all(load_keras=False)
    assert bare_loaded.models == {}
    assert bare_loaded.feature_names == []
    assert bare_loaded.scaler is not None


def test_batch_csv_scoring_schema_validation_and_reordering(
    tmp_app_config: AppConfig, small_synthetic_df: pd.DataFrame, tmp_path: Path
) -> None:
    """UnifiedDetectionEngine validates batch CSV columns, rejects missing schema, and tolerates reordering."""
    dataset = run_preprocessing_pipeline(tmp_app_config, force_synthetic=True, save_outputs=True)
    models_dir = Path(tmp_app_config.paths.models_dir)

    xgb_clf = build_xgboost_classifier(
        len(dataset.class_names),
        params_override={"n_estimators": 8, "max_depth": 3},
        device="cpu",
        early_stopping_rounds=None,
    )
    xgb_clf.fit(dataset.X_train_sm, dataset.y_train_sm)
    xgb_clf.save_model(str(models_dir / "xgb_model.json"))

    registry = ModelRegistry(models_dir=models_dir, artifacts_dir=Path(tmp_app_config.paths.artifacts_dir))
    loaded = registry.load_all(load_keras=False)
    engine = UnifiedDetectionEngine(
        loaded,
        response_engine=ActiveResponseEngine(
            dry_run=True,
            log_file=tmp_path / "resp.log",
            history_file=tmp_path / "resp.json",
        ),
    )

    # Simulate an uploaded batch CSV with reversed column order, whitespace headers, Inf, and non-numeric cells
    raw_batch = small_synthetic_df.head(5).copy()
    raw_batch["Flow Bytes/s"] = raw_batch["Flow Bytes/s"].astype(object)
    raw_batch.loc[0, "Flow Bytes/s"] = "Infinity"
    raw_batch.loc[1, "Flow Bytes/s"] = "corrupted_text"
    reversed_cols = list(reversed(raw_batch.columns.tolist()))
    raw_batch = raw_batch[reversed_cols]
    raw_batch.columns = [f" {c} " for c in raw_batch.columns]

    csv_path = tmp_path / "batch_upload.csv"
    raw_batch.to_csv(csv_path, index=False)
    uploaded_df = pd.read_csv(csv_path)

    out_df, results = engine.predict_batch(uploaded_df, model_choice="XGBoost", explain_top_k=3)
    assert len(out_df) == 5
    assert len(results) == 5
    assert set(["attack_class", "confidence", "threat_level", "action"]).issubset(out_df.columns)
    assert all(res.predicted_class in dataset.class_names for res in results)

    # Missing required CICIDS2017 columns must raise SchemaValidationError
    missing_cols_df = uploaded_df.drop(columns=[uploaded_df.columns[0], uploaded_df.columns[1]])
    with pytest.raises(SchemaValidationError, match="missing"):
        engine.predict_batch(missing_cols_df, model_choice="XGBoost")

    # Zero-row uploaded CSV must raise SchemaValidationError
    empty_df = pd.DataFrame(columns=CICIDS2017_FEATURES)
    with pytest.raises(SchemaValidationError, match="zero rows"):
        engine.predict_batch(empty_df, model_choice="XGBoost")


def test_split_strategy_validation_and_error_paths(tmp_app_config: AppConfig) -> None:
    """Split strategy config validates literals and by_day handles missing metadata/unknown filenames cleanly."""
    with pytest.raises(ValueError):
        DataConfig(split_strategy="temporal_invalid")

    assert _classify_day_bucket("Monday-WorkingHours.pcap_ISCX.csv") == "train_mon_wed"
    assert _classify_day_bucket("Tuesday-WorkingHours.pcap_ISCX.csv") == "train_mon_wed"
    assert _classify_day_bucket("Wednesday-workingHours.pcap_ISCX.csv") == "train_mon_wed"
    assert _classify_day_bucket("Thursday-WorkingHours-Morning.csv") == "thursday"
    assert _classify_day_bucket("Friday-WorkingHours-Afternoon-DDos.csv") == "friday"

    with pytest.raises(ValueError, match="Monday-Friday"):
        _classify_day_bucket("Saturday-Capture.csv")

    y_dummy = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="Source_File"):
        _split_indices_by_day(pd.DataFrame({"Other": [1] * 6}), y_dummy, 0.35, True, 42)

    missing_thu = pd.DataFrame(
        {"Source_File": ["Monday.csv", "Monday.csv", "Friday.csv", "Friday.csv", "Friday.csv", "Friday.csv"]}
    )
    with pytest.raises(ValueError, match="non-empty Monday-Wednesday train, Thursday validation"):
        _split_indices_by_day(missing_thu, y_dummy, 0.35, False, 42)

    # Synthetic corpus without raw weekday CSV filenames rejects by_day split explicitly
    tmp_app_config.data.split_strategy = "by_day"
    with pytest.raises(ValueError):
        run_preprocessing_pipeline(tmp_app_config, force_synthetic=True, save_outputs=False)

    # Disjointness & target fractions on multi-day synthetic raw CSVs
    raw_dir = tmp_app_config.paths.resolve("raw_data_dir")
    generate_synthetic_cicids2017(n_samples=150, seed=1, save_path=raw_dir / "Monday-WorkingHours.csv")
    generate_synthetic_cicids2017(n_samples=150, seed=2, save_path=raw_dir / "Thursday-WorkingHours.csv")
    generate_synthetic_cicids2017(n_samples=150, seed=3, save_path=raw_dir / "Friday-WorkingHours.csv")

    dataset_day = run_preprocessing_pipeline(tmp_app_config, force_synthetic=False, save_outputs=False)
    assert dataset_day.audit_report["split_strategy"] == "by_day"
    assert len(dataset_day.X_train_scaled) > 0
    assert len(dataset_day.X_val_scaled) > 0
    assert len(dataset_day.X_test_scaled) > 0


def test_direct_single_flow_benchmark_and_onnx_export_script(tmp_path: Path) -> None:
    """Warmed single-flow benchmarking and ONNX export feasibility checks run without errors."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(32, 10)).astype(np.float32)
    y = rng.integers(0, 3, size=32)

    mlp = build_mlp(input_dim=10, num_classes=3, hidden_units=[16, 8], dropout_rate=0.0)
    mlp.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    mlp.fit(X, y, epochs=1, batch_size=16, verbose=0)
    mlp_bench = benchmark_model_inference(
        mlp,
        X,
        model_type="keras",
        n_warmup=2,
        n_runs=5,
        batch_size=16,
    )
    assert mlp_bench["single_mean_us"] > 0.0
    assert mlp_bench["single_p95_us"] >= mlp_bench["single_median_us"]
    assert mlp_bench["batch_samples_per_sec"] > 0.0

    # Verify early-stopped XGBClassifier matches inplace_predict in benchmark_model_inference cleanly
    xgb_clf = build_xgboost_classifier(
        num_classes=3,
        params_override={"n_estimators": 12, "max_depth": 2},
        device="cpu",
        early_stopping_rounds=2,
    )
    xgb_clf.fit(X[:24], y[:24], eval_set=[(X[24:], y[24:])], verbose=False)
    xgb_bench = benchmark_model_inference(
        xgb_clf,
        X,
        model_type="xgb",
        n_warmup=2,
        n_runs=5,
        batch_size=16,
    )
    assert xgb_bench["single_mean_us"] > 0.0
    assert xgb_bench["batch_samples_per_sec"] > 0.0

    # ONNX exporter reports clear status whether optional tf2onnx/onnx are installed or absent
    env = check_onnx_environment()
    assert "available" in env and "python" in env
    missing_req = [p for p in ("onnx", "tf2onnx") if not env["available"].get(p, False)]
    (tmp_path / "feature_names.json").write_text(json.dumps([f"f{i}" for i in range(10)]), encoding="utf-8")

    exit_non_strict = export_onnx_main(["--models-dir", str(tmp_path), "--out-dir", str(tmp_path / "onnx")])
    assert exit_non_strict == (0 if missing_req else 1)  # empty models_dir fails only if deps exist
    if missing_req:
        assert export_onnx_main(["--models-dir", str(tmp_path), "--strict"]) == 1
