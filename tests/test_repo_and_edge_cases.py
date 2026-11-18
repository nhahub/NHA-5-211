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
from scripts.generate_test_report import CommandRun, parse_pytest_output, render_test_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import AppConfig, DataConfig, EvaluationConfig, ModelConfig, SplitConfig
from src.data.feature_engineering import SchemaValidationError
from src.data.loader import CICIDS2017_FEATURES, generate_synthetic_cicids2017
from src.data.preprocessing import (
    _classify_day_bucket,
    _split_indices_by_day,
    _split_indices_time_block,
    run_preprocessing_pipeline,
)
from src.detection.inference import UnifiedDetectionEngine
from src.evaluation.benchmarking import benchmark_model_inference
from src.evaluation.protocols import evaluate_feature_ablation
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

    # Unfetched Git LFS pointer stubs and corrupt .pkl files never crash load_all
    lfs_dir = models_dir / "lfs_stubs"
    lfs_reg = ModelRegistry(models_dir=lfs_dir, artifacts_dir=lfs_dir)
    lfs_stub = "version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 1024\n"
    for fname in ("best_mlp.keras", "xgb_model.json", "anomaly_baselines.pkl"):
        (lfs_dir / fname).write_text(lfs_stub, encoding="utf-8")
    (lfs_dir / "scaler.pkl").write_text("corrupt-pickle-bytes", encoding="utf-8")

    lfs_loaded = lfs_reg.load_all(load_keras=True)
    assert lfs_loaded.models == {}
    assert lfs_loaded.scaler is None
    assert lfs_loaded.anomaly_baselines is None
    assert set(lfs_loaded.lfs_pointers) == {"anomaly_baselines.pkl", "best_mlp.keras", "xgb_model.json"}


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
    """Split strategy config validates literals and by_day/time_block handle missing metadata cleanly."""
    with pytest.raises(ValueError):
        DataConfig(split_strategy="temporal_invalid")
    assert DataConfig(split_strategy="time_block").split_strategy == "time_block"
    with pytest.raises(ValueError, match="time_block_train_fraction"):
        SplitConfig(time_block_train_fraction=0.80, time_block_val_fraction=0.20)

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
    with pytest.raises(ValueError, match="Timestamp"):
        _split_indices_time_block(pd.DataFrame({"Other": [1] * 6}), y_dummy, ["BENIGN", "Attack"])

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


def test_ablation_preserves_exact_rows_even_with_port_only_differences(tmp_app_config: AppConfig) -> None:
    """Flows that differ solely on Destination Port keep identical train/val/test rows in ablation."""
    base_df = generate_synthetic_cicids2017(n_samples=240, seed=99, save_path=None)
    # Inject 40 flows whose 76 non-port features are identical and only Destination Port differs.
    port_clones = base_df.iloc[:40].copy().reset_index(drop=True)
    port_clones["Destination Port"] = np.arange(10000, 10040, dtype=np.int64)
    for col in port_clones.columns:
        if col not in ("Destination Port", "Label", "Source IP", "Destination IP", "Timestamp"):
            port_clones[col] = base_df.loc[0, col]

    combined = pd.concat([base_df, port_clones], ignore_index=True)
    raw_dir = tmp_app_config.paths.resolve("raw_data_dir")
    combined.to_csv(raw_dir / "Friday-PortOverlap-Test.csv", index=False)

    model_cfg = ModelConfig()
    res = evaluate_feature_ablation(
        app_cfg=tmp_app_config,
        model_cfg=model_cfg,
        drop_features=["Destination Port"],
        xgb_params={"n_estimators": 8, "max_depth": 3},
        imbalance_strategy="none",
        force_synthetic=False,
        compute_shap=False,
    )
    assert res["executed"] is True
    assert res["identical_test_rows_verified"] is True
    assert res["with_features"]["split_sizes"]["test"] == res["without_features"]["split_sizes"]["test"]
    assert (
        res["with_features"]["split_sizes"]["train_raw"]
        == res["without_features"]["split_sizes"]["train_raw"]
    )
    assert res["with_features"]["test_rows_sha256"] == res["without_features"]["test_rows_sha256"]


def test_latency_warmup_config_and_test_report_generator_parser() -> None:
    """latency_warmup_calls defaults to 50 and generate_test_report parses real pytest output."""
    assert EvaluationConfig().latency_warmup_calls == 50
    assert EvaluationConfig(latency_warmup_runs=12).latency_warmup_calls == 12

    rng = np.random.default_rng(11)
    X = rng.normal(size=(24, 6)).astype(np.float32)
    y = rng.integers(0, 3, size=24)
    clf = build_xgboost_classifier(
        num_classes=3,
        params_override={"n_estimators": 6, "max_depth": 2},
        device="cpu",
        early_stopping_rounds=None,
    )
    clf.fit(X, y)
    bench = benchmark_model_inference(clf, X, model_type="xgb", n_warmup=50, n_runs=10, batch_size=8)
    assert bench["warmup_calls_discarded"] == 50
    assert bench["single_median_us"] > 0.0
    assert bench["single_p95_us"] >= bench["single_median_us"]

    sample_pytest = "\n".join(
        [
            "tests/test_a.py::test_one PASSED                                        [ 50%]",
            "tests/test_b.py::test_two PASSED                                        [100%]",
            "============================== slowest durations ==============================",
            "1.25s call     tests/test_a.py::test_one",
            "0.10s setup    tests/test_a.py::test_one",
            "0.04s call     tests/test_b.py::test_two",
            "====================== 2 passed, 1 warning in 1.50s ======================",
        ]
    )
    cases, summary_txt, dur, counts = parse_pytest_output(sample_pytest)
    assert len(cases) == 2
    assert cases[0].call_sec == pytest.approx(1.25)
    assert cases[0].total_sec == pytest.approx(1.35)
    assert counts["passed"] == 2
    assert dur == pytest.approx(1.50)
    assert "2 passed" in summary_txt

    md = render_test_report(
        ruff_version="ruff 0.17.5",
        pytest_version="pytest 9.2.0",
        ruff_check=CommandRun(["ruff"], "ruff check", 0, "All checks passed!", "", 0.2),
        ruff_format=CommandRun(["ruff"], "ruff format --check", 0, "63 files already formatted", "", 0.1),
        pytest_run=CommandRun(["pytest"], "pytest tests/ -v", 0, sample_pytest, "", 1.5),
        generated_utc="2026-11-18 21:00 UTC",
    )
    assert "tests/test_a.py" in md and "1.25" in md and "1.35" in md
    assert "2 passed, 1 warning" in md
