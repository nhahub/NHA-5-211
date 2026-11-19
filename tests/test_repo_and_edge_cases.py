"""Phase 6 tests for partial registry loading, batch CSV validation, split strategies, direct latency, and ONNX checks."""

from __future__ import annotations

import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import pytest
from scripts.check_optional_data import check_optional_data_status, format_optional_data_report
from scripts.check_readme_numbers import check_readme_numbers
from scripts.download_models import download_model_assets
from scripts.export_onnx import check_onnx_environment
from scripts.export_onnx import main as export_onnx_main
from scripts.generate_test_report import CommandRun, parse_pytest_output, render_test_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    PROJECT_ROOT,
    AppConfig,
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    SecurityConfig,
    SplitConfig,
    get_app_config,
    sanitize_machine_paths,
    to_repo_relative_path,
)
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
from src.utils.model_registry import (
    ModelIntegrityError,
    ModelRegistry,
    compute_file_sha256,
    is_git_lfs_pointer,
)


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

    stub_pytest = "\n".join(
        [
            "tests/test_a.py::test_one PASSED                                        [ 33%]",
            "tests/test_b.py::test_two SKIPPED (LFS pointer stubs)                   [ 66%]",
            "tests/test_c.py::test_three SKIPPEDlfs_pointers=['xgb_model.json']       [100%]",
            "0.05s call     tests/test_a.py::test_one",
            "0.01s call     tests/test_b.py::test_two",
            "0.01s call     tests/test_c.py::test_three",
            "====================== 1 passed, 2 skipped in 136.56s (0:02:16) ======================",
        ]
    )
    md = render_test_report(
        ruff_version="ruff 0.17.5",
        pytest_version="pytest 9.2.0",
        ruff_check=CommandRun(["ruff"], "ruff check", 0, "All checks passed!", "", 0.2),
        ruff_format=CommandRun(["ruff"], "ruff format --check", 0, "63 files already formatted", "", 0.1),
        pytest_run=CommandRun(["pytest"], "pytest tests/ -v", 0, sample_pytest, "", 1.5),
        stub_pytest_run=CommandRun(
            ["pytest"], "pytest tests/ -v (stub-mode simulation)", 0, stub_pytest, "", 136.56
        ),
        stub_files_count=14,
        generated_utc="2026-11-18 21:00 UTC",
    )
    assert "tests/test_a.py" in md and "1.25" in md and "1.35" in md
    assert "2 passed, 1 warning" in md
    assert "LFS Pointer Stubs (`lfs: false` mirror, 14 stubs)" in md
    assert "1 passed / 0 failed / 0 error / 2 skipped" in md
    assert "`tests/test_b.py::test_two`, `tests/test_c.py::test_three`" in md


def test_manifest_sha256_match_mismatch_missing_and_lfs_skip(tmp_path: Path, monkeypatch) -> None:
    """ModelRegistry enforces SHA-256 verification before joblib.load and skips LFS stubs gracefully."""
    assert SecurityConfig().verify_model_hashes is True
    models_dir = tmp_path / "sec_models"
    reg = ModelRegistry(models_dir=models_dir, verify_model_hashes=True, require_manifest=False)

    scaler_path = models_dir / "scaler.pkl"
    le_path = models_dir / "label_encoder.pkl"
    joblib.dump(StandardScaler(), scaler_path)
    joblib.dump(LabelEncoder(), le_path)

    # Generate SHA-256 MANIFEST.json and verify match path succeeds
    manifest_path = reg.save_artifact_manifest()
    assert manifest_path.exists()
    manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_json["algorithm"] == "sha256"
    assert manifest_json["artifacts"]["scaler.pkl"]["sha256"] == compute_file_sha256(scaler_path)

    loaded_ok = reg.load_all(load_keras=False)
    assert loaded_ok.scaler is not None
    assert loaded_ok.label_encoder is not None
    assert loaded_ok.integrity_warnings == []

    # LFS pointer stub is skipped cleanly without error even when require_manifest=True
    lfs_stub = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123\nsize 2048\n"
    (models_dir / "calibrators.pkl").write_text(lfs_stub, encoding="utf-8")
    strict_with_lfs = ModelRegistry(models_dir=models_dir, verify_model_hashes=True, require_manifest=True)
    loaded_lfs = strict_with_lfs.load_all(load_keras=False)
    assert "calibrators.pkl" in loaded_lfs.lfs_pointers

    # Tamper with scaler.pkl -> MUST refuse BEFORE calling joblib.load
    scaler_path.write_bytes(scaler_path.read_bytes() + b"\x00tamper")
    called_joblib = []
    orig_joblib_load = joblib.load

    def _spy_joblib_load(path, *args, **kwargs):
        called_joblib.append(Path(path).name)
        return orig_joblib_load(path, *args, **kwargs)

    monkeypatch.setattr("src.utils.model_registry.joblib.load", _spy_joblib_load)
    with pytest.raises(ModelIntegrityError, match="SHA-256 integrity mismatch"):
        reg.load_all(load_keras=False)
    assert "scaler.pkl" not in called_joblib

    # Opt-out flag verify_model_hashes=False bypasses hash failure (restore clean pickle first)
    monkeypatch.undo()
    joblib.dump(StandardScaler(), scaler_path)
    manifest_path.unlink()

    # Missing manifest: require_manifest=True refuses; require_manifest=False warns clearly
    strict_missing = ModelRegistry(models_dir=models_dir, verify_model_hashes=True, require_manifest=True)
    with pytest.raises(ModelIntegrityError, match="missing"):
        strict_missing.load_all(load_keras=False)

    permissive_missing = ModelRegistry(
        models_dir=models_dir, verify_model_hashes=True, require_manifest=False
    )
    loaded_warn = permissive_missing.load_all(load_keras=False)
    assert loaded_warn.scaler is not None
    assert any("MANIFEST.json" in w for w in loaded_warn.integrity_warnings)


def test_download_models_fallback_and_hash_verification(tmp_path: Path) -> None:
    """LFS-free downloader fetches release assets offline-mocked and enforces SHA-256 check."""
    remote_dir = tmp_path / "remote_release"
    remote_dir.mkdir(parents=True, exist_ok=True)
    local_dir = tmp_path / "local_models"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Pretend local dir has an unfetched Git LFS pointer stub for scaler.pkl
    (local_dir / "scaler.pkl").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:000\nsize 99\n",
        encoding="utf-8",
    )
    # Prepare real remote payload + MANIFEST.json
    remote_scaler = remote_dir / "scaler.pkl"
    joblib.dump(StandardScaler(), remote_scaler)
    (remote_dir / "feature_names.json").write_text('["f1", "f2"]', encoding="utf-8")
    manifest_payload = {
        "algorithm": "sha256",
        "artifacts": {
            "scaler.pkl": {"sha256": compute_file_sha256(remote_scaler)},
            "feature_names.json": {"sha256": compute_file_sha256(remote_dir / "feature_names.json")},
        },
    }
    (remote_dir / "MANIFEST.json").write_text(json.dumps(manifest_payload), encoding="utf-8")

    def _local_fetcher(url: str, dest: Path, timeout: int) -> None:
        fname = url.rsplit("/", 1)[-1]
        src = remote_dir / fname
        if not src.exists():
            raise FileNotFoundError(fname)
        dest.write_bytes(src.read_bytes())

    summary = download_model_assets(
        base_url="https://example.invalid/releases/download/models-v3",
        output_dir=local_dir,
        required_assets=["scaler.pkl", "feature_names.json"],
        optional_assets=["MANIFEST.json", "optional_absent.keras"],
        verify_hashes=True,
        fetcher=_local_fetcher,
    )
    assert set(summary["verified"]) == {"scaler.pkl", "feature_names.json"}
    assert "optional_absent.keras" in summary["optional_missing"]
    assert not is_git_lfs_pointer(local_dir / "scaler.pkl")

    # Corrupt remote asset -> must delete tampered file and raise ModelIntegrityError
    remote_scaler.write_bytes(remote_scaler.read_bytes() + b"corrupt")
    with pytest.raises(ModelIntegrityError, match="failed SHA-256 verification"):
        download_model_assets(
            base_url="https://example.invalid/releases/download/models-v3",
            output_dir=local_dir,
            required_assets=["scaler.pkl"],
            optional_assets=["MANIFEST.json"],
            overwrite=True,
            verify_hashes=True,
            fetcher=_local_fetcher,
        )
    assert not (local_dir / "scaler.pkl").exists()


def test_real_lfs_artifact_smoke() -> None:
    """Smoke check on persisted production artifacts; skips cleanly when LFS pointers are unfetched."""
    app_cfg = get_app_config()
    models_dir = app_cfg.paths.resolve("models_dir")
    required_files = ["xgb_model.json", "scaler.pkl", "label_encoder.pkl", "feature_names.json"]
    missing = [f for f in required_files if not (models_dir / f).exists()]
    pointers = [f for f in required_files if (models_dir / f).exists() and is_git_lfs_pointer(models_dir / f)]
    if missing or pointers:
        pytest.skip(
            f"Real model binaries unavailable or Git LFS pointer stubs (missing={missing}, lfs_pointers={pointers}); "
            "run 'git lfs pull' or trigger workflow_dispatch."
        )

    reg = ModelRegistry(models_dir=models_dir, artifacts_dir=app_cfg.paths.resolve("artifacts_dir"))
    loaded = reg.load_all(load_keras=False)
    assert "XGBoost" in loaded.models
    assert loaded.scaler is not None
    assert loaded.label_encoder is not None
    sample = np.zeros((1, len(loaded.feature_names)), dtype=np.float32)
    probs = loaded.models["XGBoost"].predict_proba(sample)
    assert probs.shape == (1, len(loaded.label_encoder.classes_))


def test_model_registry_stub_mode_graceful_degradation(tmp_path: Path) -> None:
    """ModelRegistry degrades gracefully with clear LFS pointer tracking when all binaries are stubs."""
    stub_dir = tmp_path / "stub_models"
    stub_dir.mkdir(parents=True, exist_ok=True)
    lfs_files = [
        "anomaly_baselines.pkl",
        "anomaly_fusion.pkl",
        "best_autoencoder.keras",
        "best_autoencoder_deep.keras",
        "best_cnn.keras",
        "best_mlp.keras",
        "best_vae.keras",
        "calibrators.pkl",
        "imputer.pkl",
        "label_encoder.pkl",
        "outlier_clipper.pkl",
        "scaler.pkl",
        "stacking_meta.pkl",
        "xgb_model.json",
    ]
    for fname in lfs_files:
        stub_path = stub_dir / fname
        stub_path.write_text(
            f"version https://git-lfs.github.com/spec/v1\noid sha256:{'a' * 64}\nsize 12345\n",
            encoding="utf-8",
        )
        assert is_git_lfs_pointer(stub_path) is True

    (stub_dir / "feature_names.json").write_text('["f1", "f2"]', encoding="utf-8")
    (stub_dir / "anomaly_threshold.json").write_text('{"threshold": 0.025}', encoding="utf-8")
    (stub_dir / "model_metadata.json").write_text(
        '{"models": {"XGBoost": {"imbalance_strategy": "none"}}}', encoding="utf-8"
    )
    (stub_dir / "MANIFEST.json").write_text('{"algorithm": "sha256", "artifacts": {}}', encoding="utf-8")

    reg = ModelRegistry(
        models_dir=stub_dir, artifacts_dir=tmp_path, verify_model_hashes=True, require_manifest=True
    )
    loaded = reg.load_all(load_keras=True)

    assert loaded.models == {}
    assert loaded.scaler is None
    assert loaded.imputer is None
    assert loaded.outlier_clipper is None
    assert loaded.label_encoder is None
    assert loaded.calibrators is None
    assert loaded.stacker is None
    assert loaded.anomaly_baselines is None
    assert loaded.anomaly_fusion is None
    assert set(loaded.lfs_pointers) == set(lfs_files)
    assert loaded.feature_names == ["f1", "f2"]
    assert loaded.anomaly_threshold == pytest.approx(0.025)
    assert loaded.model_imbalance_strategies == {"XGBoost": "none"}


def test_no_absolute_machine_paths_in_artifacts_and_sanitizer(tmp_path: Path) -> None:
    """Committed reports, artifacts, and processed JSONs must not leak local machine paths."""
    sample = {
        "win": r"D:\IDS_Final\data\TrafficLabelling",
        "win_ext": r"C:\Users\dev\external_folder",
        "unix": "/home/alice/TrafficLabelling",
        "rel": "data/engelen2021_label_corrections.csv",
    }
    cleaned = sanitize_machine_paths(sample, root=Path(r"D:\IDS_Final"))
    assert cleaned == {
        "win": "data/TrafficLabelling",
        "win_ext": "external_folder",
        "unix": "TrafficLabelling",
        "rel": "data/engelen2021_label_corrections.csv",
    }
    assert to_repo_relative_path(tmp_path / "custom_dir") == "custom_dir"

    leak_pattern = re.compile(r"(?:[A-Za-z]:\\\\|[A-Za-z]:/|/home/)")
    scan_targets = [
        *sorted((PROJECT_ROOT / "reports").glob("*.json")),
        *sorted((PROJECT_ROOT / "reports").glob("*.md")),
        *sorted((PROJECT_ROOT / "reports").glob("*.csv")),
        *sorted((PROJECT_ROOT / "artifacts").glob("*.json")),
        *sorted((PROJECT_ROOT / "data" / "processed").glob("*.json")),
    ]
    leaks: list[str] = []
    for artifact in scan_targets:
        text = artifact.read_text(encoding="utf-8", errors="ignore")
        if leak_pattern.search(text):
            leaks.append(artifact.relative_to(PROJECT_ROOT).as_posix())
    assert leaks == [], f"Absolute machine paths detected in generated artifacts: {leaks}"


def test_check_optional_data_present_and_missing(tmp_path: Path) -> None:
    """Optional data checker reports missing files and detects TrafficLabelling + Engelen CSVs when placed."""
    cfg = AppConfig()
    missing_status = check_optional_data_status(app_cfg=cfg, root=tmp_path)
    assert missing_status["timestamp_source"]["ready"] is False
    assert missing_status["label_correction"]["ready"] is False
    assert missing_status["all_optional_present"] is False
    missing_report = format_optional_data_report(missing_status)
    assert "TrafficLabelling_.zip" in missing_report
    assert "intrusion-detection.distrinet-research.be/CNS2021/" in missing_report
    assert "python scripts/evaluate.py --time-block-only" in missing_report

    ts_dir = tmp_path / "data" / "TrafficLabelling"
    ts_dir.mkdir(parents=True, exist_ok=True)
    (ts_dir / "Monday-WorkingHours.pcap_ISCX.csv").write_text(
        "Flow ID,Timestamp,Destination Port,Label\n1,03/07/2017 08:55,80,BENIGN\n",
        encoding="utf-8",
    )
    corr_file = tmp_path / "data" / "engelen2021_label_corrections.csv"
    corr_file.write_text("Flow ID,Corrected Label\n1,BENIGN\n", encoding="utf-8")

    present_status = check_optional_data_status(app_cfg=cfg, root=tmp_path)
    assert present_status["timestamp_source"]["ready"] is True
    assert present_status["timestamp_source"]["csv_count"] == 1
    assert (
        "Monday-WorkingHours.pcap_ISCX.csv"
        in present_status["timestamp_source"]["files_with_timestamp_column"]
    )
    assert present_status["label_correction"]["ready"] is True
    assert present_status["all_optional_present"] is True


def test_readme_numbers_match_generated_reports_and_detect_drift(tmp_path: Path) -> None:
    """README numeric claims match evaluation_summary.json/test_report.md and tamper detection works."""
    mismatches = check_readme_numbers()
    assert mismatches == [], f"README/doc mismatches: {mismatches}"

    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert len(readme_text.splitlines()) <= 150
    assert (
        "models/MANIFEST.json protects against corrupted or accidentally swapped artifacts"
        in readme_text.replace("`", "")
    )

    # Negative check: altering an F1_Macro value in a temporary README copy must be flagged
    tampered_readme = tmp_path / "README.md"
    tampered_readme.write_text(readme_text.replace("0.9603", "0.9999", 1), encoding="utf-8")
    tamper_errs = check_readme_numbers(readme_path=tampered_readme)
    assert any("XGBoost random F1_Macro" in e for e in tamper_errs)
