"""Tests for CSV discovery, cleaning, schema alignment, and zero-leakage preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import AppConfig
from src.data.cleaner import clean_cicids_dataframe, normalize_label_text
from src.data.feature_engineering import SchemaValidationError, align_feature_schema
from src.data.loader import CICIDS2017_FEATURES, discover_csv_files, generate_synthetic_cicids2017
from src.data.preprocessing import (
    DataLeakageError,
    _split_indices_time_block,
    apply_smote_safely,
    compute_hybrid_smote_class_weights,
    run_preprocessing_pipeline,
    verify_no_leakage,
)


def test_discover_csv_files_and_whitespace_stripping(
    tmp_path: Path, small_synthetic_df: pd.DataFrame
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    spaced = small_synthetic_df.head(20).copy()
    spaced.columns = [f" {c} " for c in spaced.columns]
    csv_path = raw_dir / "Friday-Test.csv"
    spaced.to_csv(csv_path, index=False)

    found = discover_csv_files(raw_dir)
    assert found == [csv_path]

    _, y_series, _, report = clean_cicids_dataframe(pd.read_csv(csv_path))
    assert len(y_series) > 0
    assert "Source IP" in report.id_columns_removed
    assert "Destination Port" not in report.id_columns_removed


def test_cleaning_inf_nan_duplicates_and_web_attack_mojibake(small_synthetic_df: pd.DataFrame) -> None:
    df = small_synthetic_df.head(120).copy()
    # Inject Web Attack mojibake labels, duplicate row, Inf, and malformed string
    df.loc[0, "Label"] = "Web Attack \x96 Brute Force"
    df.loc[1, "Label"] = "Web Attack – XSS"
    df.loc[2, "Flow Bytes/s"] = np.inf
    df.loc[3, "Flow Packets/s"] = -np.inf
    df["Flow Duration"] = df["Flow Duration"].astype(object)
    df.loc[4, "Flow Duration"] = "bad_string"
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)

    X_df, y_series, metadata, report = clean_cicids_dataframe(df, drop_ultra_rare_below=1)
    assert report.duplicates_removed >= 1
    assert report.inf_values_converted >= 2
    assert report.coerced_non_numeric_nans >= 1
    assert "Source IP" not in X_df.columns
    assert "Timestamp" not in X_df.columns
    assert "Destination Port" in X_df.columns
    assert "Web Attack" in set(y_series.tolist())
    assert "True_Label" in metadata.columns
    assert normalize_label_text("Web Attack \x96 Sql Injection") == "Web Attack - Sql Injection"


def test_schema_alignment_errors_and_coercion(small_synthetic_df: pd.DataFrame) -> None:
    with pytest.raises(SchemaValidationError, match="zero rows"):
        align_feature_schema(pd.DataFrame(columns=CICIDS2017_FEATURES), CICIDS2017_FEATURES)

    bad_df = small_synthetic_df.head(5).drop(columns=["Flow Duration", "SYN Flag Count"])
    with pytest.raises(SchemaValidationError, match="missing 2 required CICIDS2017"):
        align_feature_schema(bad_df, CICIDS2017_FEATURES)

    # Extra metadata + Inf + string coercion succeeds cleanly
    valid_df = small_synthetic_df.head(5).copy()
    valid_df.loc[0, "Flow Bytes/s"] = np.inf
    valid_df["Flow Duration"] = valid_df["Flow Duration"].astype(object)
    valid_df.loc[1, "Flow Duration"] = "not-a-number"
    aligned = align_feature_schema(valid_df, CICIDS2017_FEATURES)
    assert list(aligned.columns) == CICIDS2017_FEATURES
    assert np.isnan(aligned.loc[0, "Flow Bytes/s"])
    assert np.isnan(aligned.loc[1, "Flow Duration"])


def test_leakage_free_preprocessing_and_dynamic_smote_k(tmp_app_config: AppConfig) -> None:
    dataset = run_preprocessing_pipeline(tmp_app_config, force_synthetic=True, save_outputs=True)

    assert len(dataset.feature_names) == 77
    assert dataset.audit_report["leakage_checks"]["passed"] is True
    assert dataset.audit_report["leakage_checks"]["scaler_fit_strictly_on_train"] is True
    assert dataset.audit_report["leakage_checks"]["train_test_exact_overlap_sample"] == 0

    # Validation and Test must remain naturally imbalanced (BENIGN >> minority)
    val_counts = np.bincount(dataset.y_val)
    test_counts = np.bincount(dataset.y_test)
    assert val_counts.max() > val_counts.min() * 5
    assert test_counts.max() > test_counts.min() * 5

    # SMOTE with tiny 3-sample minority class dynamically bounds k_neighbors = 2
    X_tiny = np.random.default_rng(7).normal(size=(33, 6)).astype(np.float32)
    y_tiny = np.array([0] * 30 + [1] * 3, dtype=np.int64)
    X_res, y_res, info = apply_smote_safely(X_tiny, y_tiny, max_k_neighbors=5, seed=42)
    assert info["applied"] is True
    assert info["k_neighbors"] == 2
    assert len(X_res) == 60
    assert int((y_res == 1).sum()) == 30


def test_post_clip_near_duplicate_removal_and_strict_leakage_error(tmp_app_config: AppConfig) -> None:
    """Deliberate near-duplicates differing only pre-imputation/pre-clipping must yield 0 overlap."""
    assert tmp_app_config.data.strict_leakage_check is True

    base_df = generate_synthetic_cicids2017(
        n_samples=1200,
        seed=99,
        duplicate_fraction=0.0,
        inf_fraction=0.0,
        nan_fraction=0.0,
        save_path=None,
    )
    # Remove any accidental large outlier on Flow Bytes/s in baseline rows so 1e13 cleanly exceeds 5-sigma.
    base_df["Flow Bytes/s"] = np.clip(base_df["Flow Bytes/s"].to_numpy(dtype=np.float64), 0.0, 1e6)
    base_df["Flow Packets/s"] = np.clip(base_df["Flow Packets/s"].to_numpy(dtype=np.float64), 0.0, 1e5)

    # Inject raw-distinct near-duplicates that collapse onto identical vectors after 5-sigma clipping & imputation.
    template_clip = base_df.iloc[[0]].copy()
    clip_clones = []
    for k in range(10):
        row = template_clip.copy()
        row["Flow Bytes/s"] = 1e13 + k * 1e8
        clip_clones.append(row)

    template_inf = base_df.iloc[[1]].copy()
    inf_clones = []
    for k, val in enumerate([np.inf, -np.inf, np.nan, np.inf, -np.inf, np.nan]):
        row = template_inf.copy()
        row["Flow Packets/s"] = val
        row["Flow ID"] = f"inf-clone-{k}"
        inf_clones.append(row)

    combined = pd.concat([base_df, *clip_clones, *inf_clones], ignore_index=True)
    raw_csv = tmp_app_config.paths.resolve("raw_data_dir") / "near_dup_sample.csv"
    combined.to_csv(raw_csv, index=False)

    dataset = run_preprocessing_pipeline(tmp_app_config, force_synthetic=False, save_outputs=True)

    checks = dataset.audit_report["leakage_checks"]
    assert dataset.audit_report["source"]["source_type"] == "cicids2017_raw"
    assert dataset.audit_report["post_clip_dedup_removed"]["total_removed"] > 0
    assert checks["passed"] is True
    assert checks["checked_full_splits"] is True
    assert checks["train_test_exact_overlap"] == 0
    assert checks["train_val_exact_overlap"] == 0
    assert checks["val_test_exact_overlap"] == 0
    assert checks["train_test_exact_overlap_sample"] == 0
    assert checks["train_val_exact_overlap_sample"] == 0
    assert checks["val_test_exact_overlap_sample"] == 0

    # Inject cross-split overlap manually and confirm strict vs non-strict verification behavior.
    X_val_leaked = np.vstack([dataset.X_val_scaled, dataset.X_train_scaled[:2]])
    y_val_leaked = np.concatenate([dataset.y_val, dataset.y_train[:2]])
    X_test_leaked = np.vstack([dataset.X_test_scaled, dataset.X_train_scaled[2:4]])
    y_test_leaked = np.concatenate([dataset.y_test, dataset.y_train[2:4]])

    with pytest.raises(DataLeakageError, match="Zero-leakage verification failed"):
        verify_no_leakage(
            dataset.X_train_scaled,
            X_val_leaked,
            X_test_leaked,
            dataset.scaler,
            dataset.y_train,
            y_val_leaked,
            y_test_leaked,
            strict=True,
        )

    non_strict_checks = verify_no_leakage(
        dataset.X_train_scaled,
        X_val_leaked,
        X_test_leaked,
        dataset.scaler,
        dataset.y_train,
        y_val_leaked,
        y_test_leaked,
        strict=False,
    )
    assert non_strict_checks["passed"] is False
    assert non_strict_checks["train_val_exact_overlap"] >= 2
    assert non_strict_checks["train_test_exact_overlap"] >= 2


def test_by_day_split_and_drop_features(tmp_app_config: AppConfig) -> None:
    """by_day split separates Mon-Wed / part of Thu / Thu+Fri while honoring drop_features."""
    raw_dir = tmp_app_config.paths.resolve("raw_data_dir")
    generate_synthetic_cicids2017(
        n_samples=180, seed=11, class_weights={"BENIGN": 1.0}, save_path=raw_dir / "Monday-WorkingHours.csv"
    )
    generate_synthetic_cicids2017(
        n_samples=220,
        seed=12,
        class_weights={"BENIGN": 0.70, "DoS Hulk": 0.30},
        save_path=raw_dir / "Wednesday-workingHours.csv",
    )
    generate_synthetic_cicids2017(
        n_samples=160,
        seed=13,
        class_weights={"BENIGN": 0.70, "Web Attack": 0.30},
        save_path=raw_dir / "Thursday-WorkingHours.csv",
    )
    generate_synthetic_cicids2017(
        n_samples=200,
        seed=14,
        class_weights={"BENIGN": 0.60, "DDoS": 0.40},
        save_path=raw_dir / "Friday-WorkingHours.csv",
    )

    tmp_app_config.data.split_strategy = "by_day"
    tmp_app_config.data.drop_features = ["Destination Port"]

    dataset = run_preprocessing_pipeline(tmp_app_config, force_synthetic=False, save_outputs=False)
    audit = dataset.audit_report

    assert audit["split_strategy"] == "by_day"
    assert "Destination Port" in audit["dropped_features"]
    assert "Destination Port" not in dataset.feature_names
    assert len(dataset.feature_names) == 76
    # Destination Port stays available in metadata for threat mapping even when excluded from X
    assert "Destination Port" in dataset.test_metadata.columns

    assert audit["split_day_files"]["train_files"] == [
        "Monday-WorkingHours.csv",
        "Wednesday-workingHours.csv",
    ]
    assert "Thursday-WorkingHours.csv" in audit["split_day_files"]["val_files"]
    assert "Friday-WorkingHours.csv" in audit["split_day_files"]["test_files"]

    split_cls = audit["split_classes_present"]
    assert "DoS Hulk" in split_cls["train"] and "DoS Hulk" not in split_cls["test"]
    assert "DDoS" in split_cls["test"] and "DDoS" not in split_cls["train"]
    assert "Web Attack" in split_cls["validation"] and "Web Attack" in split_cls["test"]
    assert audit["leakage_checks"]["passed"] is True
    assert audit["leakage_checks"]["train_test_exact_overlap"] == 0
    assert audit["leakage_checks"]["train_val_exact_overlap"] == 0
    assert audit["leakage_checks"]["val_test_exact_overlap"] == 0


def test_smote_multiplier_cap_and_min_real_samples_threshold() -> None:
    """SMOTE skips ultra-rare classes below min_real_samples and enforces cap/multiplier bounds."""
    rng = np.random.default_rng(77)
    X_imb = np.vstack(
        [
            rng.normal(loc=0.0, scale=1.0, size=(500, 8)),
            rng.normal(loc=3.0, scale=1.0, size=(80, 8)),
            rng.normal(loc=-3.0, scale=1.0, size=(12, 8)),
        ]
    ).astype(np.float32)
    y_imb = np.array([0] * 500 + [1] * 80 + [2] * 12, dtype=np.int64)

    X_res, y_res, info = apply_smote_safely(
        X_imb,
        y_imb,
        max_k_neighbors=5,
        cap_per_class=5000,
        max_multiplier=5.0,
        min_real_samples=50,
        seed=42,
    )
    counts = np.bincount(y_res)
    assert info["applied"] is True
    assert counts[0] == 500
    assert counts[1] == 400  # 80 * 5.0 = 400 (< cap 5000)
    assert counts[2] == 12  # 12 < min_real_samples=50 -> skipped by SMOTE
    assert info["skipped_classes"] == {"2": 12}

    fallback_cw = {0: 0.4, 1: 2.5, 2: 16.0}
    hybrid_cw = compute_hybrid_smote_class_weights(info, fallback_cw)
    assert hybrid_cw is not None
    assert hybrid_cw[0] == 1.0
    assert hybrid_cw[1] == 1.0
    assert hybrid_cw[2] > 10.0

    # Absolute cap_per_class takes precedence when smaller than max_multiplier * real size
    _, y_capped, _ = apply_smote_safely(
        X_imb,
        y_imb,
        cap_per_class=200,
        max_multiplier=5.0,
        min_real_samples=50,
        seed=42,
    )
    assert np.bincount(y_capped)[1] == 200


def test_time_block_split_timestamp_ordering_no_leakage_and_fallbacks(tmp_app_config: AppConfig) -> None:
    """time_block sorts chronologically per class (~60/20/20), drops Timestamp, and handles tiny classes."""
    tmp_app_config.data.split_strategy = "time_block"
    tmp_app_config.data.synthetic_fallback.n_samples = 800
    dataset = run_preprocessing_pipeline(tmp_app_config, force_synthetic=True, save_outputs=False)
    audit = dataset.audit_report

    assert audit["split_strategy"] == "time_block"
    assert "Timestamp" not in dataset.feature_names
    assert "Capture_Order" not in dataset.feature_names
    assert "Timestamp" in audit["id_columns_removed"]
    assert audit["time_block_details"]["ordering_source"] == "Timestamp"
    assert audit["leakage_checks"]["passed"] is True
    assert audit["leakage_checks"]["train_test_exact_overlap"] == 0
    assert audit["leakage_checks"]["train_val_exact_overlap"] == 0
    assert audit["leakage_checks"]["val_test_exact_overlap"] == 0

    # Verify validation timestamps strictly precede test timestamps within every class
    val_ts = pd.to_datetime(dataset.val_metadata["Timestamp"])
    test_ts = pd.to_datetime(dataset.test_metadata["Timestamp"])
    for cls_name in dataset.class_names:
        v_mask = dataset.val_metadata["True_Label"].to_numpy() == cls_name
        t_mask = dataset.test_metadata["True_Label"].to_numpy() == cls_name
        if np.any(v_mask) and np.any(t_mask):
            assert val_ts[v_mask].max() <= test_ts[t_mask].min()

    # Verify DD/MM/YYYY TrafficLabelling timestamps + tiny-class fallbacks (n=1, 2, 3, 4, 10)
    cls_names = ["BENIGN", "TinyOne", "TinyTwo", "TinyThree", "TinyFour"]
    y_tiny = np.array([0] * 10 + [1] * 1 + [2] * 2 + [3] * 3 + [4] * 4, dtype=np.int64)
    # Deliberately reverse timestamp strings vs array index to verify genuine Timestamp sorting
    meta_ts = pd.DataFrame(
        {"Timestamp": [f"{20 - idx:02d}/07/2017 09:00:{idx:02d}" for idx in range(len(y_tiny))]}
    )
    i_tr, i_va, i_te, details = _split_indices_time_block(meta_ts, y_tiny, cls_names)
    assert details["ordering_source"] == "Timestamp"
    assert len(set(i_tr) & set(i_va)) == 0
    assert len(set(i_tr) & set(i_te)) == 0
    assert len(set(i_va) & set(i_te)) == 0
    assert details["per_class_allocations"]["BENIGN"] == {"total": 10, "train": 6, "validation": 2, "test": 2}
    assert details["per_class_allocations"]["TinyFour"] == {
        "total": 4,
        "train": 2,
        "validation": 1,
        "test": 1,
    }
    assert details["per_class_allocations"]["TinyThree"] == {
        "total": 3,
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert details["per_class_allocations"]["TinyTwo"] == {"total": 2, "train": 1, "validation": 0, "test": 1}
    assert details["per_class_allocations"]["TinyOne"] == {"total": 1, "train": 1, "validation": 0, "test": 0}
    assert "TinyOne" in details["tiny_class_fallbacks"]
    # Verify earliest timestamps went to train and latest to test for BENIGN (idx 9 = oldest 11/07 -> idx 0 = newest 20/07)
    benign_tr = [i for i in i_tr if y_tiny[i] == 0]
    benign_te = [i for i in i_te if y_tiny[i] == 0]
    parsed = pd.to_datetime(meta_ts["Timestamp"], dayfirst=True)
    assert parsed.iloc[benign_tr].max() <= parsed.iloc[benign_te].min()

    # Verify transparent fallback to Capture_Order when Timestamp column is absent (MachineLearningCVE style)
    meta_cap = pd.DataFrame({"Capture_Order": np.arange(len(y_tiny))[::-1]})
    _, _, _, cap_details = _split_indices_time_block(meta_cap, y_tiny, cls_names)
    assert cap_details["ordering_source"].startswith("Capture_Order")
