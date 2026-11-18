"""Tests for CSV discovery, cleaning, schema alignment, and zero-leakage preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import AppConfig
from src.data.cleaner import clean_cicids_dataframe, normalize_label_text
from src.data.feature_engineering import SchemaValidationError, align_feature_schema
from src.data.loader import CICIDS2017_FEATURES, discover_csv_files
from src.data.preprocessing import apply_smote_safely, run_preprocessing_pipeline


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
