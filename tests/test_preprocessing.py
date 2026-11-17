"""Unit tests validating zero data leakage during scaling and SMOTE rebalancing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from smart_ids.config import CLASS_NAMES, SEED
from smart_ids.data_loader import generate_synthetic_cicids
from smart_ids.preprocessing import (
    apply_smote_rebalancing,
    clean_cicids_dataframe,
    clip_outliers,
    remove_identifier_columns,
    scale_features,
    split_data,
)


@pytest.fixture
def sample_raw_data():
    """Generate a small reproducible synthetic batch."""
    return generate_synthetic_cicids(n_samples=1500, random_state=SEED)


class TestDataCleaningPipeline:
    """Verify data cleaning, identifier exclusion, and outlier clipping."""

    def test_infinite_and_duplicate_remediation(self, sample_raw_data):
        df_clean, audit = clean_cicids_dataframe(sample_raw_data)
        # Check no infinities exist
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        assert not np.isinf(df_clean[num_cols].to_numpy()).any()
        # Check no duplicate rows exist
        assert not df_clean.duplicated().any()
        assert audit["duplicates_removed"] >= 0

    def test_identifier_exclusion_prevents_leakage(self, sample_raw_data):
        df_clean, _ = clean_cicids_dataframe(sample_raw_data)
        df_filtered, dropped = remove_identifier_columns(df_clean)

        for col in df_filtered.columns:
            if col.lower() != "label":
                # Ensure no IP, Port, or Timestamp columns leaked into feature set
                c_low = col.lower()
                assert "timestamp" not in c_low
                assert "flow id" not in c_low
                assert not any(kw in c_low for kw in ["src ip", "dst ip", "source ip", "destination ip"])

    def test_outlier_clipping_bounds(self, sample_raw_data):
        df_clean, _ = clean_cicids_dataframe(sample_raw_data)
        df_filt, _ = remove_identifier_columns(df_clean)
        df_clipped = clip_outliers(df_filt, sigma_multiple=5.0)

        num_cols = [c for c in df_clipped.columns if c.lower() != "label"]
        for col in num_cols[:10]:
            mu = df_filt[col].mean()
            sigma = df_filt[col].std()
            if sigma > 1e-6:
                assert df_clipped[col].max() <= (mu + 5.0 * sigma + 1e-4)
                assert df_clipped[col].min() >= (mu - 5.0 * sigma - 1e-4)


class TestNoDataLeakageEnforcement:
    """Strictly verify that StandardScaler and SMOTE are fit on train only."""

    def test_scaler_fit_on_train_only(self, sample_raw_data):
        df_clean, _ = clean_cicids_dataframe(sample_raw_data)
        df_filt, _ = remove_identifier_columns(df_clean)
        df_clipped = clip_outliers(df_filt)

        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(df_clipped, random_state=SEED)
        X_tr_sc, X_val_sc, X_te_sc, scaler = scale_features(X_tr, X_val, X_te)

        # 1. Scaler means must match X_tr column means exactly
        np.testing.assert_allclose(scaler.mean_, X_tr.mean(axis=0).to_numpy(), rtol=1e-5)

        # 2. Scaler means must NOT match X_te column means (different distribution)
        assert not np.allclose(scaler.mean_, X_te.mean(axis=0).to_numpy(), rtol=1e-3)

        # 3. Scaled train features must have mean ~0 and std ~1
        np.testing.assert_allclose(X_tr_sc.mean(axis=0), 0.0, atol=1e-3)
        np.testing.assert_allclose(X_tr_sc.std(axis=0), 1.0, atol=1e-3)

    def test_smote_applied_to_train_only(self, sample_raw_data):
        df_clean, _ = clean_cicids_dataframe(sample_raw_data)
        df_filt, _ = remove_identifier_columns(df_clean)
        df_clipped = clip_outliers(df_filt)

        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(df_clipped, random_state=SEED)
        X_tr_sc, X_val_sc, X_te_sc, scaler = scale_features(X_tr, X_val, X_te)

        # Pre-SMOTE distribution is imbalanced
        pre_counts = pd.Series(y_tr).value_counts()
        assert pre_counts.max() > pre_counts.min() * 5

        # Apply SMOTE to train only
        X_tr_sm, y_tr_sm = apply_smote_rebalancing(X_tr_sc, y_tr.to_numpy(), random_state=SEED)

        # Post-SMOTE train distribution must be balanced
        post_counts = pd.Series(y_tr_sm).value_counts()
        assert post_counts.nunique() == 1  # All classes equal size in train

        # Validation and Test sets MUST remain naturally imbalanced!
        val_counts = pd.Series(y_val).value_counts()
        test_counts = pd.Series(y_te).value_counts()
        assert val_counts.nunique() > 1
        assert test_counts.nunique() > 1
        assert val_counts.max() > val_counts.min() * 5
        assert test_counts.max() > test_counts.min() * 5
