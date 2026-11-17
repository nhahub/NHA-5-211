"""Data engineering, cleaning, feature selection, scaling, and SMOTE rebalancing.

Implements the 8-step data pipeline specified in the Smart IDS v3 technical report:
1. Replace +/- inf with NaN (zero-division remediation).
2. Drop columns > 50% NaN, impute rest with column median, remove duplicates.
3. Drop non-causal identifier columns (IPs, Ports, Timestamps, Flow IDs) to prevent leakage.
4. Statistical outlier clipping at mean +/- 5 std.
5. Stratified train/val/test split (70% train / 15% val / 15% test).
6. StandardScaler fit exclusively on train partition.
7. SMOTE oversampling applied exclusively to train partition with dynamic k-neighbor guard.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from smart_ids.config import (
    CLASS_NAMES,
    EXPLICIT_EXCLUDE_COLS,
    IDENTIFIER_KEYWORDS,
    NAN_DROP_THRESHOLD,
    OUTLIER_SIGMA_MULTIPLE,
    SEED,
    TEST_SIZE,
    VAL_SIZE_OF_REMAINDER,
)

logger = logging.getLogger("smart_ids.preprocessing")


def clean_cicids_dataframe(
    df: pd.DataFrame,
    nan_threshold: float = NAN_DROP_THRESHOLD,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean raw CICIDS2017 dataframe with full transparency and auditing.

    Replaces +/- inf with NaN, drops heavily corrupted columns, imputes
    remaining NaNs with median, and drops exact duplicates.

    Args:
        df: Input dataframe.
        nan_threshold: Max fraction of missing values allowed per column (default 0.50).

    Returns:
        Tuple of (cleaned_df, audit_dict).
    """
    df_clean = df.copy()

    # 1. Count and replace infinities
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    inf_count = 0
    if len(numeric_cols) > 0:
        inf_mask = np.isinf(df_clean[numeric_cols].to_numpy())
        inf_count = int(np.sum(inf_mask))
        df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Also check object columns that may contain string 'inf' or 'Infinity'
    str_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in str_cols:
        if col.lower() != "label":
            df_clean[col] = df_clean[col].replace(["inf", "-inf", "Infinity", "-Infinity"], np.nan)

    logger.info("Audit: Converted %d infinite values to NaN.", inf_count)

    # 2. Drop columns with > nan_threshold missing values
    nan_ratio = df_clean.isna().mean()
    cols_to_drop = nan_ratio[nan_ratio > nan_threshold].index.tolist()
    # Never drop Label column
    cols_to_drop = [c for c in cols_to_drop if c.lower() != "label"]
    if cols_to_drop:
        logger.warning("Dropping columns exceeding %.0f%% NaN: %s", nan_threshold * 100, cols_to_drop)
        df_clean = df_clean.drop(columns=cols_to_drop)

    # 3. Impute remaining missing values with column median
    num_cols_remaining = df_clean.select_dtypes(include=[np.number]).columns
    nan_imputed_count = int(df_clean[num_cols_remaining].isna().sum().sum())
    if nan_imputed_count > 0:
        medians = df_clean[num_cols_remaining].median()
        df_clean[num_cols_remaining] = df_clean[num_cols_remaining].fillna(medians)
        logger.info("Audit: Imputed %d NaN values using feature medians.", nan_imputed_count)

    # 4. Deduplication
    pre_dedupe_count = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    duplicates_removed = pre_dedupe_count - len(df_clean)
    logger.info("Audit: Removed %d exact duplicate rows.", duplicates_removed)

    audit = {
        "inf_converted": inf_count,
        "cols_dropped": cols_to_drop,
        "nan_imputed": nan_imputed_count,
        "duplicates_removed": duplicates_removed,
        "final_rows": len(df_clean),
        "final_cols": df_clean.shape[1],
    }

    return df_clean, audit


def remove_identifier_columns(
    df: pd.DataFrame,
    explicit_cols: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove non-generalizable network identifier columns to prevent target leakage.

    Drops IP addresses, timestamps, ports, and flow IDs. Coerces remaining
    features to numeric types.

    Args:
        df: Input dataframe.
        explicit_cols: Specific column names to remove if found.
        keywords: Substring keywords (lowercased) triggering column removal.

    Returns:
        Tuple of (filtered_df, list_of_dropped_cols).
    """
    if explicit_cols is None:
        explicit_cols = EXPLICIT_EXCLUDE_COLS
    if keywords is None:
        keywords = IDENTIFIER_KEYWORDS

    df_filtered = df.copy()
    dropped_cols: List[str] = []

    for col in df_filtered.columns:
        if col.lower() == "label":
            continue

        c_lower = col.lower()
        is_explicit = col in explicit_cols
        has_keyword = any(kw in c_lower for kw in keywords)

        if is_explicit or has_keyword:
            dropped_cols.append(col)

    if dropped_cols:
        logger.info("Dropping %d identifier/leakage columns: %s", len(dropped_cols), dropped_cols)
        df_filtered = df_filtered.drop(columns=dropped_cols, errors="ignore")

    # Defensive numeric coercion for all feature columns
    for col in df_filtered.columns:
        if col.lower() != "label":
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")
            # If any newly produced NaNs from coercion, fill with median or 0
            if df_filtered[col].isna().any():
                col_median = df_filtered[col].median()
                df_filtered[col] = df_filtered[col].fillna(0.0 if np.isnan(col_median) else col_median)

    return df_filtered, dropped_cols


def clip_outliers(
    df: pd.DataFrame,
    sigma_multiple: float = OUTLIER_SIGMA_MULTIPLE,
) -> pd.DataFrame:
    """Clip numerical features at empirical mean +/- sigma_multiple * std.

    Prevents pathological rate spikes from destabilizing neural network
    backpropagation.

    Args:
        df: Input dataframe.
        sigma_multiple: Number of standard deviations for clipping boundary (default 5.0).

    Returns:
        pd.DataFrame: Dataframe with clipped numerical features.
    """
    df_clipped = df.copy()
    num_cols = df_clipped.select_dtypes(include=[np.number]).columns

    clipped_count = 0
    for col in num_cols:
        if col.lower() == "label":
            continue
        mu = float(df_clipped[col].mean())
        sigma = float(df_clipped[col].std())
        if sigma > 1e-8:
            lower = mu - sigma_multiple * sigma
            upper = mu + sigma_multiple * sigma
            col_vals = df_clipped[col].to_numpy()
            outliers = (col_vals < lower) | (col_vals > upper)
            if np.any(outliers):
                clipped_count += int(np.sum(outliers))
                df_clipped[col] = np.clip(col_vals, lower, upper)

    logger.info("Audit: Clipped %d outlier values across features at %.1f sigma.", clipped_count, sigma_multiple)
    return df_clipped


def split_data(
    df: pd.DataFrame,
    target_col: str = "Label",
    test_size: float = TEST_SIZE,
    val_size_of_remainder: float = VAL_SIZE_OF_REMAINDER,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Execute stratified two-stage train/val/test split.

    Stage 1: Split total dataset into (train_val, test) using test_size.
    Stage 2: Split train_val into (train, val) using val_size_of_remainder.

    Args:
        df: Cleaned and clipped dataset.
        target_col: Column name containing target labels.
        test_size: Fraction of total data for test set (default 0.15).
        val_size_of_remainder: Fraction of train_val for validation set (default 0.15).
        random_state: PRNG seed for stratified splitting.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Stage 1: Hold out test partition
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Stage 2: Hold out validation partition from remainder
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size_of_remainder,
        stratify=y_train_val,
        random_state=random_state,
    )

    logger.info(
        "Stratified split complete: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        len(X_train),
        len(X_train) / len(df) * 100,
        len(X_val),
        len(X_val) / len(df) * 100,
        len(X_test),
        len(X_test) / len(df) * 100,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler strictly on training set and transform val and test.

    Enforces zero data leakage from evaluation partitions.

    Args:
        X_train: Training features dataframe.
        X_val: Validation features dataframe.
        X_test: Test features dataframe.

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info("StandardScaler fit on train only (%d samples) and applied to val/test.", len(X_train))
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def apply_smote_rebalancing(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    random_state: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE rebalancing strictly to the training partition.

    Dynamically bounds k_neighbors based on the cardinality of the rarest
    minority class to prevent nearest-neighbor runtime errors.
    Validation and test sets remain naturally imbalanced.

    Args:
        X_train_scaled: Scaled training features.
        y_train: Training integer or string labels.
        random_state: PRNG seed for reproducibility.

    Returns:
        Tuple of (X_train_sm, y_train_sm).
    """
    unique_classes, counts = np.unique(y_train, return_counts=True)
    min_count = int(np.min(counts))

    # Guard: k_neighbors must be < min_count and >= 1
    k_neighbors = min(5, max(1, min_count - 1))
    logger.info(
        "Applying SMOTE on train only. Smallest class size: %d, dynamic k_neighbors: %d",
        min_count,
        k_neighbors,
    )

    smote = SMOTE(
        sampling_strategy="auto",
        k_neighbors=k_neighbors,
        random_state=random_state,
    )

    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
    logger.info("SMOTE complete. Resampled train size: %d samples.", len(X_train_sm))

    return X_train_sm, y_train_sm
