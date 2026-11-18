"""Leakage-free feature transformers and inference schema validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SchemaValidationError(ValueError):
    """Raised when an input DataFrame violates the expected feature schema."""


class HighMissingColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns exceeding nan_threshold based strictly on training missing ratios."""

    def __init__(self, nan_threshold: float = 0.50) -> None:
        self.nan_threshold = nan_threshold
        self.cols_to_drop_: list[str] = []
        self.kept_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> HighMissingColumnDropper:
        nan_ratio = X.isna().mean()
        self.cols_to_drop_ = nan_ratio[nan_ratio > self.nan_threshold].index.tolist()
        self.kept_columns_ = [c for c in X.columns if c not in set(self.cols_to_drop_)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.kept_columns_].copy()


class MedianImputerTransformer(BaseEstimator, TransformerMixin):
    """Column-wise median imputer fit exclusively on training data."""

    def __init__(self) -> None:
        self.medians_: dict[str, float] = {}
        self.last_imputed_count_: int = 0

    def fit(self, X: pd.DataFrame, y: Any = None) -> MedianImputerTransformer:
        medians = X.median(numeric_only=True)
        self.medians_ = {col: float(0.0 if pd.isna(val) else val) for col, val in medians.items()}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = X.to_numpy(dtype=np.float64, copy=True)
        nan_mask = np.isnan(arr)
        self.last_imputed_count_ = int(nan_mask.sum())
        if self.last_imputed_count_ > 0:
            fill_vec = np.array([self.medians_.get(str(c), 0.0) for c in X.columns], dtype=np.float64)
            rows, cols = np.where(nan_mask)
            arr[rows, cols] = fill_vec[cols]
        return pd.DataFrame(arr, columns=X.columns, index=X.index)


class SigmaOutlierClipper(BaseEstimator, TransformerMixin):
    """Clip numerical features to [mu - n_sigma*sigma, mu + n_sigma*sigma] fit on training data."""

    def __init__(self, n_sigma: float = 5.0) -> None:
        self.n_sigma = n_sigma
        self.bounds_: dict[str, tuple[float, float]] = {}
        self.last_clipped_count_: int = 0

    def fit(self, X: pd.DataFrame, y: Any = None) -> SigmaOutlierClipper:
        self.bounds_ = {}
        for col in X.columns:
            mu = float(X[col].mean())
            sigma = float(X[col].std())
            if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0.0:
                self.bounds_[col] = (mu - self.n_sigma * sigma, mu + self.n_sigma * sigma)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = X.to_numpy(dtype=np.float64, copy=True)
        lo_vec = np.array(
            [self.bounds_[str(c)][0] if str(c) in self.bounds_ else -np.inf for c in X.columns],
            dtype=np.float64,
        )
        hi_vec = np.array(
            [self.bounds_[str(c)][1] if str(c) in self.bounds_ else np.inf for c in X.columns],
            dtype=np.float64,
        )
        clipped_mask = (arr < lo_vec) | (arr > hi_vec)
        self.last_clipped_count_ = int(clipped_mask.sum())
        if self.last_clipped_count_ > 0:
            arr = np.clip(arr, lo_vec, hi_vec)
        return pd.DataFrame(arr, columns=X.columns, index=X.index)


def align_feature_schema(
    df: pd.DataFrame,
    expected_features: list[str],
    fill_inf_with_nan: bool = True,
) -> pd.DataFrame:
    """Validate and align an inference DataFrame to the exact training feature schema.

    Strips header whitespace, removes metadata/label columns if present, enforces numeric
    types, converts Inf to NaN so the fitted imputer handles them cleanly, and raises a
    descriptive SchemaValidationError if any required feature is missing.
    """
    if df.empty:
        raise SchemaValidationError("Uploaded dataset contains zero rows.")

    work = df.copy()
    work.columns = work.columns.astype(str).str.strip()

    missing = [col for col in expected_features if col not in work.columns]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise SchemaValidationError(
            f"Input data is missing {len(missing)} required CICIDS2017 feature column(s): {preview}{suffix}"
        )

    aligned = work[expected_features].copy()
    for col in expected_features:
        if not pd.api.types.is_numeric_dtype(aligned[col]):
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
        if fill_inf_with_nan:
            aligned[col] = aligned[col].replace([np.inf, -np.inf], np.nan)
        aligned[col] = aligned[col].astype(np.float64)

    return aligned
