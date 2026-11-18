"""Stateless data cleaning, structural identifier exclusion, and label normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("smart_ids.data.cleaner")

SOURCE_FILE_COLUMN = "__source_file__"
CHRON_ORDER_COLUMN = "__chron_order__"
INTERNAL_PROVENANCE_COLUMNS = frozenset({SOURCE_FILE_COLUMN, CHRON_ORDER_COLUMN})


@dataclass
class CleaningReport:
    """Audit metrics emitted by the deterministic cleaning stage."""

    initial_rows: int = 0
    initial_cols: int = 0
    rows_after_dedup: int = 0
    duplicates_removed: int = 0
    inf_values_converted: int = 0
    coerced_non_numeric_nans: int = 0
    nans_before_split: int = 0
    id_columns_removed: list[str] = field(default_factory=list)
    duplicate_columns_removed: list[str] = field(default_factory=list)
    dropped_features_removed: list[str] = field(default_factory=list)
    rare_classes_dropped: dict[str, int] = field(default_factory=dict)
    label_mapping_applied: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all column names."""
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()
    return out


def extract_flow_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extract non-feature metadata columns for SOC Threat Map and audit reporting."""
    meta_cols = [
        c for c in ("Source IP", "Destination IP", "Destination Port", "Timestamp") if c in df.columns
    ]
    if not meta_cols:
        return pd.DataFrame(index=df.index)
    return df[meta_cols].copy()


def identify_structural_columns(
    columns: list[str] | pd.Index,
    identifier_keywords: list[str],
    preserve_features: list[str],
    label_col: str = "Label",
) -> list[str]:
    """Identify non-generalizable structural identifiers (IPs, timestamps, flow IDs, source port)."""
    preserve_lower = {p.lower() for p in preserve_features}
    id_cols: list[str] = []
    for col in columns:
        col_str = str(col)
        col_lower = col_str.lower()
        if col_str == label_col or col_lower in preserve_lower:
            continue
        if any(kw in col_lower for kw in identifier_keywords):
            id_cols.append(col_str)
    # Always exclude timestamp columns even if identifier_keywords was customized, so timestamps
    # remain non-feature ordering metadata and never leak into model features.
    for col in columns:
        col_str = str(col)
        if col_str != label_col and col_str.strip().lower() == "timestamp" and col_str not in id_cols:
            id_cols.append(col_str)
    return id_cols


def normalize_label_text(value: Any) -> str:
    """Normalize CICIDS2017 label strings, fixing encoding mojibake and extra spaces."""
    text = str(value).strip()
    # Replace mojibake replacement characters from Latin-1/UTF-8 dash mismatches
    text = re.sub(r"[\x80-\x9f\ufffd–—]+", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_cicids_dataframe(
    df: pd.DataFrame,
    label_col: str = "Label",
    identifier_keywords: list[str] | None = None,
    preserve_features: list[str] | None = None,
    drop_features: list[str] | None = None,
    merge_web_attacks: bool = True,
    drop_ultra_rare_below: int = 10,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, CleaningReport]:
    """Run deterministic, zero-leakage cleaning prior to train/val/test splitting.

    Note: Imputation (median) and outlier clipping (5-sigma) are deliberately NOT
    executed here; they are fit exclusively on the training partition after splitting.
    """
    raw_col_count = df.shape[1] - sum(1 for col in INTERNAL_PROVENANCE_COLUMNS if col in df.columns)
    report = CleaningReport(initial_rows=len(df), initial_cols=raw_col_count)
    work = normalize_column_names(df)

    if label_col not in work.columns:
        raise KeyError(f"Expected label column '{label_col}' not found in dataset columns.")

    # Normalize labels & merge Web Attack sub-labels if requested
    raw_labels = work[label_col].map(normalize_label_text)
    mapping_log: dict[str, str] = {}
    if merge_web_attacks:
        normalized_labels = []
        for lbl in raw_labels:
            if lbl.lower().startswith("web attack"):
                if lbl != "Web Attack":
                    mapping_log[lbl] = "Web Attack"
                normalized_labels.append("Web Attack")
            else:
                normalized_labels.append(lbl)
        work[label_col] = normalized_labels
    else:
        work[label_col] = raw_labels
    report.label_mapping_applied = mapping_log

    # Drop rows with missing/empty labels
    work = work[work[label_col].notna() & (work[label_col] != "") & (work[label_col].str.lower() != "nan")]

    # Drop ultra-rare classes that cannot survive stratified splitting or SMOTE
    if drop_ultra_rare_below > 1:
        counts = work[label_col].value_counts()
        rare = counts[counts < drop_ultra_rare_below]
        if not rare.empty:
            report.rare_classes_dropped = {str(k): int(v) for k, v in rare.items()}
            work = work[~work[label_col].isin(rare.index)]
            logger.info(
                "Dropped ultra-rare classes (< %d samples): %s",
                drop_ultra_rare_below,
                report.rare_classes_dropped,
            )

    # Remove exact duplicate rows (excluding internal provenance columns) before separating metadata
    before_dedup = len(work)
    dedup_cols = [c for c in work.columns if c not in INTERNAL_PROVENANCE_COLUMNS]
    work = work.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    report.duplicates_removed = before_dedup - len(work)
    report.rows_after_dedup = len(work)

    # Extract SOC metadata prior to dropping structural identifiers or ablated features
    metadata = extract_flow_metadata(work)
    metadata["True_Label"] = work[label_col].values
    if SOURCE_FILE_COLUMN in work.columns:
        metadata["Source_File"] = work[SOURCE_FILE_COLUMN].astype(str).values
    if CHRON_ORDER_COLUMN in work.columns:
        metadata["Capture_Order"] = work[CHRON_ORDER_COLUMN].to_numpy(dtype=np.int64)

    # Remove structural identifiers (never include internal provenance tags in id_columns_removed)
    kw_list = identifier_keywords or ["ip", "timestamp", "flow id", "src", "dst", "source port"]
    preserve = preserve_features or ["Destination Port"]
    feature_candidates = [c for c in work.columns if c not in INTERNAL_PROVENANCE_COLUMNS]
    id_cols = identify_structural_columns(feature_candidates, kw_list, preserve, label_col=label_col)
    report.id_columns_removed = id_cols

    # Drop duplicate CICFlowMeter header column 'Fwd Header Length.1' if present
    dup_cols = [c for c in ("Fwd Header Length.1",) if c in work.columns]
    report.duplicate_columns_removed = dup_cols

    # Explicit user/experiment feature ablation list (overrides preserve_features)
    drop_feat_lower = {str(c).strip().lower() for c in (drop_features or []) if str(c).strip()}
    explicit_drops = [
        c
        for c in feature_candidates
        if c != label_col
        and c not in id_cols
        and c not in dup_cols
        and str(c).strip().lower() in drop_feat_lower
    ]
    report.dropped_features_removed = explicit_drops

    y_series = work[label_col].copy()
    drop_set = set(id_cols) | set(dup_cols) | set(explicit_drops) | {label_col} | INTERNAL_PROVENANCE_COLUMNS
    X_df = work.drop(columns=[c for c in work.columns if c in drop_set]).copy()

    # Coerce all remaining features to numeric, tracking coerced NaNs and Inf replacements
    inf_count = 0
    coerced_nans = 0
    for col in X_df.columns:
        series = X_df[col]
        if pd.api.types.is_numeric_dtype(series):
            arr = series.to_numpy(dtype=np.float64, copy=False)
            col_inf = int(np.isinf(arr).sum())
            inf_count += col_inf
            if col_inf > 0:
                series = series.replace([np.inf, -np.inf], np.nan)
            X_df[col] = series.astype(np.float64)
        else:
            was_na = series.isna()
            coerced = pd.to_numeric(series, errors="coerce")
            coerced_nans += int((coerced.isna() & ~was_na).sum())
            arr = coerced.to_numpy(dtype=np.float64, copy=False)
            col_inf = int(np.isinf(arr).sum())
            inf_count += col_inf
            if col_inf > 0:
                coerced = coerced.replace([np.inf, -np.inf], np.nan)
            X_df[col] = coerced.astype(np.float64)

    report.inf_values_converted = inf_count
    report.coerced_non_numeric_nans = coerced_nans
    report.nans_before_split = int(X_df.isna().sum().sum())

    logger.info(
        "Cleaned dataset: %d -> %d rows | %d ID cols removed | %d inf -> NaN | %d coerced NaNs | %d total NaNs remaining for train-only imputation",
        report.initial_rows,
        report.rows_after_dedup,
        len(report.id_columns_removed),
        report.inf_values_converted,
        report.coerced_non_numeric_nans,
        report.nans_before_split,
    )
    return X_df, y_series, metadata, report
