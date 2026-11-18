"""Leakage-free data splitting, scaling, SMOTE rebalancing, and audit reporting."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.config import AppConfig
from src.data.cleaner import clean_cicids_dataframe
from src.data.feature_engineering import (
    HighMissingColumnDropper,
    MedianImputerTransformer,
    SigmaOutlierClipper,
)
from src.data.loader import DatasetSourceInfo, load_cicids2017_corpus
from src.utils.logging import get_logger
from src.utils.model_registry import ModelRegistry

logger = get_logger("smart_ids.data.preprocessing")


@dataclass
class PreparedDataset:
    """All split feature matrices, targets, metadata, transformers, and audit statistics."""

    X_train_scaled: np.ndarray
    y_train: np.ndarray
    X_train_sm: np.ndarray
    y_train_sm: np.ndarray
    X_val_scaled: np.ndarray
    y_val: np.ndarray
    X_test_scaled: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    class_weights: dict[int, float]
    scaler: StandardScaler
    imputer: MedianImputerTransformer
    outlier_clipper: SigmaOutlierClipper
    label_encoder: LabelEncoder
    test_metadata: pd.DataFrame
    val_metadata: pd.DataFrame
    source_info: DatasetSourceInfo
    audit_report: dict[str, Any]

    def summary_dict(self) -> dict[str, Any]:
        return self.audit_report


class DataLeakageError(RuntimeError):
    """Raised when strict split-isolation or zero-leakage checks fail."""


def _row_hash_list(arr: np.ndarray, decimals: int | None = None) -> list[bytes]:
    """Compute canonical per-row byte signatures for exact or rounded duplicate checks."""
    if len(arr) == 0:
        return []
    sub = np.asarray(arr)
    if decimals is not None:
        sub = np.round(sub, decimals)
    # Normalize IEEE -0.0 to +0.0 so byte signatures match mathematically equal floats.
    sub = np.ascontiguousarray(sub + sub.dtype.type(0.0))
    return [row.tobytes() for row in sub]


def _deduplicate_post_clip_splits(
    X_train_clip: pd.DataFrame,
    y_train: np.ndarray,
    X_val_clip: pd.DataFrame,
    y_val: np.ndarray,
    val_metadata: pd.DataFrame,
    X_test_clip: pd.DataFrame,
    y_test: np.ndarray,
    test_metadata: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    dict[str, int],
]:
    """Deduplicate clipped splits and remove val/test feature vectors that exist in train."""
    train_hashes = _row_hash_list(X_train_clip.to_numpy(dtype=np.float64, copy=False))
    seen_train_pairs: set[tuple[bytes, int]] = set()
    train_feature_set: set[bytes] = set()
    train_keep = np.zeros(len(X_train_clip), dtype=bool)
    for i, (h, label) in enumerate(zip(train_hashes, y_train, strict=True)):
        pair = (h, int(label))
        if pair not in seen_train_pairs:
            seen_train_pairs.add(pair)
            train_feature_set.add(h)
            train_keep[i] = True

    val_hashes = _row_hash_list(X_val_clip.to_numpy(dtype=np.float64, copy=False))
    seen_val_pairs: set[tuple[bytes, int]] = set()
    val_feature_set: set[bytes] = set()
    val_keep = np.zeros(len(X_val_clip), dtype=bool)
    for i, (h, label) in enumerate(zip(val_hashes, y_val, strict=True)):
        pair = (h, int(label))
        if h not in train_feature_set and pair not in seen_val_pairs:
            seen_val_pairs.add(pair)
            val_feature_set.add(h)
            val_keep[i] = True

    test_hashes = _row_hash_list(X_test_clip.to_numpy(dtype=np.float64, copy=False))
    seen_test_pairs: set[tuple[bytes, int]] = set()
    test_keep = np.zeros(len(X_test_clip), dtype=bool)
    for i, (h, label) in enumerate(zip(test_hashes, y_test, strict=True)):
        pair = (h, int(label))
        if h not in train_feature_set and h not in val_feature_set and pair not in seen_test_pairs:
            seen_test_pairs.add(pair)
            test_keep[i] = True

    stats = {
        "train_within_split_removed": int((~train_keep).sum()),
        "val_cross_or_dup_removed": int((~val_keep).sum()),
        "test_cross_or_dup_removed": int((~test_keep).sum()),
    }

    return (
        X_train_clip.loc[train_keep].reset_index(drop=True),
        y_train[train_keep],
        X_val_clip.loc[val_keep].reset_index(drop=True),
        y_val[val_keep],
        val_metadata.loc[val_keep].reset_index(drop=True),
        X_test_clip.loc[test_keep].reset_index(drop=True),
        y_test[test_keep],
        test_metadata.loc[test_keep].reset_index(drop=True),
        stats,
    )


def _drop_scaled_cross_split_collisions(
    X_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    val_metadata: pd.DataFrame,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    test_metadata: pd.DataFrame,
    decimals: int = 6,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    dict[str, int],
]:
    """Remove any residual float32/rounding collisions in val/test against earlier splits."""
    train_set = set(_row_hash_list(X_train_scaled, decimals=decimals))

    val_hashes = _row_hash_list(X_val_scaled, decimals=decimals)
    val_keep = np.array([h not in train_set for h in val_hashes], dtype=bool)
    val_set = {h for h, keep in zip(val_hashes, val_keep, strict=True) if keep}

    test_hashes = _row_hash_list(X_test_scaled, decimals=decimals)
    test_keep = np.array([h not in train_set and h not in val_set for h in test_hashes], dtype=bool)

    stats = {
        "val_scaled_collision_removed": int((~val_keep).sum()),
        "test_scaled_collision_removed": int((~test_keep).sum()),
    }
    return (
        X_val_scaled[val_keep],
        y_val[val_keep],
        val_metadata.loc[val_keep].reset_index(drop=True),
        X_test_scaled[test_keep],
        y_test[test_keep],
        test_metadata.loc[test_keep].reset_index(drop=True),
        stats,
    )


def verify_no_leakage(
    X_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    scaler: StandardScaler,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    strict: bool = True,
) -> dict[str, Any]:
    """Run automated verification assertions confirming zero train/eval leakage."""
    train_means = np.abs(X_train_scaled.mean(axis=0))
    max_train_mean = float(np.max(train_means))
    val_mean_shift = float(np.max(np.abs(X_val_scaled.mean(axis=0)))) if len(X_val_scaled) else 0.0

    # Hash every row across all splits (no truncation limit) to detect any cross-split feature overlap.
    train_hashes = set(_row_hash_list(X_train_scaled, decimals=6))
    val_hashes = set(_row_hash_list(X_val_scaled, decimals=6))
    test_hashes = set(_row_hash_list(X_test_scaled, decimals=6))

    train_test_overlap = len(train_hashes & test_hashes)
    train_val_overlap = len(train_hashes & val_hashes)
    val_test_overlap = len(val_hashes & test_hashes)

    # Confirm val and test remain imbalanced (ignoring classes absent in a split, e.g. by_day)
    val_counts = np.bincount(y_val) if len(y_val) else np.array([1])
    test_counts = np.bincount(y_test) if len(y_test) else np.array([1])
    nz_val = val_counts[val_counts > 0]
    nz_test = test_counts[test_counts > 0]
    val_imbalance_ratio = float(nz_val.max() / max(1, nz_val.min())) if len(nz_val) else 1.0
    test_imbalance_ratio = float(nz_test.max() / max(1, nz_test.min())) if len(nz_test) else 1.0

    checks = {
        "train_standardized_max_abs_mean": round(max_train_mean, 6),
        "val_mean_shift_observed": round(val_mean_shift, 6),
        "train_test_exact_overlap_sample": train_test_overlap,
        "train_val_exact_overlap_sample": train_val_overlap,
        "val_test_exact_overlap_sample": val_test_overlap,
        "train_test_exact_overlap": train_test_overlap,
        "train_val_exact_overlap": train_val_overlap,
        "val_test_exact_overlap": val_test_overlap,
        "checked_full_splits": True,
        "strict_mode": bool(strict),
        "val_max_min_class_ratio": round(val_imbalance_ratio, 2),
        "test_max_min_class_ratio": round(test_imbalance_ratio, 2),
        "scaler_n_samples_seen": int(getattr(scaler, "n_samples_seen_", len(X_train_scaled))),
        "scaler_fit_strictly_on_train": int(getattr(scaler, "n_samples_seen_", 0)) == len(X_train_scaled),
        "passed": (
            max_train_mean < 1e-4
            and int(getattr(scaler, "n_samples_seen_", 0)) == len(X_train_scaled)
            and train_test_overlap == 0
            and train_val_overlap == 0
            and val_test_overlap == 0
        ),
    }
    if not checks["passed"]:
        logger.error("Leakage verification failed: %s", checks)
        if strict:
            raise DataLeakageError(
                "Zero-leakage verification failed: "
                f"train_test_overlap={train_test_overlap}, "
                f"train_val_overlap={train_val_overlap}, "
                f"val_test_overlap={val_test_overlap}, "
                f"max_train_mean={max_train_mean:.6f}, "
                f"scaler_fit_strictly_on_train={checks['scaler_fit_strictly_on_train']}."
            )
    else:
        logger.info(
            "Leakage verification passed across full splits: scaler fit exclusively on %d train rows, "
            "0 train/val/test feature overlap.",
            len(X_train_scaled),
        )
    return checks


def apply_smote_safely(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    max_k_neighbors: int = 5,
    sampling_strategy: str | dict[int, int] = "auto",
    cap_per_class: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply SMOTE strictly on training data with dynamically bounded k_neighbors."""
    unique_cls, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min())
    k_neighbors = max(1, min(max_k_neighbors, min_count - 1))

    effective_strategy: str | dict[int, int] = sampling_strategy
    if cap_per_class is not None and sampling_strategy == "auto":
        effective_strategy = {
            int(c): max(int(cnt), min(int(counts.max()), int(cap_per_class)))
            for c, cnt in zip(unique_cls, counts, strict=False)
        }

    if min_count <= 1:
        logger.warning("Smallest training class has <= 1 sample; skipping SMOTE.")
        return (
            X_train_scaled.copy(),
            y_train.copy(),
            {
                "applied": False,
                "reason": "minority_class_too_small",
                "k_neighbors": 0,
                "pre_counts": {str(int(c)): int(cnt) for c, cnt in zip(unique_cls, counts, strict=False)},
                "post_counts": {str(int(c)): int(cnt) for c, cnt in zip(unique_cls, counts, strict=False)},
                "rows_before": int(len(y_train)),
                "rows_after": int(len(y_train)),
            },
        )

    # Note: Modern imbalanced-learn (>=0.12) removed n_jobs from SMOTE().
    smote = SMOTE(
        sampling_strategy=effective_strategy,
        k_neighbors=k_neighbors,
        random_state=seed,
    )
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
    # Shuffle deterministically so resampled minority blocks are mixed uniformly
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_res))
    X_res = X_res[perm]
    y_res = y_res[perm]
    post_cls, post_counts = np.unique(y_res, return_counts=True)

    smote_info = {
        "applied": True,
        "k_neighbors": int(k_neighbors),
        "pre_counts": {str(int(c)): int(cnt) for c, cnt in zip(unique_cls, counts, strict=False)},
        "post_counts": {str(int(c)): int(cnt) for c, cnt in zip(post_cls, post_counts, strict=False)},
        "rows_before": int(len(y_train)),
        "rows_after": int(len(y_res)),
    }
    logger.info(
        "SMOTE applied on training split only (k_neighbors=%d): %d -> %d rows",
        k_neighbors,
        len(y_train),
        len(y_res),
    )
    return X_res.astype(np.float32), y_res.astype(np.int64), smote_info


def _classify_day_bucket(filename: str) -> str:
    """Map a CICIDS2017 CSV filename to train (Mon-Wed), Thursday (val/test), or Friday (test)."""
    name = Path(str(filename)).name.lower()
    if any(day in name for day in ("monday", "tuesday", "wednesday")):
        return "train_mon_wed"
    if "thursday" in name:
        return "thursday"
    if "friday" in name:
        return "friday"
    raise ValueError(
        f"Unable to map source file '{filename}' to Monday-Friday schedule for split_strategy='by_day'."
    )


def _split_indices_by_day(
    metadata: pd.DataFrame,
    y_all: np.ndarray,
    thursday_val_fraction: float,
    stratify: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Split CICIDS2017 rows chronologically by capture day (Mon-Wed / part of Thu / Thu+Fri)."""
    if "Source_File" not in metadata.columns:
        raise ValueError(
            "split_strategy='by_day' requires raw CSV filenames ('Source_File' metadata) identifying "
            "Monday-Wednesday, Thursday, and Friday files."
        )

    src_files = metadata["Source_File"].astype(str).tolist()
    buckets = [_classify_day_bucket(fname) for fname in src_files]
    idx_train = np.array([i for i, b in enumerate(buckets) if b == "train_mon_wed"], dtype=np.int64)
    idx_thu = np.array([i for i, b in enumerate(buckets) if b == "thursday"], dtype=np.int64)
    idx_fri = np.array([i for i, b in enumerate(buckets) if b == "friday"], dtype=np.int64)

    if len(idx_train) == 0 or len(idx_thu) == 0 or len(idx_fri) == 0:
        raise ValueError(
            "split_strategy='by_day' requires non-empty Monday-Wednesday train, Thursday validation, "
            f"and Friday test partitions (got train={len(idx_train)}, thursday={len(idx_thu)}, friday={len(idx_fri)})."
        )

    y_thu = y_all[idx_thu]
    thu_counts = np.bincount(y_thu)
    nz_thu = thu_counts[thu_counts > 0]
    n_val_approx = int(round(len(idx_thu) * thursday_val_fraction))
    n_test_approx = len(idx_thu) - n_val_approx
    can_stratify = (
        stratify
        and len(nz_thu) > 1
        and int(nz_thu.min()) >= 2
        and min(n_val_approx, n_test_approx) >= len(nz_thu)
    )
    if stratify and not can_stratify:
        logger.warning(
            "Thursday validation split falling back to unstratified split (class counts=%s).",
            nz_thu.tolist(),
        )

    idx_val, idx_thu_test = train_test_split(
        idx_thu,
        train_size=thursday_val_fraction,
        random_state=seed,
        stratify=y_thu if can_stratify else None,
    )
    idx_test = np.sort(np.concatenate([idx_thu_test, idx_fri]))
    split_details = {
        "train_files": sorted({src_files[i] for i in idx_train}),
        "val_files": sorted({src_files[i] for i in idx_val}),
        "test_files": sorted({src_files[i] for i in idx_test}),
    }
    logger.info(
        "Applied split_strategy='by_day': train=%d (%s) | val=%d (Thu split %.2f) | test=%d (Thu+Fri)",
        len(idx_train),
        split_details["train_files"],
        len(idx_val),
        thursday_val_fraction,
        len(idx_test),
    )
    return np.sort(idx_train), np.sort(idx_val), idx_test, split_details


def run_preprocessing_pipeline(
    cfg: AppConfig,
    force_synthetic: bool = False,
    save_outputs: bool = True,
) -> PreparedDataset:
    """Execute the full, leakage-free data engineering pipeline."""
    seed = cfg.project.random_seed
    raw_dir = cfg.paths.resolve("raw_data_dir")
    syn_dir = cfg.paths.resolve("synthetic_data_dir")
    proc_dir = cfg.paths.resolve("processed_data_dir")
    models_dir = cfg.paths.resolve("models_dir")
    artifacts_dir = cfg.paths.resolve("artifacts_dir")

    df_raw, source_info = load_cicids2017_corpus(
        raw_dir=raw_dir,
        synthetic_dir=syn_dir,
        data_cfg=cfg.data,
        seed=seed,
        force_synthetic=force_synthetic,
    )

    X_df, y_series, metadata, clean_report = clean_cicids_dataframe(
        df_raw,
        label_col=cfg.data.label_column,
        identifier_keywords=cfg.data.identifier_keywords,
        preserve_features=cfg.data.preserve_features,
        drop_features=cfg.data.drop_features,
        merge_web_attacks=cfg.data.merge_web_attacks,
        drop_ultra_rare_below=cfg.data.drop_ultra_rare_classes_below,
    )

    # Encode labels deterministically (LabelEncoder sorts class names alphabetically)
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_series.astype(str))
    class_names = [str(c) for c in label_encoder.classes_]

    # Original class distribution (post-cleaning, pre-split)
    orig_counts = {
        class_names[int(idx)]: int(cnt)
        for idx, cnt in zip(*np.unique(y_all, return_counts=True), strict=False)
    }

    # STEP 1 OF LEAKAGE PREVENTION: Split BEFORE fitting imputer, clipper, or scaler
    by_day_files: dict[str, Any] = {}
    if cfg.data.split_strategy == "by_day":
        idx_train, idx_val, idx_test, by_day_files = _split_indices_by_day(
            metadata=metadata,
            y_all=y_all,
            thursday_val_fraction=cfg.data.split.thursday_val_fraction,
            stratify=cfg.data.split.stratify,
            seed=seed,
        )
    else:
        strat = y_all if cfg.data.split.stratify else None
        idx_all = np.arange(len(X_df))
        idx_train_val, idx_test = train_test_split(
            idx_all,
            test_size=cfg.data.split.test_size,
            random_state=seed,
            stratify=strat,
        )
        strat_tv = y_all[idx_train_val] if cfg.data.split.stratify else None
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=cfg.data.split.val_size_of_remainder,
            random_state=seed,
            stratify=strat_tv,
        )

    X_train_raw = X_df.iloc[idx_train].reset_index(drop=True)
    X_val_raw = X_df.iloc[idx_val].reset_index(drop=True)
    X_test_raw = X_df.iloc[idx_test].reset_index(drop=True)

    y_train = y_all[idx_train].astype(np.int64)
    y_val = y_all[idx_val].astype(np.int64)
    y_test = y_all[idx_test].astype(np.int64)

    val_metadata = metadata.iloc[idx_val].reset_index(drop=True)
    test_metadata = metadata.iloc[idx_test].reset_index(drop=True)

    # STEP 2: Fit transformers strictly on X_train_raw
    col_dropper = HighMissingColumnDropper(nan_threshold=cfg.data.nan_column_drop_threshold)
    X_train_kept = col_dropper.fit_transform(X_train_raw)
    X_val_kept = col_dropper.transform(X_val_raw)
    X_test_kept = col_dropper.transform(X_test_raw)

    imputer = MedianImputerTransformer()
    X_train_imp = imputer.fit_transform(X_train_kept)
    nan_imputed_train = imputer.last_imputed_count_
    X_val_imp = imputer.transform(X_val_kept)
    nan_imputed_val = imputer.last_imputed_count_
    X_test_imp = imputer.transform(X_test_kept)
    nan_imputed_test = imputer.last_imputed_count_

    clipper = SigmaOutlierClipper(n_sigma=cfg.data.outlier_clip_sigma)
    X_train_clip = clipper.fit_transform(X_train_imp)
    clipped_train = clipper.last_clipped_count_
    X_val_clip = clipper.transform(X_val_imp)
    clipped_val = clipper.last_clipped_count_
    X_test_clip = clipper.transform(X_test_imp)
    clipped_test = clipper.last_clipped_count_

    # Deduplicate again after imputation/clipping so rows collapsing onto identical values
    # cannot cross train/val/test boundaries, BEFORE fitting StandardScaler or applying SMOTE.
    (
        X_train_clip,
        y_train,
        X_val_clip,
        y_val,
        val_metadata,
        X_test_clip,
        y_test,
        test_metadata,
        post_clip_stats,
    ) = _deduplicate_post_clip_splits(
        X_train_clip,
        y_train,
        X_val_clip,
        y_val,
        val_metadata,
        X_test_clip,
        y_test,
        test_metadata,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clip).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_clip).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_clip).astype(np.float32)

    (
        X_val_scaled,
        y_val,
        val_metadata,
        X_test_scaled,
        y_test,
        test_metadata,
        scaled_collision_stats,
    ) = _drop_scaled_cross_split_collisions(
        X_train_scaled,
        X_val_scaled,
        y_val,
        val_metadata,
        X_test_scaled,
        y_test,
        test_metadata,
        decimals=6,
    )
    post_clip_dedup_removed = {
        **post_clip_stats,
        **scaled_collision_stats,
        "total_removed": int(sum(post_clip_stats.values()) + sum(scaled_collision_stats.values())),
    }
    if post_clip_dedup_removed["total_removed"] > 0:
        logger.info("Post-clipping deduplication removed %s", post_clip_dedup_removed)

    feature_names = list(X_train_clip.columns)

    # STEP 3: Apply SMOTE strictly to standardized training partition
    X_train_sm, y_train_sm, smote_info = apply_smote_safely(
        X_train_scaled,
        y_train,
        max_k_neighbors=cfg.data.imbalance.smote_max_k_neighbors,
        sampling_strategy=cfg.data.imbalance.smote_sampling_strategy,
        cap_per_class=cfg.data.imbalance.smote_cap_per_class,
        seed=seed,
    )
    post_smote_named = {class_names[int(k)]: int(v) for k, v in smote_info["post_counts"].items()}
    pre_smote_named = {class_names[int(k)]: int(v) for k, v in smote_info["pre_counts"].items()}

    # Compute balanced class weights as an alternative to SMOTE for empirical comparison
    unique_train_cls = np.unique(y_train)
    cw_values = compute_class_weight(class_weight="balanced", classes=unique_train_cls, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(unique_train_cls, cw_values, strict=False)}

    leakage_checks = verify_no_leakage(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
        y_train,
        y_val,
        y_test,
        strict=cfg.data.strict_leakage_check,
    )

    audit_report: dict[str, Any] = {
        "source": source_info.to_dict(),
        "original_shape": [clean_report.initial_rows, clean_report.initial_cols],
        "rows_after_deduplication": clean_report.rows_after_dedup,
        "duplicates_removed": clean_report.duplicates_removed,
        "inf_values_converted": clean_report.inf_values_converted,
        "coerced_non_numeric_nans": clean_report.coerced_non_numeric_nans,
        "nans_before_split": clean_report.nans_before_split,
        "nan_imputed_train": nan_imputed_train,
        "nan_imputed_val": nan_imputed_val,
        "nan_imputed_test": nan_imputed_test,
        "outlier_values_clipped_train": clipped_train,
        "outlier_values_clipped_val": clipped_val,
        "outlier_values_clipped_test": clipped_test,
        "post_clip_dedup_removed": post_clip_dedup_removed,
        "split_strategy": cfg.data.split_strategy,
        "split_day_files": by_day_files,
        "dropped_features": clean_report.dropped_features_removed,
        "removed_columns": (
            clean_report.id_columns_removed
            + clean_report.duplicate_columns_removed
            + clean_report.dropped_features_removed
            + col_dropper.cols_to_drop_
        ),
        "id_columns_removed": clean_report.id_columns_removed,
        "high_nan_columns_dropped": col_dropper.cols_to_drop_,
        "final_feature_count": len(feature_names),
        "class_names": class_names,
        "split_classes_present": {
            "train": [class_names[int(c)] for c in np.unique(y_train)],
            "validation": [class_names[int(c)] for c in np.unique(y_val)],
            "test": [class_names[int(c)] for c in np.unique(y_test)],
        },
        "original_class_distribution": orig_counts,
        "train_pre_smote_distribution": pre_smote_named,
        "train_post_smote_distribution": post_smote_named,
        "split_sizes": {
            "train_raw": int(len(X_train_scaled)),
            "train_smote": int(len(X_train_sm)),
            "validation": int(len(X_val_scaled)),
            "test": int(len(X_test_scaled)),
        },
        "smote": smote_info,
        "leakage_checks": leakage_checks,
        "cleaning_details": clean_report.to_dict(),
    }

    if save_outputs:
        proc_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            proc_dir / "preprocessed_splits.npz",
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            X_train_sm=X_train_sm,
            y_train_sm=y_train_sm,
            X_val_scaled=X_val_scaled,
            y_val=y_val,
            X_test_scaled=X_test_scaled,
            y_test=y_test,
        )
        val_metadata.to_csv(proc_dir / "val_metadata.csv", index=False)
        test_metadata.to_csv(proc_dir / "test_metadata.csv", index=False)
        with (proc_dir / "preprocessing_audit.json").open("w", encoding="utf-8") as fh:
            json.dump(audit_report, fh, indent=2)

        registry = ModelRegistry(models_dir=models_dir, artifacts_dir=artifacts_dir)
        registry.save_preprocessing_artifacts(
            scaler=scaler,
            label_encoder=label_encoder,
            feature_names=feature_names,
            imputer=imputer,
            outlier_clipper=clipper,
        )
        logger.info("Saved preprocessed splits and audit report to %s", proc_dir)

    return PreparedDataset(
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_train_sm=X_train_sm,
        y_train_sm=y_train_sm,
        X_val_scaled=X_val_scaled,
        y_val=y_val,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        feature_names=feature_names,
        class_names=class_names,
        class_weights=class_weights,
        scaler=scaler,
        imputer=imputer,
        outlier_clipper=clipper,
        label_encoder=label_encoder,
        test_metadata=test_metadata,
        val_metadata=val_metadata,
        source_info=source_info,
        audit_report=audit_report,
    )
