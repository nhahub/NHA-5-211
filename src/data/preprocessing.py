"""Leakage-free data splitting, scaling, SMOTE rebalancing, and audit reporting."""

from __future__ import annotations

from dataclasses import dataclass
import json
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


def verify_no_leakage(
    X_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    scaler: StandardScaler,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Run automated verification assertions confirming zero train/eval leakage."""
    train_means = np.abs(X_train_scaled.mean(axis=0))
    max_train_mean = float(np.max(train_means))
    val_mean_shift = float(np.max(np.abs(X_val_scaled.mean(axis=0))))

    # Check that train is standardized near 0 mean, and verify unique row hashes don't leak across splits
    # after deduplication
    def _row_hashes(arr: np.ndarray, limit: int = 50000) -> set[bytes]:
        sub = np.round(arr[:limit], 6)
        return {row.tobytes() for row in sub}

    train_hashes = _row_hashes(X_train_scaled)
    val_hashes = _row_hashes(X_val_scaled)
    test_hashes = _row_hashes(X_test_scaled)

    train_test_overlap = len(train_hashes & test_hashes)
    train_val_overlap = len(train_hashes & val_hashes)

    # Confirm val and test remain imbalanced (non-uniform counts)
    val_counts = np.bincount(y_val)
    test_counts = np.bincount(y_test)
    val_imbalance_ratio = float(val_counts.max() / max(1, val_counts.min()))
    test_imbalance_ratio = float(test_counts.max() / max(1, test_counts.min()))

    checks = {
        "train_standardized_max_abs_mean": round(max_train_mean, 6),
        "val_mean_shift_observed": round(val_mean_shift, 6),
        "train_test_exact_overlap_sample": train_test_overlap,
        "train_val_exact_overlap_sample": train_val_overlap,
        "val_max_min_class_ratio": round(val_imbalance_ratio, 2),
        "test_max_min_class_ratio": round(test_imbalance_ratio, 2),
        "scaler_n_samples_seen": int(getattr(scaler, "n_samples_seen_", len(X_train_scaled))),
        "scaler_fit_strictly_on_train": int(getattr(scaler, "n_samples_seen_", 0)) == len(X_train_scaled),
        "passed": (
            max_train_mean < 1e-4
            and int(getattr(scaler, "n_samples_seen_", 0)) == len(X_train_scaled)
            and train_test_overlap == 0
        ),
    }
    if not checks["passed"]:
        logger.warning("Leakage verification flagged an anomaly: %s", checks)
    else:
        logger.info(
            "Leakage verification passed: scaler fit exclusively on %d train rows, 0 train/test overlap.",
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

    # STEP 1 OF LEAKAGE PREVENTION: Stratified split BEFORE fitting imputer, clipper, or scaler
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clip).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_clip).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_clip).astype(np.float32)

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
        X_train_scaled, X_val_scaled, X_test_scaled, scaler, y_train, y_val, y_test
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
        "removed_columns": (
            clean_report.id_columns_removed
            + clean_report.duplicate_columns_removed
            + col_dropper.cols_to_drop_
        ),
        "id_columns_removed": clean_report.id_columns_removed,
        "high_nan_columns_dropped": col_dropper.cols_to_drop_,
        "final_feature_count": len(feature_names),
        "class_names": class_names,
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
