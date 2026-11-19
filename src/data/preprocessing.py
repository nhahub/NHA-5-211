"""Leakage-free data splitting, scaling, SMOTE rebalancing, and audit reporting."""

from __future__ import annotations

import copy
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
    smote_class_weights: dict[int, float] | None = None
    train_row_indices: np.ndarray | None = None
    val_row_indices: np.ndarray | None = None
    test_row_indices: np.ndarray | None = None

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
    train_row_indices: np.ndarray | None = None,
    val_row_indices: np.ndarray | None = None,
    test_row_indices: np.ndarray | None = None,
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
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
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
        train_row_indices[train_keep] if train_row_indices is not None else None,
        val_row_indices[val_keep] if val_row_indices is not None else None,
        test_row_indices[test_keep] if test_row_indices is not None else None,
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
    val_row_indices: np.ndarray | None = None,
    test_row_indices: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    dict[str, int],
    np.ndarray | None,
    np.ndarray | None,
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
        val_row_indices[val_keep] if val_row_indices is not None else None,
        test_row_indices[test_keep] if test_row_indices is not None else None,
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
    max_multiplier: float | None = None,
    min_real_samples: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply SMOTE strictly on training data with config caps and a rare-class threshold.

    Classes with real training count below `min_real_samples` are NOT synthetically
    oversampled; they keep their real samples and rely on fallback class weights instead.
    """
    unique_cls, counts = np.unique(y_train, return_counts=True)
    count_map = {int(c): int(cnt) for c, cnt in zip(unique_cls, counts, strict=False)}
    max_count = int(counts.max()) if len(counts) else 0
    skipped_classes = {c: cnt for c, cnt in count_map.items() if cnt < min_real_samples}

    if sampling_strategy == "auto":
        effective_strategy: str | dict[int, int] = {}
        for c, cnt in count_map.items():
            if cnt < min_real_samples:
                effective_strategy[c] = cnt
                continue
            target = max_count
            if cap_per_class is not None:
                target = min(target, int(cap_per_class))
            if max_multiplier is not None:
                target = min(target, int(round(cnt * max_multiplier)))
            effective_strategy[c] = max(cnt, target)
    elif isinstance(sampling_strategy, dict):
        effective_strategy = {
            int(c): (count_map[int(c)] if count_map[int(c)] < min_real_samples else int(tgt))
            for c, tgt in sampling_strategy.items()
            if int(c) in count_map
        }
    else:
        effective_strategy = sampling_strategy

    if isinstance(effective_strategy, dict):
        upsampled_cls = [
            c for c, tgt in effective_strategy.items() if count_map.get(c, 0) >= 2 and tgt > count_map[c]
        ]
    else:
        upsampled_cls = [
            c for c, cnt in count_map.items() if cnt >= max(2, min_real_samples) and cnt < max_count
        ]

    if not upsampled_cls:
        reason = (
            "all_minority_classes_below_min_real_samples"
            if skipped_classes
            else "all_classes_already_at_or_above_target"
        )
        logger.info("Skipping SMOTE (%s); skipped_classes=%s.", reason, skipped_classes)
        return (
            X_train_scaled.copy(),
            y_train.copy(),
            {
                "applied": False,
                "reason": reason,
                "k_neighbors": 0,
                "min_real_samples": int(min_real_samples),
                "cap_per_class": cap_per_class,
                "max_multiplier": max_multiplier,
                "skipped_classes": {str(c): int(cnt) for c, cnt in skipped_classes.items()},
                "pre_counts": {str(c): int(cnt) for c, cnt in count_map.items()},
                "post_counts": {str(c): int(cnt) for c, cnt in count_map.items()},
                "rows_before": int(len(y_train)),
                "rows_after": int(len(y_train)),
            },
        )

    min_upsampled = min(count_map[c] for c in upsampled_cls)
    k_neighbors = max(1, min(max_k_neighbors, min_upsampled - 1))

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
        "min_real_samples": int(min_real_samples),
        "cap_per_class": cap_per_class,
        "max_multiplier": max_multiplier,
        "skipped_classes": {str(c): int(cnt) for c, cnt in skipped_classes.items()},
        "pre_counts": {str(c): int(cnt) for c, cnt in count_map.items()},
        "post_counts": {str(int(c)): int(cnt) for c, cnt in zip(post_cls, post_counts, strict=False)},
        "rows_before": int(len(y_train)),
        "rows_after": int(len(y_res)),
    }
    logger.info(
        "SMOTE applied on training split only (k_neighbors=%d, skipped_rare=%s): %d -> %d rows",
        k_neighbors,
        skipped_classes,
        len(y_train),
        len(y_res),
    )
    return X_res.astype(np.float32), y_res.astype(np.int64), smote_info


def compute_hybrid_smote_class_weights(
    smote_info: dict[str, Any],
    fallback_class_weights: dict[int, float],
) -> dict[int, float] | None:
    """Return class weights compensating for classes skipped by SMOTE (< min_real_samples)."""
    skipped = {int(k): int(v) for k, v in smote_info.get("skipped_classes", {}).items()}
    if not skipped:
        return None
    if not smote_info.get("applied", False):
        return {int(k): float(v) for k, v in fallback_class_weights.items()}

    post_counts = {int(k): int(v) for k, v in smote_info.get("post_counts", {}).items()}
    max_post = max(post_counts.values()) if post_counts else 1
    non_skipped_minorities = [
        cnt for cls_id, cnt in post_counts.items() if cls_id not in skipped and cnt < max_post
    ]
    ref_count = float(np.median(non_skipped_minorities)) if non_skipped_minorities else float(max_post)
    return {
        cls_id: (float(max(1.0, ref_count / max(1, cnt))) if cls_id in skipped else 1.0)
        for cls_id, cnt in post_counts.items()
    }


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
    """Split CICIDS2017 rows under the open-set cross-day protocol (Mon-Wed / part of Thu / Thu+Fri)."""
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
        "Applied open-set cross-day split_strategy='by_day': train=%d (%s) | val=%d (Thu split %.2f) | test=%d (Thu+Fri)",
        len(idx_train),
        split_details["train_files"],
        len(idx_val),
        thursday_val_fraction,
        len(idx_test),
    )
    return np.sort(idx_train), np.sort(idx_val), idx_test, split_details


def _parse_timestamp_series(ts_series: pd.Series) -> pd.Series:
    """Parse ISO or CICIDS2017 DD/MM/YYYY timestamp strings into naive datetimes."""
    raw = ts_series.astype(str).str.strip()
    valid_mask = raw.ne("") & raw.str.lower().ne("nan") & raw.str.lower().ne("nat")
    if not valid_mask.any():
        return pd.Series(pd.NaT, index=ts_series.index, dtype="datetime64[ns]")
    sample = raw[valid_mask]
    iso_ratio = float(sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean())
    if iso_ratio >= 0.5:
        return pd.to_datetime(raw, format="mixed", errors="coerce")
    return pd.to_datetime(raw, format="mixed", dayfirst=True, errors="coerce")


def _split_indices_time_block(
    metadata: pd.DataFrame,
    y_all: np.ndarray,
    class_names: list[str],
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Split each class chronologically (earliest ~60% train, middle ~20% val, latest ~20% test).

    Ordering precedence:
    1. Parsed ``Timestamp`` column (kept strictly in non-feature metadata and dropped from X).
    2. Pre-sample chronological ``Capture_Order`` (file session schedule + raw CSV row index)
       when MachineLearningCVE exports omit a ``Timestamp`` column.

    Tiny-class fallback policy (documented):
    - ``n >= 5``: ``round(n * train_fraction)`` train, ``round(n * val_fraction)`` val, remainder
      test, clamped so every partition retains at least 1 row.
    - ``n == 4``: 2 earliest -> train, 1 -> val, 1 -> test.
    - ``n == 3``: 1 earliest -> train, 1 -> val, 1 -> test.
    - ``n == 2``: 1 earliest -> train, 1 -> test (val omitted so both train and test cover class).
    - ``n == 1``: 1 -> train (val and test omitted to avoid eval-only classes).
    """
    n_total = len(y_all)
    if n_total == 0:
        raise ValueError("Cannot perform time_block split on empty dataset.")

    secondary_key = (
        metadata["Capture_Order"].to_numpy(dtype=np.int64)
        if "Capture_Order" in metadata.columns
        else np.arange(n_total, dtype=np.int64)
    )
    ordering_source = ""
    primary_key: np.ndarray | None = None

    if "Timestamp" in metadata.columns:
        parsed_ts = _parse_timestamp_series(metadata["Timestamp"])
        valid_frac = float(parsed_ts.notna().mean()) if n_total else 0.0
        n_unique_ts = int(parsed_ts.nunique(dropna=True))
        if valid_frac >= 0.95 and n_unique_ts >= 2:
            ts_ns = parsed_ts.dt.as_unit("ns").fillna(pd.Timestamp.max.as_unit("ns"))
            primary_key = ts_ns.to_numpy(dtype="datetime64[ns]").view(np.int64)
            ordering_source = "Timestamp"

    if primary_key is None:
        if "Capture_Order" in metadata.columns:
            primary_key = secondary_key
            ordering_source = "Capture_Order (capture session schedule + CSV row index; Timestamp column absent in MachineLearningCVE CSVs)"
        else:
            raise ValueError(
                "split_strategy='time_block' requires 'Timestamp' or 'Capture_Order' metadata to sort "
                "flows chronologically."
            )

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    per_class_allocations: dict[str, dict[str, int]] = {}
    tiny_class_fallbacks: dict[str, str] = {}

    for cls_id in np.unique(y_all):
        cls_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(cls_id)
        cls_idx = np.where(y_all == cls_id)[0]
        order = np.lexsort((secondary_key[cls_idx], primary_key[cls_idx]))
        cls_sorted = cls_idx[order]
        n = len(cls_sorted)

        if n >= 5:
            n_train = max(1, int(round(n * train_fraction)))
            n_val = max(1, int(round(n * val_fraction)))
            if n_train + n_val >= n:
                overflow = n_train + n_val - (n - 1)
                reduce_train = min(overflow, max(0, n_train - 1))
                n_train -= reduce_train
                n_val -= overflow - reduce_train
            n_test = n - n_train - n_val
            if n < 30:
                tiny_class_fallbacks[cls_name] = (
                    f"Low-support chronological split (n={n}): {n_train} earliest train / {n_val} middle val / {n_test} latest test"
                )
        elif n == 4:
            n_train, n_val, n_test = 2, 1, 1
            tiny_class_fallbacks[cls_name] = "Tiny-class fallback (n=4): 2 train / 1 val / 1 test"
        elif n == 3:
            n_train, n_val, n_test = 1, 1, 1
            tiny_class_fallbacks[cls_name] = "Tiny-class fallback (n=3): 1 train / 1 val / 1 test"
        elif n == 2:
            n_train, n_val, n_test = 1, 0, 1
            tiny_class_fallbacks[cls_name] = (
                "Tiny-class fallback (n=2): 1 earliest train / 0 val / 1 latest test"
            )
        else:
            n_train, n_val, n_test = 1, 0, 0
            tiny_class_fallbacks[cls_name] = (
                "Singleton fallback (n=1): retained in train only (0 val / 0 test)"
            )

        tr_slice = cls_sorted[:n_train]
        va_slice = cls_sorted[n_train : n_train + n_val]
        te_slice = cls_sorted[n_train + n_val :]
        train_parts.append(tr_slice)
        if len(va_slice) > 0:
            val_parts.append(va_slice)
        if len(te_slice) > 0:
            test_parts.append(te_slice)
        per_class_allocations[cls_name] = {
            "total": int(n),
            "train": int(len(tr_slice)),
            "validation": int(len(va_slice)),
            "test": int(len(te_slice)),
        }

    idx_train = np.sort(np.concatenate(train_parts)).astype(np.int64)
    idx_val = (
        np.sort(np.concatenate(val_parts)).astype(np.int64) if val_parts else np.empty(0, dtype=np.int64)
    )
    idx_test = (
        np.sort(np.concatenate(test_parts)).astype(np.int64) if test_parts else np.empty(0, dtype=np.int64)
    )

    if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
        raise ValueError(
            "split_strategy='time_block' produced an empty split partition "
            f"(train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)})."
        )

    split_details = {
        "ordering_source": ordering_source,
        "has_real_timestamps": bool(ordering_source == "Timestamp"),
        "ordering_warning": (
            ""
            if ordering_source == "Timestamp"
            else "time_block ordering = CSV row order (no real Timestamp)"
        ),
        "train_fraction_target": float(train_fraction),
        "val_fraction_target": float(val_fraction),
        "test_fraction_target": round(float(1.0 - train_fraction - val_fraction), 4),
        "tiny_class_fallbacks": tiny_class_fallbacks,
        "per_class_allocations": per_class_allocations,
    }
    logger.info(
        "Applied split_strategy='time_block' (%s): train=%d | val=%d | test=%d | fallbacks=%d",
        ordering_source,
        len(idx_train),
        len(idx_val),
        len(idx_test),
        len(tiny_class_fallbacks),
    )
    return idx_train, idx_val, idx_test, split_details


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
    time_block_details: dict[str, Any] = {}
    if cfg.data.split_strategy == "by_day":
        idx_train, idx_val, idx_test, by_day_files = _split_indices_by_day(
            metadata=metadata,
            y_all=y_all,
            thursday_val_fraction=cfg.data.split.thursday_val_fraction,
            stratify=cfg.data.split.stratify,
            seed=seed,
        )
    elif cfg.data.split_strategy == "time_block":
        idx_train, idx_val, idx_test, time_block_details = _split_indices_time_block(
            metadata=metadata,
            y_all=y_all,
            class_names=class_names,
            train_fraction=cfg.data.split.time_block_train_fraction,
            val_fraction=cfg.data.split.time_block_val_fraction,
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
        idx_train,
        idx_val,
        idx_test,
    ) = _deduplicate_post_clip_splits(
        X_train_clip,
        y_train,
        X_val_clip,
        y_val,
        val_metadata,
        X_test_clip,
        y_test,
        test_metadata,
        train_row_indices=np.asarray(idx_train, dtype=np.int64),
        val_row_indices=np.asarray(idx_val, dtype=np.int64),
        test_row_indices=np.asarray(idx_test, dtype=np.int64),
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
        idx_val,
        idx_test,
    ) = _drop_scaled_cross_split_collisions(
        X_train_scaled,
        X_val_scaled,
        y_val,
        val_metadata,
        X_test_scaled,
        y_test,
        test_metadata,
        decimals=6,
        val_row_indices=idx_val,
        test_row_indices=idx_test,
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
        max_multiplier=cfg.data.imbalance.smote_max_multiplier,
        min_real_samples=cfg.data.imbalance.smote_min_real_samples,
        seed=seed,
    )
    post_smote_named = {class_names[int(k)]: int(v) for k, v in smote_info["post_counts"].items()}
    pre_smote_named = {class_names[int(k)]: int(v) for k, v in smote_info["pre_counts"].items()}
    skipped_named = {class_names[int(k)]: int(v) for k, v in smote_info.get("skipped_classes", {}).items()}
    smote_info["skipped_classes_named"] = skipped_named

    # Compute balanced class weights as an alternative to SMOTE and as fallback for skipped rare classes
    unique_train_cls = np.unique(y_train)
    cw_values = compute_class_weight(class_weight="balanced", classes=unique_train_cls, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(unique_train_cls, cw_values, strict=False)}
    smote_class_weights = compute_hybrid_smote_class_weights(smote_info, class_weights)
    if smote_class_weights:
        smote_info["hybrid_class_weights_named"] = {
            class_names[int(k)]: round(float(v), 4)
            for k, v in smote_class_weights.items()
            if k in map(int, smote_info.get("skipped_classes", {}))
        }

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
        "timestamp_alignment": source_info.timestamp_alignment,
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
        "time_block_details": time_block_details,
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
        smote_class_weights=smote_class_weights,
        train_row_indices=idx_train,
        val_row_indices=idx_val,
        test_row_indices=idx_test,
    )


def drop_features_from_prepared_dataset(
    dataset: PreparedDataset,
    drop_features: list[str],
    cfg: AppConfig,
) -> PreparedDataset:
    """Create an apples-to-apples ablated PreparedDataset by dropping columns after full-feature checks.

    Preserves the exact post-deduplication train/validation/test row indices from the verified
    full-feature dataset so feature ablation evaluates identical samples across variants.
    """
    full_checks = dataset.audit_report.get("leakage_checks", {})
    if not full_checks.get("passed", False):
        raise DataLeakageError(
            "Source dataset must pass full-feature zero-leakage checks before dropping ablated features."
        )

    drop_lower = {str(col).strip().lower() for col in drop_features if str(col).strip()}
    if not drop_lower:
        raise ValueError("drop_features must contain at least one non-empty feature name.")

    dropped_names = [f for f in dataset.feature_names if f.strip().lower() in drop_lower]
    if not dropped_names:
        raise ValueError(f"None of drop_features={drop_features} matched dataset features.")

    keep_mask = np.array([f.strip().lower() not in drop_lower for f in dataset.feature_names], dtype=bool)
    if not np.any(keep_mask):
        raise ValueError("Cannot drop every feature from the prepared dataset.")

    new_feature_names = [f for f, keep in zip(dataset.feature_names, keep_mask, strict=True) if keep]
    X_train_sub = dataset.X_train_scaled[:, keep_mask].copy()
    X_val_sub = dataset.X_val_scaled[:, keep_mask].copy()
    X_test_sub = dataset.X_test_scaled[:, keep_mask].copy()

    seed = cfg.project.random_seed
    X_train_sm_sub, y_train_sm_sub, smote_info = apply_smote_safely(
        X_train_sub,
        dataset.y_train,
        max_k_neighbors=cfg.data.imbalance.smote_max_k_neighbors,
        sampling_strategy=cfg.data.imbalance.smote_sampling_strategy,
        cap_per_class=cfg.data.imbalance.smote_cap_per_class,
        max_multiplier=cfg.data.imbalance.smote_max_multiplier,
        min_real_samples=cfg.data.imbalance.smote_min_real_samples,
        seed=seed,
    )
    smote_info["skipped_classes_named"] = {
        dataset.class_names[int(k)]: int(v) for k, v in smote_info.get("skipped_classes", {}).items()
    }
    post_smote_named = {
        dataset.class_names[int(k)]: int(v) for k, v in smote_info.get("post_counts", {}).items()
    }
    smote_class_weights = compute_hybrid_smote_class_weights(smote_info, dataset.class_weights)
    if smote_class_weights:
        smote_info["hybrid_class_weights_named"] = {
            dataset.class_names[int(k)]: round(float(v), 4)
            for k, v in smote_class_weights.items()
            if k in map(int, smote_info.get("skipped_classes", {}))
        }

    audit_sub = copy.deepcopy(dataset.audit_report)
    audit_sub["dropped_features"] = dropped_names
    audit_sub["removed_columns"] = list(audit_sub.get("removed_columns", [])) + dropped_names
    audit_sub["final_feature_count"] = len(new_feature_names)
    audit_sub["train_post_smote_distribution"] = post_smote_named
    audit_sub["split_sizes"] = {
        "train_raw": int(len(X_train_sub)),
        "train_smote": int(len(X_train_sm_sub)),
        "validation": int(len(X_val_sub)),
        "test": int(len(X_test_sub)),
    }
    audit_sub["smote"] = smote_info
    audit_sub["full_feature_leakage_checks"] = copy.deepcopy(full_checks)
    audit_sub["ablation_exact_row_preservation"] = True

    return PreparedDataset(
        X_train_scaled=X_train_sub,
        y_train=dataset.y_train.copy(),
        X_train_sm=X_train_sm_sub,
        y_train_sm=y_train_sm_sub,
        X_val_scaled=X_val_sub,
        y_val=dataset.y_val.copy(),
        X_test_scaled=X_test_sub,
        y_test=dataset.y_test.copy(),
        feature_names=new_feature_names,
        class_names=list(dataset.class_names),
        class_weights=dict(dataset.class_weights),
        scaler=dataset.scaler,
        imputer=dataset.imputer,
        outlier_clipper=dataset.outlier_clipper,
        label_encoder=dataset.label_encoder,
        test_metadata=dataset.test_metadata.copy(),
        val_metadata=dataset.val_metadata.copy(),
        source_info=dataset.source_info,
        audit_report=audit_sub,
        smote_class_weights=smote_class_weights,
        train_row_indices=dataset.train_row_indices.copy() if dataset.train_row_indices is not None else None,
        val_row_indices=dataset.val_row_indices.copy() if dataset.val_row_indices is not None else None,
        test_row_indices=dataset.test_row_indices.copy() if dataset.test_row_indices is not None else None,
    )
