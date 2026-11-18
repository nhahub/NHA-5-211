"""Controlled latency, throughput, and model size benchmarking."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd


def benchmark_preprocessing_latency(
    raw_df_sample: pd.DataFrame,
    feature_names: list[str],
    imputer: Any,
    clipper: Any,
    scaler: Any,
    n_warmup: int = 5,
    n_runs: int = 100,
) -> dict[str, float]:
    """Benchmark single-flow preprocessing latency separately from model inference."""
    single_df = raw_df_sample[feature_names].iloc[:1].copy()

    for _ in range(n_warmup):
        x = imputer.transform(single_df)
        x = clipper.transform(x)
        _ = scaler.transform(x)

    times_us: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        x = imputer.transform(single_df)
        x = clipper.transform(x)
        _ = scaler.transform(x)
        times_us.append((time.perf_counter() - t0) * 1e6)

    arr = np.asarray(times_us, dtype=np.float64)
    mean_us = float(np.mean(arr))
    return {
        "mean_us": mean_us,
        "median_us": float(np.median(arr)),
        "p95_us": float(np.percentile(arr, 95)),
        "samples_per_sec": float(1e6 / max(mean_us, 1e-9)),
    }


def benchmark_model_inference(
    model: Any,
    X_test: np.ndarray,
    model_type: str,
    model_path: Path | None = None,
    n_warmup: int = 5,
    n_runs: int = 100,
    batch_size: int = 1024,
) -> dict[str, float]:
    """Measure warm-up-corrected single-sample latency and batch throughput."""
    single = np.asarray(X_test[:1], dtype=np.float32)
    batch = np.asarray(X_test[: min(batch_size, len(X_test))], dtype=np.float32)

    if model_type == "cnn":
        single_in = single.reshape(1, single.shape[1], 1)
        batch_in = batch.reshape(batch.shape[0], batch.shape[1], 1)
    else:
        single_in = single
        batch_in = batch

    def _infer_single() -> Any:
        if model_type in ("keras", "cnn", "ae"):
            # Direct model(..., training=False) avoids Keras Dataset wrapper overhead on 1-row streaming flows
            return model(single_in, training=False)
        if model_type == "sklearn_anomaly":
            return -model.decision_function(single_in)
        return model.predict_proba(single_in)

    def _infer_batch() -> Any:
        if model_type in ("keras", "cnn", "ae"):
            return model.predict(batch_in, batch_size=512, verbose=0)
        if model_type == "sklearn_anomaly":
            return -model.decision_function(batch_in)
        return model.predict_proba(batch_in)

    # Warm-up runs (JIT / graph compilation / cache warm)
    for _ in range(n_warmup):
        _ = _infer_single()

    times_us: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = _infer_single()
        times_us.append((time.perf_counter() - t0) * 1e6)

    arr = np.asarray(times_us, dtype=np.float64)
    mean_us = float(np.mean(arr))

    # Measure batch throughput across 3 repeated calls
    _ = _infer_batch()
    batch_times: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = _infer_batch()
        batch_times.append(time.perf_counter() - t0)
    batch_sec = float(np.mean(batch_times))
    batch_sps = float(len(batch_in) / max(batch_sec, 1e-9))

    size_mb = 0.0
    if model_path is not None and model_path.exists():
        size_mb = float(model_path.stat().st_size / (1024.0 * 1024.0))

    return {
        "single_mean_us": mean_us,
        "single_median_us": float(np.median(arr)),
        "single_p95_us": float(np.percentile(arr, 95)),
        "single_samples_per_sec": float(1e6 / max(mean_us, 1e-9)),
        "batch_samples_per_sec": batch_sps,
        "model_size_mb": round(size_mb, 3),
    }
