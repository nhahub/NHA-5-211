"""SHAP TreeExplainer, neural Gradient/KernelExplainer comparison, and waterfall explanations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap

from src.utils.logging import get_logger

logger = get_logger("smart_ids.explainability.shap")


@dataclass
class GlobalSHAPResult:
    """Global feature attribution summary and raw SHAP values."""

    model_name: str
    explainer_type: str
    feature_importance: list[dict[str, float | str]]
    mean_abs_shap: np.ndarray
    n_samples_explained: int
    elapsed_sec: float


def _collapse_multiclass_shap(raw_values: Any) -> np.ndarray:
    """Normalize SHAP output across versions (list, 3D ndarray, Explanation, 2D ndarray) -> (n_samples, n_features)."""
    vals = getattr(raw_values, "values", raw_values)
    if isinstance(vals, list):
        stacked = np.stack([np.asarray(v, dtype=np.float64) for v in vals], axis=-1)
        return np.mean(np.abs(stacked), axis=-1)
    arr = np.asarray(vals, dtype=np.float64)
    if arr.ndim == 3:
        # shape: (n_samples, n_features, n_classes)
        return np.mean(np.abs(arr), axis=2)
    if arr.ndim == 2:
        return np.abs(arr)
    raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")


def _extract_class_shap_2d(raw_values: Any, class_idx: int) -> np.ndarray:
    """Extract (n_samples, n_features) signed SHAP matrix for a specific target class."""
    vals = getattr(raw_values, "values", raw_values)
    if isinstance(vals, list):
        return np.asarray(vals[class_idx], dtype=np.float64)
    arr = np.asarray(vals, dtype=np.float64)
    if arr.ndim == 3:
        return arr[:, :, class_idx]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")


def _extract_base_value(expected_value: Any, class_idx: int) -> float:
    """Extract scalar base value for a target class from explainer.expected_value."""
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        arr = np.asarray(expected_value, dtype=np.float64).flatten()
        if len(arr) > class_idx:
            return float(arr[class_idx])
        return float(arr[0])
    return float(expected_value)


def compute_xgboost_global_shap(
    xgb_model: Any,
    X_test: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    max_samples: int = 500,
    summary_fig_path: Path | None = None,
) -> tuple[GlobalSHAPResult, Any, Any]:
    """Compute exact global SHAP values for XGBoost via TreeExplainer."""
    n_shap = min(max_samples, len(X_test))
    X_sub = pd.DataFrame(X_test[:n_shap], columns=feature_names)

    t0 = time.perf_counter()
    explainer = shap.TreeExplainer(xgb_model)
    raw_shap = explainer.shap_values(X_sub)
    elapsed = float(time.perf_counter() - t0)

    abs_matrix = _collapse_multiclass_shap(raw_shap)
    mean_abs = np.mean(abs_matrix, axis=0)
    order = np.argsort(mean_abs)[::-1]

    ranking = [{"feature": feature_names[int(i)], "mean_abs_shap": float(mean_abs[int(i)])} for i in order]

    if summary_fig_path is not None:
        summary_fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
        # Left: top-15 global importance bar plot
        top_k = min(15, len(order))
        top_idx = order[:top_k][::-1]
        axes[0].barh(
            [feature_names[int(i)] for i in top_idx],
            [mean_abs[int(i)] for i in top_idx],
            color="#38bdf8",
        )
        axes[0].set_title(f"XGBoost Global SHAP Importance (Top {top_k}, n={n_shap})", fontweight="bold")
        axes[0].set_xlabel("Mean |SHAP value| across all classes")

        # Right: attack-class directional scatter/beeswarm for first non-benign class
        non_benign_idx = next(
            (idx for idx, cname in enumerate(class_names) if "benign" not in cname.lower()),
            0,
        )
        cls_shap = _extract_class_shap_2d(raw_shap, non_benign_idx)
        cls_title = class_names[non_benign_idx]
        top_idx_beeswarm = order[: min(12, len(order))][::-1]
        for row_pos, feat_i in enumerate(top_idx_beeswarm):
            y_jitter = np.full(n_shap, row_pos, dtype=np.float64) + np.random.default_rng(
                42 + row_pos
            ).uniform(-0.22, 0.22, size=n_shap)
            sc = axes[1].scatter(
                cls_shap[:, int(feat_i)],
                y_jitter,
                c=X_sub.iloc[:, int(feat_i)].to_numpy(),
                cmap="coolwarm",
                s=10,
                alpha=0.65,
            )
        axes[1].axvline(0.0, color="#334155", linestyle="--", lw=1)
        axes[1].set_yticks(range(len(top_idx_beeswarm)))
        axes[1].set_yticklabels([feature_names[int(i)] for i in top_idx_beeswarm])
        axes[1].set_title(f"XGBoost Directional SHAP — '{cls_title}'", fontweight="bold")
        axes[1].set_xlabel(f"SHAP value (impact on '{cls_title}' log-odds/margin)")
        fig.colorbar(sc, ax=axes[1], label="Standardized feature value")

        plt.tight_layout()
        fig.savefig(summary_fig_path, dpi=170)
        plt.close(fig)

    res = GlobalSHAPResult(
        model_name="XGBoost",
        explainer_type="TreeExplainer",
        feature_importance=ranking,
        mean_abs_shap=mean_abs,
        n_samples_explained=n_shap,
        elapsed_sec=elapsed,
    )
    logger.info("XGBoost TreeExplainer completed %d samples in %.2fs", n_shap, elapsed)
    return res, explainer, raw_shap


def compute_mlp_global_shap(
    mlp_model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    n_background: int = 50,
    n_explain: int = 20,
    fig_path: Path | None = None,
    benchmark_kernel: bool = True,
) -> tuple[GlobalSHAPResult, dict[str, Any]]:
    """Compute MLP global SHAP values preferring fast GradientExplainer with KernelExplainer benchmark."""
    bg = np.asarray(X_background[: min(n_background, len(X_background))], dtype=np.float32)
    X_sub = np.asarray(X_explain[: min(n_explain, len(X_explain))], dtype=np.float32)

    benchmark_info: dict[str, Any] = {}
    mean_abs_primary: np.ndarray | None = None
    explainer_used = "GradientExplainer"
    elapsed_primary = 0.0

    # 1. Try GradientExplainer (orders of magnitude faster for deep models)
    try:
        t0 = time.perf_counter()
        grad_exp = shap.GradientExplainer(mlp_model, bg)
        raw_grad = grad_exp.shap_values(X_sub)
        elapsed_primary = float(time.perf_counter() - t0)
        mean_abs_primary = np.mean(_collapse_multiclass_shap(raw_grad), axis=0)
        benchmark_info["gradient_explainer"] = {
            "status": "success",
            "n_background": int(len(bg)),
            "n_explained": int(len(X_sub)),
            "elapsed_sec": round(elapsed_primary, 4),
        }
    except Exception as exc:
        logger.warning("GradientExplainer unavailable for MLP (%s); falling back to KernelExplainer.", exc)
        benchmark_info["gradient_explainer"] = {"status": f"failed: {exc}"}

    # 2. Benchmark KernelExplainer on small subset (Report Ch. 9.3: 10 background, <=20 explain)
    mean_abs_kernel: np.ndarray | None = None
    if benchmark_kernel or mean_abs_primary is None:
        bg_k = bg[: min(10, len(bg))]
        X_k = X_sub[: min(10 if mean_abs_primary is not None else len(X_sub), len(X_sub))]

        def _predict_fn(x: np.ndarray) -> np.ndarray:
            return mlp_model.predict(np.asarray(x, dtype=np.float32), verbose=0)

        t0 = time.perf_counter()
        kern_exp = shap.KernelExplainer(_predict_fn, bg_k)
        raw_kern = kern_exp.shap_values(X_k, silent=True)
        elapsed_k = float(time.perf_counter() - t0)
        mean_abs_kernel = np.mean(_collapse_multiclass_shap(raw_kern), axis=0)
        benchmark_info["kernel_explainer"] = {
            "status": "success",
            "n_background": int(len(bg_k)),
            "n_explained": int(len(X_k)),
            "elapsed_sec": round(elapsed_k, 4),
        }
        if mean_abs_primary is None:
            mean_abs_primary = mean_abs_kernel
            explainer_used = "KernelExplainer"
            elapsed_primary = elapsed_k

    assert mean_abs_primary is not None
    if mean_abs_kernel is not None and explainer_used == "GradientExplainer":
        corr, _ = spearmanr(mean_abs_primary, mean_abs_kernel)
        benchmark_info["spearman_rank_agreement"] = round(float(0.0 if np.isnan(corr) else corr), 4)
        k_time = benchmark_info["kernel_explainer"]["elapsed_sec"]
        benchmark_info["speedup_vs_kernel"] = round(float(k_time / max(elapsed_primary, 1e-6)), 2)

    order = np.argsort(mean_abs_primary)[::-1]
    ranking = [
        {"feature": feature_names[int(i)], "mean_abs_shap": float(mean_abs_primary[int(i)])} for i in order
    ]

    if fig_path is not None:
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        top_k = min(15, len(order))
        top_idx = order[:top_k][::-1]
        vals = [mean_abs_primary[int(i)] for i in top_idx]
        names = [feature_names[int(i)] for i in top_idx]

        fig, ax = plt.subplots(figsize=(10, 6.5))
        colors = plt.cm.YlOrRd(np.linspace(0.35, 0.9, top_k))
        bars = ax.barh(names, vals, color=colors)
        for bar, val in zip(bars, vals, strict=False):
            ax.text(
                val + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=8,
            )
        ax.set_title(f"MLP Top-{top_k} Global SHAP Importance ({explainer_used})", fontweight="bold")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_xlim(0, max(vals) * 1.16 if max(vals) > 0 else 1.0)
        plt.tight_layout()
        fig.savefig(fig_path, dpi=170)
        plt.close(fig)

    res = GlobalSHAPResult(
        model_name="MLP",
        explainer_type=explainer_used,
        feature_importance=ranking,
        mean_abs_shap=mean_abs_primary,
        n_samples_explained=int(len(X_sub)),
        elapsed_sec=elapsed_primary,
    )
    logger.info("MLP SHAP (%s) completed in %.2fs", explainer_used, elapsed_primary)
    return res, benchmark_info


def explain_single_sample(
    model_name: str,
    model: Any,
    x_single: np.ndarray,
    X_background: np.ndarray,
    feature_names: list[str],
    class_idx: int,
    class_name: str,
    waterfall_path: Path | None = None,
    max_display: int = 15,
) -> dict[str, Any]:
    """Compute exact local SHAP feature contributions and optional waterfall plot for 1 flow."""
    row_2d = np.asarray(x_single, dtype=np.float32).reshape(1, -1)
    bg = np.asarray(X_background[: min(50, len(X_background))], dtype=np.float32)

    if model_name == "XGBoost":
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(pd.DataFrame(row_2d, columns=feature_names))
        shap_1d = _extract_class_shap_2d(raw, class_idx)[0]
        base_val = _extract_base_value(explainer.expected_value, class_idx)
        explainer_used = "TreeExplainer"
    else:
        # MLP / deep tabular classifier
        try:
            grad_exp = shap.GradientExplainer(model, bg)
            raw = grad_exp.shap_values(row_2d)
            shap_1d = _extract_class_shap_2d(raw, class_idx)[0]
            base_preds = model.predict(bg, verbose=0)
            base_val = float(np.mean(base_preds[:, class_idx]))
            explainer_used = "GradientExplainer"
        except Exception:
            bg_k = bg[: min(10, len(bg))]
            kern_exp = shap.KernelExplainer(
                lambda z: model.predict(np.asarray(z, dtype=np.float32), verbose=0), bg_k
            )
            raw = kern_exp.shap_values(row_2d, silent=True)
            shap_1d = _extract_class_shap_2d(raw, class_idx)[0]
            base_val = _extract_base_value(kern_exp.expected_value, class_idx)
            explainer_used = "KernelExplainer"

    order = np.argsort(np.abs(shap_1d))[::-1]
    top_features = [
        {
            "feature": feature_names[int(i)],
            "shap_value": float(shap_1d[int(i)]),
            "abs_shap": float(abs(shap_1d[int(i)])),
            "scaled_value": float(row_2d[0, int(i)]),
        }
        for i in order[:max_display]
    ]

    if waterfall_path is not None:
        waterfall_path.parent.mkdir(parents=True, exist_ok=True)
        explanation = shap.Explanation(
            values=shap_1d,
            base_values=base_val,
            data=row_2d[0],
            feature_names=feature_names,
        )
        plt.figure(figsize=(11, 7))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.title(f"{model_name} Local Explanation — Predicted '{class_name}'", fontweight="bold")
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=170, bbox_inches="tight")
        plt.close("all")

    return {
        "model": model_name,
        "explainer": explainer_used,
        "target_class": class_name,
        "target_class_idx": int(class_idx),
        "base_value": float(base_val),
        "shap_sum_plus_base": float(base_val + np.sum(shap_1d)),
        "top_features": top_features,
        "all_shap_values": shap_1d.tolist(),
    }
