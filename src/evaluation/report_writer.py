"""Markdown academic report generator derived strictly from actual evaluation metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Format a DataFrame as a GitHub-Flavored Markdown table without external dependencies."""
    if df.empty:
        return "_No rows available._"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for row in df.itertuples(index=False, name=None):
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def write_academic_reports(
    summary: dict[str, Any],
    comparison_df: pd.DataFrame,
    reports_dir: Path,
) -> tuple[Path, Path]:
    """Write model_comparison_report.md and experimental_results.md from measured results."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    src = summary.get("dataset_source", {})
    audit = summary.get("preprocessing_audit", {})
    diag = summary.get("training_diagnostics", {})
    ens = summary.get("ensemble_comparison", {})
    ae = summary.get("autoencoder_evaluation", {})
    unseen = summary.get("unseen_attack_protocol", {})
    prep_lat = summary.get("preprocessing_latency", {})
    shap_sum = summary.get("shap_summary", {})

    is_synth = bool(src.get("is_synthetic", False))
    mode_banner = (
        "> **Mode Notice — Synthetic Demonstration Run:** Raw CICIDS2017 CSV files were not "
        "present in `data/raw/`. Every metric below was measured on the schema-faithful synthetic "
        "demonstration corpus (`data/synthetic/cicids2017_synthetic_demo.csv`) and is strictly "
        "distinguished from genuine CICIDS2017 benchmark results.\n"
        if is_synth
        else "> **Mode Notice — Genuine CICIDS2017 Benchmark Run:** All metrics below were measured "
        "on held-out CICIDS2017 evaluation data.\n"
    )

    cmp_md = _df_to_markdown(comparison_df)

    # 1. Model Comparison Report
    comp_lines = [
        "# SMART IDS v3 — Model Comparison Report\n",
        mode_banner,
        "## 1. Unified Four-Model Benchmark\n",
        "- **Supervised models (MLP, 1D-CNN, XGBoost):** Evaluated on multi-class attack attribution using weighted and macro metrics.\n",
        "- **Unsupervised Autoencoder:** Evaluated separately under binary novelty framing (Benign vs. Anomalous).\n\n",
        cmp_md + "\n\n",
        "## 2. Ensemble Strategy Comparison (Test Partition)\n",
    ]
    if ens:
        ens_df = pd.DataFrame(ens).T.reset_index(names="Strategy")
        comp_lines.append(_df_to_markdown(ens_df) + "\n\n")

    comp_lines.append("## 3. XGBoost Baseline vs. Optuna Hyperparameter Optimization\n")
    xgb_opt = diag.get("xgboost_optimization", {})
    if xgb_opt:
        opt_df = pd.DataFrame(
            [
                {
                    "Variant": "Baseline (300 trees, depth=6, lr=0.1)",
                    **xgb_opt.get("baseline_val_metrics", {}),
                },
                {
                    "Variant": f"Optuna Tuned ({xgb_opt.get('n_trials_completed', 0)} trials)",
                    **xgb_opt.get("tuned_val_metrics", {}),
                },
            ]
        )
        comp_lines.append(_df_to_markdown(opt_df) + "\n")
        comp_lines.append(f"\n- **Selected Variant:** `{xgb_opt.get('selected_variant', 'baseline')}`\n\n")

    comp_lines.append("## 4. Architectural Notes on 1D-CNN vs. MLP & XGBoost\n")
    comp_lines.append(
        "Tabular CICFlowMeter features do not form a physical time series; 1D convolutions scan "
        "adjacent engineered statistics (packet sizes, IATs, flags). Measured validation and test "
        "F1/latency above quantify whether 1D-CNN yields measurable benefit over the dense MLP and XGBoost.\n\n"
    )

    comp_lines.append("## 5. Preprocessing vs. Model Inference Latency\n")
    if prep_lat:
        comp_lines.append(
            f"- **Isolated Single-Flow Preprocessing (Imputer + 5-Sigma Clipper + StandardScaler):** "
            f"mean = `{prep_lat.get('mean_us', 0):.2f} µs`, median = `{prep_lat.get('median_us', 0):.2f} µs`, "
            f"P95 = `{prep_lat.get('p95_us', 0):.2f} µs` (`{prep_lat.get('samples_per_sec', 0):.1f}` flows/sec).\n"
        )

    cmp_path = reports_dir / "model_comparison_report.md"
    cmp_path.write_text("\n".join(comp_lines), encoding="utf-8")

    # 2. Experimental Results Report
    exp_lines = [
        "# SMART IDS v3 — Experimental Results Report\n",
        mode_banner,
        "## 1. Data Engineering & Leakage Safeguard Audit\n",
        f"- **Original Dataset Dimensions:** `{audit.get('original_shape', [])}`\n",
        f"- **Rows After Exact Deduplication:** `{audit.get('rows_after_deduplication', 0):,}` (removed `{audit.get('duplicates_removed', 0):,}` duplicates)\n",
        f"- **Infinity Values Converted to NaN:** `{audit.get('inf_values_converted', 0):,}`\n",
        f"- **Coerced Non-Numeric NaNs:** `{audit.get('coerced_non_numeric_nans', 0):,}`\n",
        f"- **NaNs Imputed (Train-Only Median Fit):** train=`{audit.get('nan_imputed_train', 0):,}`, val=`{audit.get('nan_imputed_val', 0):,}`, test=`{audit.get('nan_imputed_test', 0):,}`\n",
        f"- **Outlier Values Clipped (Train-Only 5-Sigma Bounds):** train=`{audit.get('outlier_values_clipped_train', 0):,}`, val=`{audit.get('outlier_values_clipped_val', 0):,}`, test=`{audit.get('outlier_values_clipped_test', 0):,}`\n",
        f"- **Removed Structural / Redundant Columns:** `{audit.get('removed_columns', [])}`\n",
        f"- **Final Feature Count:** `{audit.get('final_feature_count', 0)}`\n",
        f"- **Leakage Verification Passed:** `{audit.get('leakage_checks', {}).get('passed')}`\n\n",
    ]

    split_sizes = audit.get("split_sizes", {})
    exp_lines.append(
        f"- **Partition Sizes:** Train (pre-SMOTE) = `{split_sizes.get('train_raw', 0):,}`, "
        f"Train (post-SMOTE) = `{split_sizes.get('train_smote', 0):,}`, "
        f"Validation = `{split_sizes.get('validation', 0):,}`, Test = `{split_sizes.get('test', 0):,}`.\n\n"
    )

    imb_exp = diag.get("imbalance_experiment", {})
    if imb_exp:
        exp_lines.append("## 2. Class Imbalance Experiment (SMOTE vs. Class Weighting)\n")
        exp_lines.append(f"- **Selected Strategy:** `{imb_exp.get('selected_strategy')}`\n")
        exp_lines.append(f"- **Justification:** {imb_exp.get('justification')}\n\n")

    cal_rep = diag.get("calibration_report", {})
    if cal_rep:
        exp_lines.append("## 3. Probability Calibration (Validation Temperature Scaling)\n")
        cal_df = pd.DataFrame(cal_rep).T.reset_index(names="Model")
        exp_lines.append(_df_to_markdown(cal_df) + "\n\n")

    if ae:
        exp_lines.append("## 4. Autoencoder Zero-Day Novelty Detection & Percentile Sweep\n")
        exp_lines.append(
            f"- **95th Percentile Benign Validation Threshold:** `{ae.get('threshold', 0):.6f}`\n"
            f"- **Binary Test Metrics:** Accuracy=`{ae.get('accuracy', 0):.4f}`, Precision=`{ae.get('precision', 0):.4f}`, "
            f"Recall=`{ae.get('recall', 0):.4f}`, F1=`{ae.get('f1', 0):.4f}`, ROC-AUC=`{ae.get('roc_auc', 0):.4f}`, "
            f"FPR=`{ae.get('fpr', 0):.4f}`, FNR=`{ae.get('fnr', 0):.4f}`\n\n"
        )
        if ae.get("percentile_sweep"):
            sweep_df = pd.DataFrame(ae["percentile_sweep"])[
                ["percentile", "threshold", "precision", "recall", "f1", "fpr", "fnr"]
            ]
            exp_lines.append(_df_to_markdown(sweep_df) + "\n\n")

    if unseen.get("executed"):
        exp_lines.append("## 5. Unseen-Attack Holdout Protocol\n")
        exp_lines.append(
            f"- **Held-Out Attack Class:** `{unseen.get('holdout_attack_class')}` (`{unseen.get('n_unseen_test_samples')}` test flows)\n"
            f"- **Closed-Set Supervised Unseen-Class Recall:** `0.0000`\n"
            f"- **Benign-Only Autoencoder Unseen-Attack Recall:** `{unseen.get('autoencoder_unseen_attack_recall', 0):.4f}` "
            f"(Benign vs Unseen ROC-AUC = `{unseen.get('autoencoder_benign_vs_unseen_metrics', {}).get('roc_auc', 0):.4f}`)\n\n"
        )

    if shap_sum:
        exp_lines.append("## 6. SHAP Explainability Summary\n")
        if "XGBoost" in shap_sum:
            top_x = ", ".join(
                f"`{f['feature']}` ({f['mean_abs_shap']:.4f})"
                for f in shap_sum["XGBoost"].get("top_features", [])[:5]
            )
            exp_lines.append(
                f"- **XGBoost Top-5 (`TreeExplainer`, {shap_sum['XGBoost'].get('elapsed_sec', 0):.2f}s):** {top_x}\n"
            )
        if "MLP" in shap_sum:
            top_m = ", ".join(
                f"`{f['feature']}` ({f['mean_abs_shap']:.4f})"
                for f in shap_sum["MLP"].get("top_features", [])[:5]
            )
            bench = shap_sum["MLP"].get("explainer_benchmark", {})
            exp_lines.append(
                f"- **MLP Top-5 (`{shap_sum['MLP'].get('explainer_type')}`, {shap_sum['MLP'].get('elapsed_sec', 0):.2f}s):** {top_m}\n"
            )
            if bench:
                exp_lines.append(
                    f"- **MLP Explainer Trade-Off:** `GradientExplainer` speedup vs `KernelExplainer` = "
                    f"`{bench.get('speedup_vs_kernel', 'N/A')}x`, Spearman rank correlation = `{bench.get('spearman_rank_agreement', 'N/A')}`.\n"
                )

    exp_path = reports_dir / "experimental_results.md"
    exp_path.write_text("\n".join(exp_lines), encoding="utf-8")
    return cmp_path, exp_path
