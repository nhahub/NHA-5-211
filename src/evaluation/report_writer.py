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
    by_day = summary.get("by_day_protocol", {})
    ablation = summary.get("feature_ablation", {})
    xgb_cv = summary.get("xgboost_cross_validation", {})
    sup_evals = summary.get("supervised_evaluations", {})
    prep_lat = summary.get("preprocessing_latency", {})
    shap_sum = summary.get("shap_summary", {})
    split_strat = summary.get("split_strategy") or audit.get("split_strategy", "random")

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
    sec_num = 1
    comp_lines = [
        "# SMART IDS v3 — Model Comparison Report\n",
        mode_banner,
        f"## {sec_num}. Unified Four-Model Benchmark (Primary Split: `{split_strat}`)\n",
        "- **Supervised models (MLP, 1D-CNN, XGBoost):** Evaluated on multi-class attack attribution using weighted and macro metrics.\n",
        "- **Unsupervised Autoencoder:** Evaluated separately under binary novelty framing (Benign vs. Anomalous).\n\n",
        cmp_md + "\n\n",
    ]
    ci_lines: list[str] = []
    for key in ("XGBoost", "CNN", "MLP"):
        ev = sup_evals.get(key, {})
        ci = ev.get("f1_macro_ci95", {})
        if ci.get("n_bootstrap", 0) > 0:
            ci_lines.append(
                f"- **{ev.get('model', key)} Macro-F1 (95% Bootstrap CI, n={ci['n_bootstrap']}):** "
                f"`{ev.get('f1_macro', 0.0):.4f}` (`[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]`, "
                f"bootstrap std=`{ci['std']:.4f}`)\n"
            )
    ae_ci = ae.get("f1_ci95", {})
    if ae and ae_ci.get("n_bootstrap", 0) > 0:
        ci_lines.append(
            f"- **Autoencoder Binary F1 (95% Bootstrap CI, n={ae_ci['n_bootstrap']}):** "
            f"`{ae.get('f1', 0.0):.4f}` (`[{ae_ci['ci95_low']:.4f}, {ae_ci['ci95_high']:.4f}]`, "
            f"bootstrap std=`{ae_ci['std']:.4f}`)\n"
        )
    if ci_lines:
        comp_lines.extend([*ci_lines, "\n"])

    if by_day.get("executed"):
        sec_num += 1
        bd_sizes = by_day.get("split_sizes", {})
        unseen_cls = ", ".join(f"`{c}`" for c in by_day.get("unseen_test_classes", [])) or "None"
        comp_lines.append(
            f"## {sec_num}. Cross-Day Temporal Split Protocol (`by_day`: Mon–Wed Train / Thu Val / Thu–Fri Test)\n"
        )
        comp_lines.append(
            f"- **Partitions (`split_strategy='by_day'`):** Train pre-SMOTE=`{bd_sizes.get('train_raw', 0):,}`, "
            f"Validation (Thursday half)=`{bd_sizes.get('validation', 0):,}`, "
            f"Test (Thursday remainder + Friday)=`{bd_sizes.get('test', 0):,}`.\n"
        )
        comp_lines.append(f"- **Thursday–Friday Unseen Attack Classes:** {unseen_cls}.\n")
        if by_day.get("caveat_note"):
            comp_lines.append(f"- **Protocol Interpretation:** {by_day['caveat_note']}\n")
        comp_lines.append("\n### Closed-Set Multi-Class Evaluation (`by_day`)\n\n")
        mc_rows = [
            {
                "Model": name,
                "Accuracy": vals.get("accuracy", 0.0),
                "Precision_Weighted": vals.get("precision_weighted", 0.0),
                "Recall_Weighted": vals.get("recall_weighted", 0.0),
                "F1_Weighted": vals.get("f1_weighted", 0.0),
                "F1_Macro_All": vals.get("f1_macro", 0.0),
                "F1_Macro_Present_Test": vals.get("f1_macro_present_test_classes", 0.0),
                "ROC_AUC_OVR": vals.get("roc_auc_macro_ovr", 0.0),
            }
            for name, vals in by_day.get("multiclass_models", {}).items()
        ]
        comp_lines.append(_df_to_markdown(pd.DataFrame(mc_rows)) + "\n\n")
        comp_lines.append("### Cross-Day Binary Intrusion Detection (`by_day`: Benign vs. Attack)\n\n")
        bin_rows = [
            {
                "Model": name,
                "Accuracy": vals.get("accuracy", 0.0),
                "Precision": vals.get("precision", 0.0),
                "Recall": vals.get("recall", 0.0),
                "F1": vals.get("f1", 0.0),
                "ROC_AUC": vals.get("roc_auc", 0.0),
                "PR_AUC": vals.get("pr_auc", 0.0),
                "FPR": vals.get("fpr", 0.0),
                "FNR": vals.get("fnr", 0.0),
            }
            for name, vals in by_day.get("binary_cross_day_models", {}).items()
        ]
        comp_lines.append(_df_to_markdown(pd.DataFrame(bin_rows)) + "\n\n")

    if ablation.get("executed"):
        sec_num += 1
        w = ablation.get("with_features", {})
        wo = ablation.get("without_features", {})
        dropped_str = ", ".join(ablation.get("dropped_features", ["Destination Port"]))
        comp_lines.append(f"## {sec_num}. Shortcut Feature Ablation (With vs. Without `{dropped_str}`)\n")
        abl_df = pd.DataFrame(
            [
                {
                    "Variant": f"XGBoost (With {dropped_str})",
                    "Features": w.get("n_features", 0),
                    "Test_Rows": w.get("split_sizes", {}).get("test", 0),
                    "Val_Macro_F1": w.get("val", {}).get("f1_macro", 0.0),
                    "Test_Accuracy": w.get("test", {}).get("accuracy", 0.0),
                    "Test_F1_Weighted": w.get("test", {}).get("f1_weighted", 0.0),
                    "Test_F1_Macro": w.get("test", {}).get("f1_macro", 0.0),
                    "Test_ROC_AUC": w.get("test", {}).get("roc_auc_macro_ovr", 0.0),
                },
                {
                    "Variant": f"XGBoost (Without {dropped_str})",
                    "Features": wo.get("n_features", 0),
                    "Test_Rows": wo.get("split_sizes", {}).get("test", 0),
                    "Val_Macro_F1": wo.get("val", {}).get("f1_macro", 0.0),
                    "Test_Accuracy": wo.get("test", {}).get("accuracy", 0.0),
                    "Test_F1_Weighted": wo.get("test", {}).get("f1_weighted", 0.0),
                    "Test_F1_Macro": wo.get("test", {}).get("f1_macro", 0.0),
                    "Test_ROC_AUC": wo.get("test", {}).get("roc_auc_macro_ovr", 0.0),
                },
            ]
        )
        comp_lines.append(_df_to_markdown(abl_df) + "\n\n")
        comp_lines.append(
            f"- **Delta Test Macro-F1 (without - with):** `{ablation.get('delta_test_f1_macro', 0.0):+.4f}` "
            f"| **Delta Val Macro-F1:** `{ablation.get('delta_val_f1_macro', 0.0):+.4f}` "
            f"| **Delta Test Weighted-F1:** `{ablation.get('delta_test_f1_weighted', 0.0):+.4f}`.\n"
        )
        wo_top = ", ".join(
            f"`{item['feature']}` ({item['mean_abs_shap']:.4f})"
            for item in wo.get("top_shap_features", [])[:5]
        )
        if wo_top:
            comp_lines.append(
                f"- **Top-5 XGBoost SHAP Features After Removing `{dropped_str}`:** {wo_top}.\n"
            )
        comp_lines.append("\n")

    if any(sup_evals.get(k, {}).get("per_class_metrics") for k in ("XGBoost", "CNN", "MLP")):
        sec_num += 1
        comp_lines.append(f"## {sec_num}. Per-Class Precision, Recall, F1 & Support (Test Partition)\n")
        comp_lines.append(
            "> **Statistical Reliability Note:** Classes marked `Statistically Unreliable` have "
            "low held-out test support (`< 30` flows); one or two misclassified flows swing per-class "
            "F1 sharply and drive Macro-F1 variance.\n\n"
        )
        for key in ("XGBoost", "CNN", "MLP"):
            ev = sup_evals.get(key, {})
            per_cls = ev.get("per_class_metrics")
            if not per_cls:
                continue
            comp_lines.append(f"### {ev.get('model', key)} Per-Class Metrics\n\n")
            pc_rows = [
                {
                    "Class": row["class_name"],
                    "Precision": float(row["precision"]),
                    "Recall": float(row["recall"]),
                    "F1_Score": float(row["f1_score"]),
                    "Support": int(row["support"]),
                    "Reliability": "Statistically Unreliable"
                    if row.get("statistically_unreliable")
                    else "Adequate",
                }
                for row in per_cls
            ]
            comp_lines.append(_df_to_markdown(pd.DataFrame(pc_rows)) + "\n\n")

    if xgb_cv.get("executed"):
        sec_num += 1
        comp_lines.append(
            f"## {sec_num}. XGBoost {xgb_cv.get('n_splits', 5)}-Fold Stratified Cross-Validation\n"
        )
        comp_lines.append(
            f"- **Development Pool Rows (Held-Out Test Excluded):** `{xgb_cv.get('dev_pool_rows', 0):,}` "
            f"| **Imbalance Strategy:** `{xgb_cv.get('imbalance_strategy')}`\n"
            f"- **Macro-F1 (Mean +/- Std):** `{xgb_cv.get('mean_macro_f1', 0.0):.4f} ± {xgb_cv.get('std_macro_f1', 0.0):.4f}`\n"
            f"- **Weighted-F1 (Mean +/- Std):** `{xgb_cv.get('mean_weighted_f1', 0.0):.4f} ± {xgb_cv.get('std_weighted_f1', 0.0):.4f}` "
            f"| **Accuracy:** `{xgb_cv.get('mean_accuracy', 0.0):.4f} ± {xgb_cv.get('std_accuracy', 0.0):.4f}`\n\n"
        )
        fold_rows = [
            {
                "Fold": int(f["fold"]),
                "Train_Rows": int(f["train_rows"]),
                "Val_Rows": int(f["val_rows"]),
                "Accuracy": float(f["accuracy"]),
                "F1_Weighted": float(f["f1_weighted"]),
                "F1_Macro": float(f["f1_macro"]),
                "ROC_AUC_OVR": float(f["roc_auc_macro_ovr"]),
            }
            for f in xgb_cv.get("folds", [])
        ]
        if fold_rows:
            comp_lines.append(_df_to_markdown(pd.DataFrame(fold_rows)) + "\n\n")

    sec_num += 1
    comp_lines.append(f"## {sec_num}. Ensemble Strategy Comparison (Test Partition)\n")
    if ens:
        ens_df = pd.DataFrame(ens).T.reset_index(names="Strategy")
        comp_lines.append(_df_to_markdown(ens_df) + "\n\n")

    sec_num += 1
    comp_lines.append(f"## {sec_num}. XGBoost Baseline vs. Optuna Hyperparameter Optimization\n")
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

    sec_num += 1
    comp_lines.append(f"## {sec_num}. Architectural Notes on 1D-CNN vs. MLP & XGBoost\n")
    comp_lines.append(
        "Tabular CICFlowMeter features do not form a physical time series; 1D convolutions scan "
        "adjacent engineered statistics (packet sizes, IATs, flags). Measured validation and test "
        "F1/latency above quantify whether 1D-CNN yields measurable benefit over the dense MLP and XGBoost.\n\n"
    )

    sec_num += 1
    comp_lines.append(f"## {sec_num}. Preprocessing vs. Model Inference Latency\n")
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
        f"- **Split Strategy:** `{audit.get('split_strategy', split_strat)}`\n",
        f"- **Explicitly Dropped Features (`drop_features`):** `{audit.get('dropped_features', [])}`\n",
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
        exp_lines.append("## 2. Class Imbalance Experiment (Unweighted vs. Class Weighting vs. SMOTE)\n")
        imb_rows = [
            {"Strategy": label, **imb_exp[key]}
            for label, key in [
                ("Unweighted ('none')", "baseline_unweighted_metrics"),
                ("Class Weighting ('class_weight')", "class_weight_metrics"),
                ("Hybrid SMOTE ('smote')", "smote_metrics"),
            ]
            if imb_exp.get(key)
        ]
        if imb_rows:
            exp_lines.append(_df_to_markdown(pd.DataFrame(imb_rows)) + "\n\n")
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
