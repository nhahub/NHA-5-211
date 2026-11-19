"""Markdown academic report generator derived strictly from actual evaluation metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def format_ensemble_empirical_conclusion(ens_analysis: dict[str, Any]) -> str:
    """Generate the ensemble empirical conclusion directly from paired bootstrap test results."""
    best_single = str(ens_analysis.get("best_single_model", "XGBoost"))
    best_single_f1 = float(ens_analysis.get("best_single_f1_macro", 0.0))
    best_fusion = str(ens_analysis.get("best_ensemble_strategy", "stacking"))
    best_fusion_f1 = float(ens_analysis.get("best_ensemble_f1_macro", 0.0))
    delta_obs = float(ens_analysis.get("delta_vs_best_single", best_fusion_f1 - best_single_f1))

    paired_vs_best = ens_analysis.get("paired_bootstrap_vs_best_single", {}) or {}
    target_key = "stacking" if "stacking" in paired_vs_best else best_fusion
    test_res = paired_vs_best.get(target_key)

    if isinstance(test_res, dict) and test_res.get("n_bootstrap", 0) > 0:
        obs_d = float(test_res.get("observed_diff", delta_obs))
        mean_d = float(test_res.get("mean_diff", obs_d))
        ci_lo = float(test_res.get("ci95_low", 0.0))
        ci_hi = float(test_res.get("ci95_high", 0.0))
        sig = bool(test_res.get("statistically_significant", ci_lo > 0.0 or ci_hi < 0.0))
        n_boot = int(test_res.get("n_bootstrap", 1000))
        if sig and mean_d > 0.0:
            return (
                f"Ensemble fusion '{target_key}' achieved Test Macro-F1={best_fusion_f1:.4f} vs. "
                f"{best_single}={best_single_f1:.4f} (paired bootstrap n={n_boot}, mean diff={mean_d:+.4f}, "
                f"95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]; statistically significant, CI excludes 0)."
            )
        if sig and mean_d < 0.0:
            return (
                f"Ensemble fusion '{target_key}' (Test Macro-F1={best_fusion_f1:.4f}) scored below standalone "
                f"{best_single} ({best_single_f1:.4f}; paired bootstrap n={n_boot}, mean diff={mean_d:+.4f}, "
                f"95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]; statistically significant, CI excludes 0). "
                f"Standalone {best_single} is retained for supervised attribution with the Autoencoder orthogonal."
            )
        return (
            f"Ensemble fusion '{target_key}' (Test Macro-F1={best_fusion_f1:.4f}) is comparable to "
            f"{best_single}, difference not statistically significant ({best_single}={best_single_f1:.4f}; "
            f"observed delta={obs_d:+.4f}, paired bootstrap n={n_boot} mean diff={mean_d:+.4f}, "
            f"95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] includes 0; statistically_significant=False). "
            f"Standalone {best_single} is kept as the supervised default with the Autoencoder orthogonal."
        )

    return (
        f"Ensemble fusion '{best_fusion}' (Test Macro-F1={best_fusion_f1:.4f}) is comparable to "
        f"{best_single}, difference not statistically significant without a paired bootstrap test excluding 0 "
        f"({best_single}={best_single_f1:.4f}, delta={delta_obs:+.4f})."
    )


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Format a DataFrame as a GitHub-Flavored Markdown table without external dependencies."""
    if df.empty:
        return "_No rows available._"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for row in df.itertuples(index=False, name=None):
        cells = [
            "N/A"
            if (v is None or (isinstance(v, float) and pd.isna(v)))
            else (f"{v:.4f}" if isinstance(v, float) else str(v))
            for v in row
        ]
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
    time_block = summary.get("time_block_protocol", {})
    ablation = summary.get("feature_ablation", {})
    xgb_cv = summary.get("xgboost_cross_validation", {})
    sup_evals = summary.get("supervised_evaluations", {})
    ens_analysis = summary.get("ensemble_analysis", {})
    anom_evals = summary.get("anomaly_evaluations", {})
    anom_comp = summary.get("anomaly_detector_comparison", {})
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
        f"## {sec_num}. Unified Model Benchmark (Primary Split: `{split_strat}`)\n",
        "- **Supervised models (MLP, 1D-CNN, XGBoost):** Evaluated on multi-class attack attribution using weighted and macro metrics.\n",
        "- **Unsupervised anomaly detectors (Autoencoder, Isolation Forest, One-Class SVM, LOF):** Trained strictly on benign flows and evaluated under identical binary novelty framing (Benign vs. Anomalous).\n\n",
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
    det_sources = anom_evals or ({"Autoencoder": ae} if ae else {})
    for det_name, det_ev in det_sources.items():
        det_ci = det_ev.get("f1_ci95", {})
        if det_ci.get("n_bootstrap", 0) > 0:
            ci_lines.append(
                f"- **{det_name} Binary F1 (95% Bootstrap CI, n={det_ci['n_bootstrap']}):** "
                f"`{det_ev.get('f1', 0.0):.4f}` (`[{det_ci['ci95_low']:.4f}, {det_ci['ci95_high']:.4f}]`, "
                f"bootstrap std=`{det_ci['std']:.4f}`)\n"
            )
    if ci_lines:
        comp_lines.extend([*ci_lines, "\n"])

    if time_block.get("executed"):
        sec_num += 1
        tb_sizes = time_block.get("split_sizes", {})
        comp_lines.append(
            f"## {sec_num}. Chronological Drift Evaluation (`time_block`: Per-Class Earliest ~60% Train / Next ~20% Val / Last ~20% Test)\n"
        )
        comp_lines.append(
            f"- **Partitions (`split_strategy='time_block'`):** Train pre-SMOTE=`{tb_sizes.get('train_raw', 0):,}`, "
            f"Validation=`{tb_sizes.get('validation', 0):,}`, Test=`{tb_sizes.get('test', 0):,}` "
            f"(Ordering=`{time_block.get('ordering_source', 'Timestamp')}`, "
            f"`leakage_checks.passed={time_block.get('leakage_checks', {}).get('passed')}`).\n"
        )
        fallbacks = time_block.get("tiny_class_fallbacks", {})
        if fallbacks:
            fb_summary = "; ".join(f"`{k}`: {v}" for k, v in fallbacks.items())
            comp_lines.append(f"- **Low-Support / Tiny-Class Chronological Allocations:** {fb_summary}.\n")
        comp_lines.append(
            "- **Protocol Interpretation:** Closed-set temporal drift split where every class is trained on its "
            "earliest ~60% of flows and evaluated on its latest ~20%, isolating concept drift over capture time "
            "without introducing unseen attack labels.\n\n"
        )
        tb_rows = []
        for m_name, m_vals in time_block.get("models", {}).items():
            ci = m_vals.get("f1_macro_ci95", {})
            tb_rows.append(
                {
                    "Model": m_name,
                    "Accuracy": m_vals.get("accuracy", 0.0),
                    "Precision_Weighted": m_vals.get("precision_weighted", 0.0),
                    "Recall_Weighted": m_vals.get("recall_weighted", 0.0),
                    "F1_Weighted": m_vals.get("f1_weighted", 0.0),
                    "F1_Macro": m_vals.get("f1_macro", 0.0),
                    "F1_Macro_CI95_Low": ci.get("ci95_low", m_vals.get("f1_macro", 0.0)),
                    "F1_Macro_CI95_High": ci.get("ci95_high", m_vals.get("f1_macro", 0.0)),
                    "ROC_AUC_OVR": m_vals.get("roc_auc_macro_ovr", 0.0),
                    "PR_AUC_OVR": m_vals.get("pr_auc_macro_ovr", 0.0),
                }
            )
        if tb_rows:
            comp_lines.append(_df_to_markdown(pd.DataFrame(tb_rows)) + "\n\n")
        for m_name, m_vals in time_block.get("models", {}).items():
            ci = m_vals.get("f1_macro_ci95", {})
            if ci.get("n_bootstrap", 0) > 0:
                comp_lines.append(
                    f"- **{m_name} (`time_block`) Macro-F1 (95% Bootstrap CI, n={ci['n_bootstrap']}):** "
                    f"`{m_vals.get('f1_macro', 0.0):.4f}` (`[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]`, "
                    f"bootstrap std=`{ci['std']:.4f}`)\n"
                )
        tb_ae = time_block.get("autoencoder", {})
        if tb_ae:
            ae_ci = tb_ae.get("f1_ci95", {})
            ci_txt = (
                f" (`[{ae_ci['ci95_low']:.4f}, {ae_ci['ci95_high']:.4f}]`, n={ae_ci['n_bootstrap']})"
                if ae_ci.get("n_bootstrap", 0) > 0
                else ""
            )
            comp_lines.append(
                f"- **Autoencoder (`time_block`, benign-only binary):** Accuracy=`{tb_ae.get('accuracy', 0.0):.4f}`, "
                f"F1=`{tb_ae.get('f1', 0.0):.4f}`{ci_txt}, ROC-AUC=`{tb_ae.get('roc_auc', 0.0):.4f}`, "
                f"FPR=`{tb_ae.get('fpr', 0.0):.4f}`.\n"
            )
        comp_lines.append("\n")
        tb_pc_map = time_block.get("per_class_metrics", {})
        pc_target = "XGBoost" if "XGBoost" in tb_pc_map else next(iter(tb_pc_map), None)
        if pc_target and tb_pc_map.get(pc_target):
            comp_lines.append(f"### {pc_target} Per-Class Metrics (`time_block` Test Partition)\n\n")
            tb_pc_rows = [
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
                for row in tb_pc_map[pc_target]
            ]
            comp_lines.append(_df_to_markdown(pd.DataFrame(tb_pc_rows)) + "\n\n")

    if by_day.get("executed"):
        sec_num += 1
        bd_sizes = by_day.get("split_sizes", {})
        unseen_cls = ", ".join(f"`{c}`" for c in by_day.get("unseen_test_classes", [])) or "None"
        comp_lines.append(
            f"## {sec_num}. Open-Set Cross-Day Protocol (`by_day`: Mon–Wed Train / Thu Val / Thu–Fri Unseen Attacks)\n"
        )
        comp_lines.append(
            f"- **Partitions (`split_strategy='by_day'`):** Train pre-SMOTE=`{bd_sizes.get('train_raw', 0):,}`, "
            f"Validation (Thursday half)=`{bd_sizes.get('validation', 0):,}`, "
            f"Test (Thursday remainder + Friday)=`{bd_sizes.get('test', 0):,}` "
            f"(`leakage_checks.passed={by_day.get('leakage_checks', {}).get('passed')}`).\n"
        )
        comp_lines.append(f"- **Thursday–Friday Unseen Attack Classes:** {unseen_cls}.\n")
        if by_day.get("caveat_note"):
            comp_lines.append(f"\n> **Open-Set Interpretation:** {by_day['caveat_note']}\n\n")
        comp_lines.append("### Open-Set Cross-Day Binary Detection (`by_day`: Benign vs. Attack)\n\n")
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
        comp_lines.append("### Closed-Set Multi-Class Reference Under Open-Set Split (`by_day`)\n\n")
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
        if ablation.get("identical_test_rows_verified"):
            comp_lines.append(
                f"- **Apples-to-Apples Row Parity Verified:** Identical post-deduplication train/val/test rows "
                f"(`Test_Rows={w.get('split_sizes', {}).get('test', 0):,}`, "
                f"`sha256={str(w.get('test_rows_sha256', ''))[:12]}...`) evaluated after full 77-feature "
                "leakage verification.\n"
            )
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
            f"- **Variant Evaluated:** `{xgb_cv.get('variant', 'baseline')}` "
            f"| **Development Pool Rows (Held-Out Test Excluded):** `{xgb_cv.get('dev_pool_rows', 0):,}` "
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
    paired_best = ens_analysis.get("paired_bootstrap_vs_best_single", {}) or {}
    if paired_best:
        pb_rows = [
            {
                "Strategy_vs_BestSingle": strat,
                "Observed_Diff": float(vals.get("observed_diff", 0.0)),
                "Mean_Diff": float(vals.get("mean_diff", 0.0)),
                "CI95_Low": float(vals.get("ci95_low", 0.0)),
                "CI95_High": float(vals.get("ci95_high", 0.0)),
                "Statistically_Significant": bool(vals.get("statistically_significant", False)),
                "N_Bootstrap": int(vals.get("n_bootstrap", 0)),
            }
            for strat, vals in paired_best.items()
        ]
        comp_lines.append(
            f"### Paired Bootstrap Macro-F1 Significance vs. Best Single (`{ens_analysis.get('best_single_model', 'XGBoost')}`)\n\n"
        )
        comp_lines.append(_df_to_markdown(pd.DataFrame(pb_rows)) + "\n\n")
    paired_stack = ens_analysis.get("paired_bootstrap_stacking_vs_all", {}) or {}
    if paired_stack:
        ps_rows = [
            {
                "Comparison": comp_name,
                "Observed_Diff": float(vals.get("observed_diff", 0.0)),
                "Mean_Diff": float(vals.get("mean_diff", 0.0)),
                "CI95_Low": float(vals.get("ci95_low", 0.0)),
                "CI95_High": float(vals.get("ci95_high", 0.0)),
                "Statistically_Significant": bool(vals.get("statistically_significant", False)),
                "N_Bootstrap": int(vals.get("n_bootstrap", 0)),
            }
            for comp_name, vals in paired_stack.items()
        ]
        comp_lines.append("### Paired Bootstrap Macro-F1 Comparison: Stacking vs. Other Strategies\n\n")
        comp_lines.append(_df_to_markdown(pd.DataFrame(ps_rows)) + "\n\n")
    if ens_analysis:
        conclusion = format_ensemble_empirical_conclusion(ens_analysis)
        comp_lines.append(f"- **Empirical Conclusion:** {conclusion}\n\n")

    if anom_comp.get("detectors"):
        sec_num += 1
        comp_lines.append(f"## {sec_num}. Unsupervised Anomaly Detector Baselines (Benign-Only Training)\n")
        comp_lines.append(_df_to_markdown(pd.DataFrame(anom_comp["detectors"])) + "\n\n")
        if anom_comp.get("verdict"):
            comp_lines.append(f"- **Verdict:** {anom_comp['verdict']}\n\n")
        sweep_rows = [
            {
                "Detector": d_name,
                "Percentile": row["percentile"],
                "Threshold": row["threshold"],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1": row["f1"],
                "FPR": row["fpr"],
                "FNR": row["fnr"],
            }
            for d_name, d_ev in anom_evals.items()
            for row in d_ev.get("percentile_sweep", [])
        ]
        if sweep_rows:
            comp_lines.append("### Percentile Threshold Trade-Off Sweep Across Detectors\n\n")
            comp_lines.append(_df_to_markdown(pd.DataFrame(sweep_rows)) + "\n\n")

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
        comp_lines.append(
            f"\n- **Selected Variant:** `{xgb_opt.get('selected_variant', 'baseline')}` "
            f"(Baseline CV Macro-F1 = `{xgb_opt.get('cv_macro_f1_mean', 0.0):.4f} ± {xgb_opt.get('cv_macro_f1_std', 0.0):.4f}`, "
            f"Val Macro-F1 delta=`{xgb_opt.get('macro_f1_improvement', 0.0):+.4f}`, "
            f"Meaningful Improvement=`{xgb_opt.get('meaningful_improvement', False)}`)\n"
        )
        if xgb_opt.get("selection_note"):
            comp_lines.append(f"- **Decision Basis:** {xgb_opt['selection_note']}\n")
        cv_note = xgb_opt.get("cv_method_note") or (
            "Baseline CV std is computed via stratified K-fold CV across the development pool with "
            "per-fold preprocessing and the selected imbalance strategy (ddof=1 across fold Macro-F1 scores)."
        )
        comp_lines.append(f"- **How CV Std is Computed:** {cv_note}\n")
        comp_lines.append("\n")

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
        warmup_n = int(prep_lat.get("warmup_calls_discarded", 50))
        comp_lines.append(
            f"- **Isolated Single-Flow Preprocessing (Imputer + 5-Sigma Clipper + StandardScaler):** "
            f"median = `{prep_lat.get('median_us', 0):.2f} µs`, P95 = `{prep_lat.get('p95_us', 0):.2f} µs` "
            f"(mean = `{prep_lat.get('mean_us', 0):.2f} µs`, `{warmup_n}` warm-up calls discarded; "
            f"`{prep_lat.get('samples_per_sec', 0):.1f}` flows/sec).\n"
        )

    cmp_path = reports_dir / "model_comparison_report.md"
    cmp_path.write_text("\n".join(comp_lines), encoding="utf-8")

    # 2. Experimental Results Report
    eff_split = audit.get("split_strategy", split_strat)
    split_desc = {
        "random": "Stratified Random Split",
        "time_block": "Chronological Drift Split (earliest ~60% train / ~20% val / ~20% test)",
        "by_day": "Open-Set Cross-Day Protocol (Mon-Wed train / Thu val / Thu-Fri unseen attacks)",
    }.get(eff_split, eff_split)
    exp_lines = [
        "# SMART IDS v3 — Experimental Results Report\n",
        mode_banner,
        "## 1. Data Engineering & Leakage Safeguard Audit\n",
        f"- **Split Strategy:** `{eff_split}` ({split_desc})\n",
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
        per_model_imb = imb_exp.get("per_model", {})
        if per_model_imb:
            cap_note = imb_exp.get("neural_epochs_cap")
            cap_str = (
                f" (neural candidates capped at `{cap_note}` epochs during selection; full training runs only for each model's winner)"
                if cap_note
                else ""
            )
            exp_lines.append(f"Selection mode: `{imb_exp.get('selection_mode', 'per_model')}`{cap_str}.\n\n")
            matrix_rows = []
            strat_display = {
                "none": "Unweighted ('none')",
                "class_weight": "Class Weighting ('class_weight')",
                "smote": "Hybrid SMOTE ('smote')",
            }
            for m_key in ("MLP", "CNN", "XGBoost"):
                m_entry = per_model_imb.get(m_key)
                if not m_entry:
                    continue
                m_disp = "1D-CNN" if m_key == "CNN" else m_key
                sel_m = m_entry.get("selected_strategy", "")
                m_strats = m_entry.get("metrics_by_strategy", {})
                for strat_key in ("none", "class_weight", "smote"):
                    s_metrics = m_strats.get(strat_key)
                    if not s_metrics:
                        continue
                    matrix_rows.append(
                        {
                            "Model": m_disp,
                            "Strategy": strat_display.get(strat_key, strat_key),
                            "Val_Macro_F1": round(float(s_metrics.get("val_macro_f1", 0.0)), 4),
                            "Val_Weighted_F1": round(float(s_metrics.get("val_weighted_f1", 0.0)), 4),
                            "Val_Macro_Recall": round(float(s_metrics.get("val_macro_recall", 0.0)), 4),
                            "Val_MLogLoss": round(float(s_metrics.get("val_mlogloss", 0.0)), 4),
                            "Selected": "YES" if strat_key == sel_m else "",
                        }
                    )
            if matrix_rows:
                exp_lines.append(_df_to_markdown(pd.DataFrame(matrix_rows)) + "\n\n")
            for m_key in ("MLP", "CNN", "XGBoost"):
                m_entry = per_model_imb.get(m_key)
                if not m_entry:
                    continue
                m_disp = "1D-CNN" if m_key == "CNN" else m_key
                exp_lines.append(
                    f"- **{m_disp} (`{m_entry.get('selected_strategy')}`):** {m_entry.get('justification', '')}\n"
                )
            exp_lines.append(f"- **Overall Summary:** {imb_exp.get('justification', '')}\n\n")
        else:
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
        recal_models = [
            f"`{m}` (T={v.get('temperature', 1.0):.4f}, ECE {v.get('ece_before', 0):.6f} -> {v.get('ece_after', 0):.6f})"
            for m, v in cal_rep.items()
            if abs(float(v.get("temperature", 1.0)) - 1.0) > 0.12
        ]
        if recal_models:
            exp_lines.append(
                "- **Calibration Evidence:** Bounded float64 Brent optimization recalibrated overconfident probabilities on "
                + ", ".join(recal_models)
                + ".\n\n"
            )
        else:
            exp_lines.append(
                "- **Calibration Evidence:** Models were already well-calibrated near `T = 1.0` with minimal NLL/ECE change.\n\n"
            )

    if ae or anom_comp.get("detectors"):
        exp_lines.append("## 4. Unsupervised Anomaly Detection & Classical Baselines\n")
        if anom_comp.get("detectors"):
            exp_lines.append(_df_to_markdown(pd.DataFrame(anom_comp["detectors"])) + "\n\n")
            if anom_comp.get("verdict"):
                exp_lines.append(f"- **Assessment:** {anom_comp['verdict']}\n\n")
        if ae:
            exp_lines.append(
                f"- **Autoencoder 95th Percentile Benign Validation Threshold:** `{ae.get('threshold', 0):.6f}`\n"
                f"- **Autoencoder Binary Test Metrics:** Accuracy=`{ae.get('accuracy', 0):.4f}`, Precision=`{ae.get('precision', 0):.4f}`, "
                f"Recall=`{ae.get('recall', 0):.4f}`, F1=`{ae.get('f1', 0):.4f}`, ROC-AUC=`{ae.get('roc_auc', 0):.4f}`, "
                f"FPR=`{ae.get('fpr', 0):.4f}`, FNR=`{ae.get('fnr', 0):.4f}`\n\n"
            )
        sweep_src = (
            [
                {"Detector": d_name, **row}
                for d_name, d_ev in anom_evals.items()
                for row in d_ev.get("percentile_sweep", [])
            ]
            if anom_evals
            else [{"Detector": "Autoencoder", **row} for row in ae.get("percentile_sweep", [])]
        )
        if sweep_src:
            sweep_df = pd.DataFrame(sweep_src)[
                ["Detector", "percentile", "threshold", "precision", "recall", "f1", "fpr", "fnr"]
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
