"""Tab 3: Model Performance — 4-Model Benchmark, Curves, Confusion Matrices, & Calibration."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.cards import apply_soc_plotly_theme
from dashboard.data_access import (
    get_figure_path,
    load_evaluation_summary,
    load_model_comparison_df,
    load_model_metadata,
    load_training_histories,
)
from src.evaluation.report_writer import (
    extract_reliable_metadata,
    format_reliable_f1_footnote,
    normalize_fusion_display_df,
)


def _display_df_with_na(df: pd.DataFrame) -> pd.DataFrame:
    """Format NaN/None cells safely as 'N/A' strings so PyArrow renders without type conflicts."""
    out = df.copy()
    for col in out.columns:
        if out[col].isna().any():
            if pd.api.types.is_numeric_dtype(out[col]):
                digits = (
                    1
                    if col == "Samples_Per_Sec"
                    else (2 if "Latency" in str(col) or "Duration" in str(col) else 4)
                )
                out[col] = out[col].map(lambda v, d=digits: "N/A" if pd.isna(v) else f"{float(v):.{d}f}")
            else:
                out[col] = out[col].fillna("N/A").astype(str)
    return out


def render_model_performance_page() -> None:
    """Render interactive model comparison metrics, training curves, matrices, and benchmarks."""
    st.subheader("Model Performance & Benchmarking Analytics")

    df_cmp = load_model_comparison_df()
    eval_summary = load_evaluation_summary()
    histories = load_training_histories()

    if df_cmp.empty and not eval_summary:
        st.warning(
            "No evaluation reports found yet. Run `python scripts/train.py --run-eval` to generate real benchmarks."
        )
        return

    src_meta = eval_summary.get("dataset_source", {})
    if src_meta.get("is_synthetic"):
        st.info(
            "ℹ️ Reported benchmarks reflect **Synthetic Demonstration Mode** (explicitly separated from genuine CICIDS2017 runs)."
        )

    st.markdown("#### 🏆 Unified Model Comparison Table")
    st.caption(
        "Note: MLP, 1D-CNN, and XGBoost are evaluated under multi-class supervised framing (weighted & macro metrics). "
        "The Autoencoder is evaluated separately under binary novelty framing (Benign vs Anomalous)."
    )
    if not df_cmp.empty:
        df_cmp_norm, is_deg_fusion, deg_fusion_note = normalize_fusion_display_df(
            df_cmp, eval_summary, name_col="Model"
        )
        st.dataframe(_display_df_with_na(df_cmp_norm), use_container_width=True, hide_index=True)
        if "F1_Macro_Reliable" in df_cmp_norm.columns:
            sup_evals = eval_summary.get("supervised_evaluations", {})
            sup_ref = sup_evals.get("XGBoost") or next(iter(sup_evals.values()), {})
            rel_t, rel_excl = extract_reliable_metadata(sup_ref)
            st.caption(format_reliable_f1_footnote(rel_t, rel_excl).lstrip("> "))
        if is_deg_fusion and deg_fusion_note:
            st.caption(deg_fusion_note)

        c1, c2 = st.columns(2)
        with c1:
            melted = df_cmp_norm.melt(
                id_vars=["Model"],
                value_vars=[
                    col
                    for col in ["Accuracy", "F1_Weighted", "F1_Macro", "ROC_AUC"]
                    if col in df_cmp_norm.columns
                ],
                var_name="Metric",
                value_name="Score",
            )
            fig_m = px.bar(
                melted,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                title="Predictive Performance Comparison",
                range_y=[0, 1.05],
            )
            st.plotly_chart(apply_soc_plotly_theme(fig_m, height=380), use_container_width=True)

        with c2:
            if "Latency_Median_us" in df_cmp_norm.columns or "Latency_Mean_us" in df_cmp_norm.columns:
                lat_cols = [
                    c for c in ("Latency_Median_us", "Latency_P95_us") if c in df_cmp_norm.columns
                ] or ["Latency_Mean_us"]
                df_lat = df_cmp_norm.dropna(subset=lat_cols)
                fig_l = px.bar(
                    df_lat,
                    x="Model",
                    y=lat_cols,
                    barmode="group",
                    title="Single-Flow Inference Latency: Median & P95 (µs, lower is faster)",
                )
                st.plotly_chart(apply_soc_plotly_theme(fig_l, height=380), use_container_width=True)

    # XGBoost Deployment Profiles & Time-Aware Selection (reads generated metadata/reports only)
    xgb_meta = load_model_metadata().get("models", {}).get("XGBoost", {})
    tb_deploy = eval_summary.get("time_block_protocol", {})
    tb_sel = eval_summary.get("time_block_profile_selection") or tb_deploy.get("profile_selection") or {}
    if xgb_meta or tb_sel:
        st.markdown("#### ⚖️ XGBoost Deployment Profiles & Time-Aware Selection")
        xgb_profiles_meta = xgb_meta.get("profiles", {}) if isinstance(xgb_meta, dict) else {}
        active_profile = (
            xgb_meta.get("active_profile") or tb_deploy.get("xgb_imbalance_strategy") or "class_weight"
        )
        alt_profile = "none" if active_profile == "class_weight" else "class_weight"
        active_artifact = (
            xgb_profiles_meta.get(active_profile, {}).get("artifact_file")
            or xgb_meta.get("profile_files", {}).get(active_profile)
            or "xgb_model.json"
        )
        alt_artifact = (
            xgb_profiles_meta.get(alt_profile, {}).get("artifact_file")
            or xgb_meta.get("profile_files", {}).get(alt_profile)
            or f"xgb_model_{alt_profile}.json"
        )
        st.caption(
            f"Active deployment profile: `{active_profile}` (`models/{active_artifact}`); "
            f"alternate: `{alt_profile}` (`models/{alt_artifact}`). "
            "Profile choice is made on validation data only — never on held-out test flows."
        )

        sel_map = tb_sel.get("metrics_by_profile", {}) if isinstance(tb_sel, dict) else {}
        if sel_map:
            st.markdown("**Chronological `time_block` Validation Selection (`used_test_data=False`)**")
            sel_rows = []
            for prof in ("class_weight", "none"):
                m = sel_map.get(prof)
                if not m:
                    continue
                sel_rows.append(
                    {
                        "Profile": prof,
                        "Val_Macro_F1": round(float(m.get("val_macro_f1", 0.0)), 4),
                        "Val_Weighted_F1": round(float(m.get("val_weighted_f1", 0.0)), 4),
                        "Val_Benign_FPR": round(float(m.get("val_benign_fpr", 0.0)), 4),
                        "Selected": "YES" if prof == tb_sel.get("selected_profile") else "",
                    }
                )
            if sel_rows:
                st.dataframe(pd.DataFrame(sel_rows), use_container_width=True, hide_index=True)
            if tb_sel.get("justification"):
                st.caption(f"Selection justification: {tb_sel['justification']}")

        rand_ctx = tb_sel.get("random_split_context", {}) if isinstance(tb_sel, dict) else {}
        if rand_ctx:
            st.markdown("**Random-Split Validation Context (Alternate-Profile Rationale)**")
            rand_rows = []
            for prof in ("class_weight", "none"):
                m = rand_ctx.get(prof)
                if not isinstance(m, dict):
                    continue
                rand_rows.append(
                    {
                        "Profile": prof,
                        "Val_Macro_F1": round(float(m.get("val_macro_f1", 0.0)), 4),
                        "Val_Weighted_F1": round(float(m.get("val_weighted_f1", 0.0)), 4),
                        "Val_Benign_FPR": round(float(m.get("val_benign_fpr", 0.0)), 4),
                    }
                )
            if rand_rows:
                st.dataframe(pd.DataFrame(rand_rows), use_container_width=True, hide_index=True)

        mit_rows = tb_deploy.get("mitigation_experiments", {}).get("rows", [])
        if mit_rows:
            wanted = {
                f"Baseline XGBoost ('{active_profile}')": f"{active_profile} (active)",
                f"XGBoost ('{alt_profile}')": f"{alt_profile} (alternate)",
            }
            trade_rows = []
            for row in mit_rows:
                label = wanted.get(str(row.get("Option")))
                if not label:
                    continue
                trade_rows.append(
                    {
                        "Profile": label,
                        "Benign_FPR": row.get("Benign_FPR"),
                        "Overall_Attack_Recall": row.get("Overall_Attack_Recall"),
                        "Focus_Mean_Recall": row.get("Focus_Mean_Recall"),
                        "Binary_F1": row.get("Binary_F1"),
                        "Multiclass_Macro_F1": row.get("Multiclass_Macro_F1"),
                        "Focus_Class_Recalls": row.get("Focus_Class_Recalls"),
                    }
                )
            if trade_rows:
                st.markdown(
                    "**Held-Out `time_block` Test Trade-Off (Descriptive Only, Not Used for Selection)**"
                )
                st.dataframe(
                    _display_df_with_na(pd.DataFrame(trade_rows)),
                    use_container_width=True,
                    hide_index=True,
                )

        tb_pc = tb_deploy.get("per_class_metrics", {}).get("XGBoost", [])
        caveats = []
        for cname in ("Bot", "Heartbleed"):
            pc = next((r for r in tb_pc if r.get("class_name") == cname), None)
            if not pc:
                continue
            suffix = " (statistically unreliable)" if pc.get("statistically_unreliable") else ""
            caveats.append(
                f"`{cname}` recall={float(pc.get('recall', 0.0)):.3f} "
                f"on {int(pc.get('support', 0))} test flows{suffix}"
            )
        drift = tb_deploy.get("failure_analysis", {}).get("feature_drift_top_features", [])
        bot_ks = next(
            (r for r in drift if r.get("Class") == "Bot" and r.get("Feature") == "Flow Duration"), None
        )
        if bot_ks:
            caveats.append(f"`Bot` Flow Duration shift KS={float(bot_ks.get('KS_Statistic', 0.0)):.4f}")
        if caveats:
            st.caption("Honest caveats — " + "; ".join(caveats) + ".")

    # Interactive Training Curves
    if histories:
        st.markdown("#### 📉 Training & Validation Convergence Curves")
        t_cols = st.columns(3)
        for idx, m_name in enumerate(("MLP", "CNN", "Autoencoder")):
            hist = histories.get(m_name, {})
            with t_cols[idx]:
                if "loss" in hist and "val_loss" in hist:
                    epochs = list(range(1, len(hist["loss"]) + 1))
                    fig_h = go.Figure()
                    fig_h.add_trace(
                        go.Scatter(x=epochs, y=hist["loss"], name="Train Loss", mode="lines+markers")
                    )
                    fig_h.add_trace(
                        go.Scatter(
                            x=epochs,
                            y=hist["val_loss"],
                            name="Val Loss",
                            mode="lines+markers",
                            line=dict(dash="dash"),
                        )
                    )
                    fig_h.update_layout(title=f"{m_name} Loss Curve", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(apply_soc_plotly_theme(fig_h, height=300), use_container_width=True)

    # Methodological Experiments: Ensemble, SMOTE vs Class-Weight, Optuna, Calibration, Unseen Attack
    st.markdown("#### 🔬 Methodological Experiments & Ablations")
    e1, e2 = st.columns(2)
    with e1:
        ens_cmp = eval_summary.get("ensemble_comparison", {})
        if ens_cmp:
            st.markdown("**Ensemble Decision Strategy Comparison (Test Split)**")
            ens_df_ui = pd.DataFrame(ens_cmp).T.reset_index(names="Strategy")
            st.dataframe(ens_df_ui, use_container_width=True, hide_index=True)
            if "f1_macro_reliable" in ens_df_ui.columns or "F1_Macro_Reliable" in ens_df_ui.columns:
                sup_evals = eval_summary.get("supervised_evaluations", {})
                sup_ref = sup_evals.get("XGBoost") or next(iter(sup_evals.values()), {})
                ens_meta = eval_summary.get("ensemble_analysis", {})
                ens_t, ens_excl = extract_reliable_metadata(
                    ens_meta if "unreliable_class_names" in ens_meta else sup_ref
                )
                st.caption(format_reliable_f1_footnote(ens_t, ens_excl).lstrip("> "))
            ens_note = eval_summary.get("ensemble_analysis", {}).get("honest_verdict", "")
            if ens_note:
                st.caption(ens_note)

        train_diag = eval_summary.get("training_diagnostics", {})
        imb_exp = train_diag.get("imbalance_experiment", {})
        if imb_exp:
            st.markdown("**Class Imbalance Experiment (Validation Split)**")
            per_model_imb = imb_exp.get("per_model", {})
            if per_model_imb:
                pm_rows = []
                for m_key in ("MLP", "CNN", "XGBoost"):
                    m_entry = per_model_imb.get(m_key)
                    if not m_entry:
                        continue
                    m_disp = "1D-CNN" if m_key == "CNN" else m_key
                    sel_m = m_entry.get("selected_strategy", "")
                    for strat_key, strat_label in (
                        ("none", "Unweighted"),
                        ("class_weight", "Class Weighting"),
                        ("smote", "SMOTE"),
                    ):
                        m_vals = m_entry.get("metrics_by_strategy", {}).get(strat_key)
                        if m_vals:
                            pm_rows.append(
                                {
                                    "Model": m_disp,
                                    "Approach": strat_label,
                                    "Val Macro-F1": round(float(m_vals.get("val_macro_f1", 0.0)), 4),
                                    "Val Weighted-F1": round(float(m_vals.get("val_weighted_f1", 0.0)), 4),
                                    "Val Macro-Recall": round(float(m_vals.get("val_macro_recall", 0.0)), 4),
                                    "Val LogLoss": round(float(m_vals.get("val_mlogloss", 0.0)), 4),
                                    "Selected": "✓" if strat_key == sel_m else "",
                                }
                            )
                if pm_rows:
                    st.dataframe(pd.DataFrame(pm_rows), use_container_width=True, hide_index=True)
                for m_key in ("MLP", "CNN", "XGBoost"):
                    m_entry = per_model_imb.get(m_key)
                    if m_entry:
                        m_disp = "1D-CNN" if m_key == "CNN" else m_key
                        st.caption(
                            f"**{m_disp}** (`{m_entry.get('selected_strategy')}`): {m_entry.get('justification', '')}"
                        )
            else:
                imb_rows = []
                for label, key in [
                    ("Unweighted", "baseline_unweighted_metrics"),
                    ("Class Weighting", "class_weight_metrics"),
                    ("SMOTE", "smote_metrics"),
                ]:
                    if imb_exp.get(key):
                        imb_rows.append({"Approach": label, **imb_exp[key]})
                if imb_rows:
                    st.dataframe(pd.DataFrame(imb_rows), use_container_width=True, hide_index=True)
                st.caption(imb_exp.get("justification", ""))

    with e2:
        xgb_opt = eval_summary.get("training_diagnostics", {}).get("xgboost_optimization", {})
        if xgb_opt:
            st.markdown("**XGBoost: Report Baseline vs Optuna Optimization**")
            opt_rows = [
                {"Variant": "Baseline (300 trees, depth 6)", **xgb_opt.get("baseline_val_metrics", {})},
                {
                    "Variant": f"Optuna Tuned ({xgb_opt.get('n_trials_completed', 0)} trials)",
                    **xgb_opt.get("tuned_val_metrics", {}),
                },
            ]
            st.dataframe(pd.DataFrame(opt_rows), use_container_width=True, hide_index=True)
            st.caption(
                f"Selected variant: `{xgb_opt.get('selected_variant', 'baseline')}`"
                + (f" — {xgb_opt['selection_note']}" if xgb_opt.get("selection_note") else "")
            )

        cal_rep = eval_summary.get("training_diagnostics", {}).get("calibration_report", {})
        if cal_rep:
            st.markdown("**Probability Calibration (Temperature Scaling on Validation)**")
            st.dataframe(
                pd.DataFrame(cal_rep).T.reset_index(names="Model"), use_container_width=True, hide_index=True
            )

    # Unseen-attack novelty protocol & Unsupervised anomaly baselines comparison
    ae_eval = eval_summary.get("autoencoder_evaluation", {})
    anom_comp = eval_summary.get("anomaly_detector_comparison", {})
    unseen = eval_summary.get("unseen_attack_protocol", {})
    if ae_eval or anom_comp or unseen.get("executed"):
        st.markdown("#### 🧬 Zero-Day Autoencoder & Classical Anomaly Baselines")
        if anom_comp.get("detectors"):
            st.markdown("**Benign-Only Unsupervised Anomaly Detector Comparison**")
            anom_df_norm, anom_deg, anom_deg_note = normalize_fusion_display_df(
                pd.DataFrame(anom_comp["detectors"]), eval_summary, name_col="Detector"
            )
            st.dataframe(_display_df_with_na(anom_df_norm), use_container_width=True, hide_index=True)
            if anom_deg and anom_deg_note:
                st.caption(anom_deg_note)
            if anom_comp.get("verdict"):
                st.caption(anom_comp["verdict"])
        ae1, ae2 = st.columns(2)
        with ae1:
            if ae_eval.get("percentile_sweep"):
                st.markdown("**Threshold Percentile Trade-Off Sweep**")
                st.dataframe(
                    pd.DataFrame(ae_eval["percentile_sweep"])[
                        ["percentile", "threshold", "precision", "recall", "f1", "fpr", "fnr"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
        with ae2:
            if unseen.get("executed"):
                st.markdown(f"**Unseen-Attack Holdout Experiment (`{unseen.get('holdout_attack_class')}`)**")
                st.json(unseen)

    # Open-Set Cross-Day (by_day) & Chronological Drift (time_block) Protocols
    time_blk = eval_summary.get("time_block_protocol", {})
    by_day = eval_summary.get("by_day_protocol", {})
    if time_blk.get("executed") or by_day.get("executed"):
        st.markdown("#### 🕒 Chronological Drift (`time_block`) & Open-Set Cross-Day (`by_day`) Protocols")
        if time_blk.get("executed"):
            st.markdown(
                "**Chronological Drift Evaluation (`time_block`: Per-Class Earliest ~60% / Next ~20% / Last ~20%)**"
            )
            tb_rows = []
            for m_name, m_vals in time_blk.get("models", {}).items():
                ci = m_vals.get("f1_macro_ci95", {})
                tb_rows.append(
                    {
                        "Model": m_name,
                        "Accuracy": m_vals.get("accuracy", 0.0),
                        "F1_Weighted": m_vals.get("f1_weighted", 0.0),
                        "F1_Macro": m_vals.get("f1_macro", 0.0),
                        "F1_Macro_Reliable": m_vals.get("f1_macro_reliable", 0.0),
                        "CI95_Low": ci.get("ci95_low", 0.0),
                        "CI95_High": ci.get("ci95_high", 0.0),
                        "ROC_AUC_OVR": m_vals.get("roc_auc_macro_ovr", 0.0),
                    }
                )
            if tb_rows:
                tb_ui_df = pd.DataFrame(tb_rows)
                st.dataframe(tb_ui_df, use_container_width=True, hide_index=True)
                tb_ref = (time_blk.get("models") or {}).get("XGBoost") or next(
                    iter((time_blk.get("models") or {}).values()), {}
                )
                tb_t, tb_excl = extract_reliable_metadata(
                    tb_ref, per_class_rows=(time_blk.get("per_class_metrics") or {}).get("XGBoost")
                )
                st.caption(format_reliable_f1_footnote(tb_t, tb_excl).lstrip("> "))
            st.caption(
                f"Ordering: `{time_blk.get('ordering_source', 'Timestamp')}` | "
                f"Leakage Check Passed: `{time_blk.get('leakage_checks', {}).get('passed')}`."
            )
            tb_pc_xgb = time_blk.get("per_class_metrics", {}).get("XGBoost", [])
            if tb_pc_xgb:
                with st.expander("View `time_block` XGBoost Per-Class Breakdown", expanded=False):
                    st.dataframe(pd.DataFrame(tb_pc_xgb), use_container_width=True, hide_index=True)

        if by_day.get("executed"):
            st.markdown(
                "**Open-Set Cross-Day Protocol (`by_day`: Mon–Wed Train / Thu Val / Thu–Fri Unseen Attacks)**"
            )
            if by_day.get("caveat_note"):
                st.info(by_day["caveat_note"])
            bin_rows = [
                {"Model": name, **vals} for name, vals in by_day.get("binary_cross_day_models", {}).items()
            ]
            if bin_rows:
                st.markdown("*Open-Set Cross-Day Binary Detection (Benign vs. Unseen Attacks)*")
                st.dataframe(pd.DataFrame(bin_rows), use_container_width=True, hide_index=True)
            mc_rows = [{"Model": name, **vals} for name, vals in by_day.get("multiclass_models", {}).items()]
            if mc_rows:
                with st.expander("View Reference Closed-Set Multi-Class Scores (`by_day`)", expanded=False):
                    mc_ui_df = pd.DataFrame(mc_rows)
                    st.dataframe(mc_ui_df, use_container_width=True, hide_index=True)
                    if "f1_macro_reliable" in mc_ui_df.columns or "F1_Macro_Reliable" in mc_ui_df.columns:
                        bd_ref = (by_day.get("multiclass_models") or {}).get("XGBoost") or next(
                            iter((by_day.get("multiclass_models") or {}).values()), {}
                        )
                        bd_t, bd_excl = extract_reliable_metadata(bd_ref)
                        st.caption(format_reliable_f1_footnote(bd_t, bd_excl).lstrip("> "))

    # Diagnostic Figures
    st.markdown("#### 🖼️ Confusion Matrices, ROC & Precision-Recall Curves")
    fig_cols = st.columns(2)
    cm_img = get_figure_path("confusion_matrices.png")
    roc_img = get_figure_path("roc_curves.png")
    pr_img = get_figure_path("pr_curves.png")
    ae_img = get_figure_path("autoencoder_diagnostics.png")
    with fig_cols[0]:
        if cm_img:
            st.image(
                str(cm_img), caption="Normalized Confusion Matrices (4 Models)", use_container_width=True
            )
        if pr_img:
            st.image(str(pr_img), caption="Precision-Recall Curves", use_container_width=True)
    with fig_cols[1]:
        if roc_img:
            st.image(
                str(roc_img), caption="One-vs-Rest & Binary Anomaly ROC Curves", use_container_width=True
            )
        if ae_img:
            st.image(
                str(ae_img),
                caption="Autoencoder MSE Separation & Threshold Trade-off",
                use_container_width=True,
            )
