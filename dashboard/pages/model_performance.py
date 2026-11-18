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
    load_training_histories,
)


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
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            melted = df_cmp.melt(
                id_vars=["Model"],
                value_vars=[
                    col for col in ["Accuracy", "F1_Weighted", "F1_Macro", "ROC_AUC"] if col in df_cmp.columns
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
            if "Latency_Mean_us" in df_cmp.columns:
                fig_l = px.bar(
                    df_cmp,
                    x="Model",
                    y=["Latency_Mean_us", "Latency_P95_us"]
                    if "Latency_P95_us" in df_cmp.columns
                    else ["Latency_Mean_us"],
                    barmode="group",
                    title="Single-Flow Inference Latency (µs, lower is faster)",
                )
                st.plotly_chart(apply_soc_plotly_theme(fig_l, height=380), use_container_width=True)

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
            st.dataframe(
                pd.DataFrame(ens_cmp).T.reset_index(names="Strategy"),
                use_container_width=True,
                hide_index=True,
            )
            ens_note = eval_summary.get("ensemble_analysis", {}).get("honest_verdict", "")
            if ens_note:
                st.caption(ens_note)

        train_diag = eval_summary.get("training_diagnostics", {})
        imb_exp = train_diag.get("imbalance_experiment", {})
        if imb_exp:
            st.markdown("**Class Imbalance Experiment (Validation Split)**")
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

    # Unseen-attack novelty protocol & Autoencoder percentile sweep
    ae_eval = eval_summary.get("autoencoder_evaluation", {})
    unseen = eval_summary.get("unseen_attack_protocol", {})
    if ae_eval or unseen.get("executed"):
        st.markdown("#### 🧬 Zero-Day Autoencoder Threshold & Unseen-Attack Protocol")
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
