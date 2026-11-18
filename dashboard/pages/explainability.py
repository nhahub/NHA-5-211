"""Tab 5: Explainability — Global & Local Live SHAP Feature Attributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.cards import apply_soc_plotly_theme
from dashboard.data_access import (
    get_figure_path,
    load_cached_artifacts,
    load_shap_summary,
    load_test_split_sample,
)
from src.explainability.shap_explainer import explain_single_sample


def render_explainability_page() -> None:
    """Render global SHAP rankings, explainer benchmark trade-offs, and interactive local SHAP."""
    st.subheader("Explainable AI (XAI) — SHAP Feature Attributions")

    shap_data = load_shap_summary()
    artifacts = load_cached_artifacts()

    if not shap_data and not artifacts.models:
        st.warning(
            "No SHAP explanations or trained models found. Run `git lfs pull` or `python scripts/train.py --run-eval` first."
        )
        return
    if not artifacts.models:
        st.info(
            "Precomputed SHAP plots/rankings are displayed below; live single-flow SHAP inspection requires "
            "fetching `models/` via `git lfs pull` or training with `python scripts/train.py --run-eval`."
        )

    if shap_data:
        g1, g2 = st.columns(2)
        with g1:
            xgb_info = shap_data.get("XGBoost", {})
            if xgb_info.get("top_features"):
                df_x = pd.DataFrame(xgb_info["top_features"][:15]).sort_values(
                    "mean_abs_shap", ascending=True
                )
                fig_x = px.bar(
                    df_x,
                    x="mean_abs_shap",
                    y="feature",
                    orientation="h",
                    title=f"XGBoost Global Top-15 ({xgb_info.get('explainer_type')}, n={xgb_info.get('n_samples_explained')}, {xgb_info.get('elapsed_sec', 0):.2f}s)",
                    color="mean_abs_shap",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(apply_soc_plotly_theme(fig_x, height=430), use_container_width=True)

        with g2:
            mlp_info = shap_data.get("MLP", {})
            if mlp_info.get("top_features"):
                df_m = pd.DataFrame(mlp_info["top_features"][:15]).sort_values(
                    "mean_abs_shap", ascending=True
                )
                fig_m = px.bar(
                    df_m,
                    x="mean_abs_shap",
                    y="feature",
                    orientation="h",
                    title=f"MLP Global Top-15 ({mlp_info.get('explainer_type')}, n={mlp_info.get('n_samples_explained')}, {mlp_info.get('elapsed_sec', 0):.2f}s)",
                    color="mean_abs_shap",
                    color_continuous_scale="YlOrRd",
                )
                st.plotly_chart(apply_soc_plotly_theme(fig_m, height=430), use_container_width=True)

        bench = shap_data.get("MLP", {}).get("explainer_benchmark", {})
        if bench:
            with st.expander(
                "⏱️ Deep Explainer Computational Trade-Off Benchmark (GradientExplainer vs KernelExplainer)"
            ):
                st.json(bench)

    # Static Report Figures (Beeswarm + Waterfall)
    f1, f2 = st.columns(2)
    xgb_fig = get_figure_path("shap_xgb_summary.png")
    wf_fig = get_figure_path("shap_waterfall.png")
    with f1:
        if xgb_fig:
            st.image(
                str(xgb_fig), caption="XGBoost Global Bar & Attack Beeswarm Summary", use_container_width=True
            )
    with f2:
        if wf_fig:
            st.image(
                str(wf_fig),
                caption="Local Waterfall Decomposition (Critical Detection Sample)",
                use_container_width=True,
            )

    # Live Local Explanation on Held-Out Test Flow
    st.markdown("---")
    st.markdown("#### 🔎 Live Single-Flow Local SHAP Inspector")
    X_test, y_test, _ = load_test_split_sample(max_rows=500)
    if (
        len(X_test) > 0
        and artifacts.label_encoder is not None
        and any(m in artifacts.models for m in ("XGBoost", "MLP"))
    ):
        class_names = [str(c) for c in artifacts.label_encoder.classes_]
        avail = [m for m in ("XGBoost", "MLP") if m in artifacts.models]
        c_sel1, c_sel2, c_sel3 = st.columns([1, 1, 2])
        with c_sel1:
            exp_model = st.selectbox("Model to Explain", options=avail)
        with c_sel2:
            sample_idx = st.number_input(
                "Test Sample Index", min_value=0, max_value=len(X_test) - 1, value=0, step=1
            )
        with c_sel3:
            true_cls_name = class_names[int(y_test[int(sample_idx)])]
            st.info(f"True Ground-Truth Class at Index `{sample_idx}`: **{true_cls_name}**")

        if st.button("Explain Selected Test Flow", type="primary"):
            row = X_test[int(sample_idx)]
            model_obj = artifacts.models[exp_model]
            probs = (
                model_obj.predict_proba(row.reshape(1, -1))[0]
                if exp_model == "XGBoost"
                else model_obj.predict(row.reshape(1, -1), verbose=0)[0]
            )
            pred_idx = int(np.argmax(probs))
            pred_cls = class_names[pred_idx]
            pred_conf = float(probs[pred_idx])

            exp_res = explain_single_sample(
                model_name=exp_model,
                model=model_obj,
                x_single=row,
                X_background=X_test[:100],
                feature_names=artifacts.feature_names,
                class_idx=pred_idx,
                class_name=pred_cls,
                max_display=15,
            )
            st.success(
                f"Predicted **{pred_cls}** (`{pred_conf * 100:.2f}%`) using `{exp_res['explainer']}` "
                f"(Base value={exp_res['base_value']:.4f})"
            )
            tf_df = pd.DataFrame(exp_res["top_features"]).sort_values("abs_shap", ascending=True)
            tf_df["Impact"] = np.where(tf_df["shap_value"] >= 0, "Pushes Toward Prediction", "Pushes Away")
            fig_local = px.bar(
                tf_df,
                x="shap_value",
                y="feature",
                color="Impact",
                orientation="h",
                color_discrete_map={"Pushes Toward Prediction": "#ef4444", "Pushes Away": "#38bdf8"},
                title=f"Local Feature Contributions for Flow #{sample_idx} -> '{pred_cls}'",
            )
            st.plotly_chart(apply_soc_plotly_theme(fig_local, height=440), use_container_width=True)
