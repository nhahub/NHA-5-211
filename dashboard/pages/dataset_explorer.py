"""Tab 2: Dataset Explorer — Upload, EDA, Correlations, Missing Values, and Audit Report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.cards import apply_soc_plotly_theme, render_metric_card
from dashboard.data_access import load_preprocessing_audit, load_sample_raw_dataset
from src.data.cleaner import normalize_column_names


def render_dataset_explorer_page() -> None:
    """Render dataset preview, class distribution, correlation heatmap, and quality audit."""
    st.subheader("Dataset Explorer & Data Quality Audit")

    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload a custom CICIDS2017 CSV to explore (optional)",
            type=["csv"],
            key="explorer_csv_upload",
        )
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, nrows=10000, low_memory=False)
            df = normalize_column_names(df)
            source_desc = f"Uploaded CSV: `{uploaded.name}` ({len(df):,} rows loaded)"
        except Exception as exc:
            st.error(f"Could not parse uploaded CSV file: {exc}")
            return
    else:
        df, source_desc = load_sample_raw_dataset(max_rows=5000)

    with col_info:
        st.info(f"**Active Corpus View:** {source_desc}")

    audit = load_preprocessing_audit()
    if audit:
        with st.expander("📋 Preprocessing & Leakage Audit Report (Full Training Pipeline)", expanded=True):
            a1, a2, a3, a4 = st.columns(4)
            orig_shape = audit.get("original_shape", [0, 0])
            splits = audit.get("split_sizes", {})
            with a1:
                render_metric_card("Original Corpus", f"{orig_shape[0]:,}", f"{orig_shape[1]} raw columns")
            with a2:
                render_metric_card(
                    "Duplicates / Inf",
                    f"{audit.get('duplicates_removed', 0):,}",
                    f"{audit.get('inf_values_converted', 0):,} Inf -> NaN",
                )
            with a3:
                render_metric_card(
                    "Final Features",
                    f"{audit.get('final_feature_count', 0)}",
                    f"{len(audit.get('id_columns_removed', []))} ID cols dropped",
                )
            with a4:
                render_metric_card(
                    "Train (SMOTE)",
                    f"{splits.get('train_smote', 0):,}",
                    f"Val {splits.get('validation', 0):,} | Test {splits.get('test', 0):,}",
                )

            leak = audit.get("leakage_checks", {})
            if leak:
                if leak.get("passed"):
                    st.success(
                        f"✅ Data Leakage Verification Passed — Scaler fit strictly on `{leak.get('scaler_n_samples_seen', 0):,}` "
                        f"training rows; Train/Test exact overlap = `{leak.get('train_test_exact_overlap_sample', 0)}`."
                    )
                else:
                    st.warning(f"Leakage audit details: {leak}")

            col_pre, col_post = st.columns(2)
            with col_pre:
                orig_dist = audit.get("original_class_distribution", {})
                if orig_dist:
                    df_orig = pd.DataFrame(
                        {"Class": list(orig_dist.keys()), "Count": list(orig_dist.values())}
                    )
                    fig_o = px.bar(
                        df_orig,
                        x="Count",
                        y="Class",
                        orientation="h",
                        title="Original Class Distribution (Imbalanced)",
                        color="Count",
                        color_continuous_scale="Teal",
                    )
                    st.plotly_chart(apply_soc_plotly_theme(fig_o, height=340), use_container_width=True)
            with col_post:
                sm_dist = audit.get("train_post_smote_distribution", {})
                if sm_dist:
                    df_sm = pd.DataFrame({"Class": list(sm_dist.keys()), "Count": list(sm_dist.values())})
                    fig_s = px.bar(
                        df_sm,
                        x="Count",
                        y="Class",
                        orientation="h",
                        title="Post-SMOTE Training Distribution (Balanced)",
                        color="Count",
                        color_continuous_scale="Emrld",
                    )
                    st.plotly_chart(apply_soc_plotly_theme(fig_s, height=340), use_container_width=True)

    if df.empty:
        st.warning("No sample CSV available on disk to inspect.")
        return

    st.markdown("#### 🔍 Sample Records Preview")
    st.dataframe(df.head(50), use_container_width=True)

    # Data quality diagnostics on loaded sample
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    inf_counts = (
        np.isinf(numeric_df.to_numpy(dtype=np.float64, copy=False)).sum(axis=0)
        if not numeric_df.empty
        else np.array([])
    )
    nan_counts = df.isna().sum()
    quality_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Dtype": [str(t) for t in df.dtypes],
            "Missing_Count": [int(nan_counts[c]) for c in df.columns],
            "Missing_Pct": [round(float(nan_counts[c] / max(1, len(df)) * 100.0), 3) for c in df.columns],
        }
    )
    if not numeric_df.empty:
        inf_map = {col: int(cnt) for col, cnt in zip(numeric_df.columns, inf_counts, strict=False)}
        quality_df["Inf_Count"] = [inf_map.get(c, 0) for c in df.columns]

    q1, q2 = st.columns([1.2, 1.8])
    with q1:
        st.markdown("#### 🧪 Missing & Infinity Profile")
        st.dataframe(
            quality_df[quality_df["Missing_Count"] + quality_df.get("Inf_Count", 0) > 0]
            if (
                quality_df["Missing_Count"].sum() > 0 or quality_df.get("Inf_Count", pd.Series([0])).sum() > 0
            )
            else quality_df.head(20),
            use_container_width=True,
            hide_index=True,
        )

    with q2:
        st.markdown("#### 📈 Feature Summary Statistics")
        if not numeric_df.empty:
            clean_num = numeric_df.replace([np.inf, -np.inf], np.nan)
            st.dataframe(
                clean_num.describe().T[["count", "mean", "std", "min", "50%", "max"]].head(25),
                use_container_width=True,
            )

    if not numeric_df.empty and numeric_df.shape[1] >= 4:
        st.markdown("#### 🔥 Top-15 High-Variance Feature Correlation Heatmap")
        clean_num = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        variances = clean_num.var(numeric_only=True).sort_values(ascending=False)
        top_cols = variances.head(15).index.tolist()
        if len(top_cols) >= 2:
            corr = clean_num[top_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
                title="Pearson Correlation Matrix (Top 15 Variance Features)",
            )
            st.plotly_chart(apply_soc_plotly_theme(fig_corr, height=520), use_container_width=True)
