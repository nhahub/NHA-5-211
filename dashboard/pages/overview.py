"""Tab 1: Overview — SOC KPIs, Detection Log, Analytics, and Threat Map."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard.components.cards import render_metric_card, threat_badge_text
from dashboard.components.charts import (
    plot_attack_class_donut,
    plot_attack_port_heatmap,
    plot_confidence_histogram,
    plot_threat_level_donut,
    plot_timeline_area,
    plot_top_attacking_ips,
    plot_top_destination_ports,
)
from dashboard.data_access import (
    build_test_detection_logs,
    generate_demo_logs,
    get_detection_engine,
    load_cached_artifacts,
)
from src.data.feature_engineering import SchemaValidationError


def render_overview_page(sidebar_state: dict[str, Any]) -> pd.DataFrame:
    """Render SOC Overview KPI strip, filterable detection log, analytics, and threat map."""
    st.subheader("Security Operations Center — Real-Time Telemetry Overview")

    mode = sidebar_state["data_source_mode"]
    active_model = sidebar_state["active_model"]
    conf_floor = sidebar_state["confidence_threshold"]
    anom_thresh = sidebar_state["anomaly_threshold"]

    df_logs = pd.DataFrame()
    if mode == "Test Evaluation Split":
        df_logs = build_test_detection_logs(
            model_choice=active_model,
            max_rows=1000,
            min_actionable_confidence=conf_floor,
        )
        if df_logs.empty:
            st.info(
                "No trained test-split logs available yet. Showing synthetic demo telemetry until training completes."
            )
            df_logs = generate_demo_logs(n_rows=250)
    elif mode == "Upload Flow CSV":
        artifacts = load_cached_artifacts()
        if not artifacts.models or not artifacts.feature_names:
            st.warning(
                "Trained model binaries are absent or un-pulled Git LFS pointers. Run `git lfs pull` or "
                "`python scripts/train.py --run-eval` before running live CSV inference."
            )
            return pd.DataFrame()
        uploaded = st.file_uploader(
            "Upload CICIDS2017-compatible flow CSV for SOC overview analysis",
            type=["csv"],
            key="overview_csv_uploader",
        )
        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded, low_memory=False)
                engine = get_detection_engine()
                df_logs, _ = engine.predict_batch(
                    raw_df.head(2000),
                    model_choice=active_model,
                    anomaly_threshold_override=anom_thresh,
                    min_actionable_confidence=conf_floor,
                    trigger_response=False,
                )
            except SchemaValidationError as exc:
                st.error(f"CSV Schema Error: {exc}")
                return pd.DataFrame()
            except Exception as exc:
                st.error(f"Failed processing uploaded CSV: {exc}")
                return pd.DataFrame()
        else:
            st.info("Upload a CICIDS2017 flow CSV above, or switch Telemetry Source in the sidebar.")
            return pd.DataFrame()
    else:
        df_logs = generate_demo_logs(n_rows=300)

    if df_logs.empty:
        st.warning("No flow telemetry loaded.")
        return df_logs

    # Compute KPI metrics reactively
    total_flows = len(df_logs)
    benign_count = int((df_logs["attack_class"].str.upper() == "BENIGN").sum())
    attack_count = total_flows - benign_count
    critical_count = int((df_logs["threat_level"] == "Critical").sum())
    anomaly_count = int(df_logs["is_anomaly"].sum()) if "is_anomaly" in df_logs.columns else 0
    det_rate = (attack_count / max(1, total_flows)) * 100.0
    artifacts = load_cached_artifacts()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_metric_card("Total Analyzed Flows", f"{total_flows:,}", f"Model: {active_model}")
    with c2:
        render_metric_card(
            "Benign Traffic", f"{benign_count:,}", f"{(benign_count / max(1, total_flows)) * 100:.1f}% share"
        )
    with c3:
        render_metric_card("Detected Attacks", f"{attack_count:,}", f"{det_rate:.1f}% detection rate")
    with c4:
        render_metric_card("Critical Alerts", f"{critical_count:,}", "Immediate BLOCK tier")
    with c5:
        render_metric_card("Zero-Day Anomalies", f"{anomaly_count:,}", f"MSE > {anom_thresh:.2f}")
    with c6:
        render_metric_card(
            "Models Online",
            f"{len(artifacts.models)} / 4",
            "Hybrid Ensemble Ready" if artifacts.models else "Not Trained",
        )

    st.markdown("---")
    sub_log, sub_analytics, sub_map = st.tabs(
        ["📋 Recent Security Events", "📊 Traffic & Confidence Analytics", "🗺️ Threat Map"]
    )

    with sub_log:
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_threats = st.multiselect(
                "Filter Threat Level",
                options=["Low", "Medium", "High", "Critical"],
                default=["Medium", "High", "Critical"],
            )
        with f2:
            all_classes = sorted(df_logs["attack_class"].unique().tolist())
            sel_classes = st.multiselect(
                "Filter Attack Class",
                options=all_classes,
                default=all_classes,
            )
        with f3:
            min_conf_filter = st.slider("Minimum Confidence Filter", 0.0, 1.0, 0.0, 0.05)

        mask = (
            df_logs["threat_level"].isin(sel_threats)
            & df_logs["attack_class"].isin(sel_classes)
            & (df_logs["confidence"] >= min_conf_filter)
        )
        filtered = df_logs[mask].copy()
        if not filtered.empty:
            filtered["threat_badge"] = filtered["threat_level"].map(threat_badge_text)
        st.caption(f"Showing {len(filtered):,} of {len(df_logs):,} analyzed flows")

        display_cols = [
            c
            for c in [
                "timestamp",
                "src_ip",
                "dst_port",
                "attack_class",
                "confidence",
                "anomaly_score",
                "threat_badge",
                "action",
                "model_used",
            ]
            if c in filtered.columns
        ]
        st.dataframe(
            filtered[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.TextColumn("🕒 Timestamp"),
                "src_ip": st.column_config.TextColumn("🌐 Source IP"),
                "dst_port": st.column_config.NumberColumn("🔌 Dst Port", format="%d"),
                "attack_class": st.column_config.TextColumn("🎯 Predicted Class"),
                "confidence": st.column_config.ProgressColumn(
                    "📈 Confidence", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "anomaly_score": st.column_config.NumberColumn("🧬 AE MSE", format="%.3f"),
                "threat_badge": st.column_config.TextColumn("⚠️ Threat Tier"),
                "action": st.column_config.TextColumn("🛡️ Action"),
                "model_used": st.column_config.TextColumn("🧠 Engine"),
            },
        )
        st.download_button(
            label="⬇️ Export Filtered Events (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="smart_ids_filtered_events.csv",
            mime="text/csv",
        )

    with sub_analytics:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_attack_class_donut(df_logs), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_threat_level_donut(df_logs), use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.plotly_chart(plot_timeline_area(df_logs), use_container_width=True)
        with col_d:
            st.plotly_chart(plot_confidence_histogram(df_logs), use_container_width=True)

    with sub_map:
        m1, m2 = st.columns(2)
        with m1:
            st.plotly_chart(plot_top_attacking_ips(df_logs), use_container_width=True)
        with m2:
            st.plotly_chart(plot_top_destination_ports(df_logs), use_container_width=True)
        st.plotly_chart(plot_attack_port_heatmap(df_logs), use_container_width=True)

    return df_logs
