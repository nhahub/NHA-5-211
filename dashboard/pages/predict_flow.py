"""Tab 4: Predict Flow — Single-Flow Interactive Inference & Schema-Validated Batch CSV Prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.cards import apply_soc_plotly_theme, threat_badge_html
from dashboard.data_access import get_detection_engine, load_cached_artifacts, load_test_split_sample
from src.data.feature_engineering import SchemaValidationError


def render_predict_flow_page(sidebar_state: dict[str, Any]) -> None:
    """Render live single-flow prediction and batch CSV inference tabs."""
    st.subheader("Live Network Flow Prediction & Batch CSV Inference")

    artifacts = load_cached_artifacts()
    if not artifacts.models or not artifacts.feature_names:
        hint = (
            "Un-pulled Git LFS pointer files were detected in `models/` — run `git lfs pull` first."
            if artifacts.lfs_pointers
            else "Run `git lfs pull` or `python scripts/train.py --run-eval` first to populate `models/`."
        )
        st.error(f"Trained models and feature schema are not loaded. {hint}")
        return

    engine = get_detection_engine()
    active_model = sidebar_state["active_model"]
    conf_floor = sidebar_state["confidence_threshold"]
    anom_thresh = sidebar_state["anomaly_threshold"]

    tab_single, tab_batch = st.tabs(["🎛️ Single Flow Inference", "📁 Batch CSV Prediction"])

    with tab_single:
        st.caption(
            f"Configure core flow attributes below (remaining {len(artifacts.feature_names) - 11} features default to "
            "median/baseline values from the training scaler so all inputs remain schema-consistent)."
        )

        # Build sensible default vector from scaler empirical means
        defaults = {
            fname: float(artifacts.scaler.mean_[idx]) if artifacts.scaler is not None else 0.0
            for idx, fname in enumerate(artifacts.feature_names)
        }

        preset = st.selectbox(
            "Quick Preset Profile",
            options=[
                "Training Baseline (Mean)",
                "HTTP Flood Pattern",
                "PortScan Probe Pattern",
                "SSH Brute-Force Pattern",
            ],
        )
        if preset == "HTTP Flood Pattern":
            defaults.update(
                {
                    "Destination Port": 80.0,
                    "Flow Duration": 450000.0,
                    "Total Fwd Packets": 18.0,
                    "Fwd Packet Length Mean": 520.0,
                    "Bwd Packet Length Mean": 1100.0,
                    "Flow Bytes/s": 125000.0,
                    "Flow Packets/s": 420.0,
                    "SYN Flag Count": 2.0,
                    "PSH Flag Count": 4.0,
                    "ACK Flag Count": 12.0,
                }
            )
        elif preset == "PortScan Probe Pattern":
            defaults.update(
                {
                    "Destination Port": 445.0,
                    "Flow Duration": 250.0,
                    "Total Fwd Packets": 2.0,
                    "Fwd Packet Length Mean": 40.0,
                    "Bwd Packet Length Mean": 20.0,
                    "Flow Bytes/s": 240000.0,
                    "Flow Packets/s": 8000.0,
                    "SYN Flag Count": 2.0,
                    "FIN Flag Count": 0.0,
                    "PSH Flag Count": 0.0,
                    "ACK Flag Count": 0.0,
                }
            )
        elif preset == "SSH Brute-Force Pattern":
            defaults.update(
                {
                    "Destination Port": 22.0,
                    "Flow Duration": 180000.0,
                    "Total Fwd Packets": 22.0,
                    "Fwd Packet Length Mean": 140.0,
                    "Bwd Packet Length Mean": 160.0,
                    "Flow Bytes/s": 35000.0,
                    "Flow Packets/s": 210.0,
                    "SYN Flag Count": 1.0,
                    "PSH Flag Count": 4.0,
                    "ACK Flag Count": 18.0,
                }
            )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            src_ip = st.text_input("Source IP", value="203.0.113.42")
            dst_port = st.number_input(
                "Destination Port",
                min_value=0,
                max_value=65535,
                value=int(max(0, defaults.get("Destination Port", 80))),
            )
            flow_dur = st.number_input(
                "Flow Duration (µs)",
                min_value=0.0,
                value=float(max(0.0, defaults.get("Flow Duration", 1000.0))),
            )
        with c2:
            fwd_pkts = st.number_input(
                "Total Fwd Packets",
                min_value=0.0,
                value=float(max(0.0, defaults.get("Total Fwd Packets", 5.0))),
            )
            fwd_len = st.number_input(
                "Fwd Packet Length Mean",
                min_value=0.0,
                value=float(max(0.0, defaults.get("Fwd Packet Length Mean", 100.0))),
            )
            bwd_len = st.number_input(
                "Bwd Packet Length Mean",
                min_value=0.0,
                value=float(max(0.0, defaults.get("Bwd Packet Length Mean", 100.0))),
            )
        with c3:
            flow_bps = st.number_input(
                "Flow Bytes/s", min_value=0.0, value=float(max(0.0, defaults.get("Flow Bytes/s", 5000.0)))
            )
            flow_pps = st.number_input(
                "Flow Packets/s", min_value=0.0, value=float(max(0.0, defaults.get("Flow Packets/s", 50.0)))
            )
            syn_cnt = st.number_input(
                "SYN Flag Count", min_value=0.0, value=float(max(0.0, defaults.get("SYN Flag Count", 1.0)))
            )
        with c4:
            fin_cnt = st.number_input(
                "FIN Flag Count", min_value=0.0, value=float(max(0.0, defaults.get("FIN Flag Count", 0.0)))
            )
            psh_cnt = st.number_input(
                "PSH Flag Count", min_value=0.0, value=float(max(0.0, defaults.get("PSH Flag Count", 1.0)))
            )
            ack_cnt = st.number_input(
                "ACK Flag Count", min_value=0.0, value=float(max(0.0, defaults.get("ACK Flag Count", 3.0)))
            )

        log_to_response = st.checkbox("Record decision in Active Response audit history", value=True)

        if st.button("🚀 Run Live Flow Detection", type="primary"):
            row = dict(defaults)
            row.update(
                {
                    "Source IP": src_ip,
                    "Destination Port": float(dst_port),
                    "Flow Duration": float(flow_dur),
                    "Total Fwd Packets": float(fwd_pkts),
                    "Fwd Packet Length Mean": float(fwd_len),
                    "Bwd Packet Length Mean": float(bwd_len),
                    "Flow Bytes/s": float(flow_bps),
                    "Flow Packets/s": float(flow_pps),
                    "SYN Flag Count": float(syn_cnt),
                    "FIN Flag Count": float(fin_cnt),
                    "PSH Flag Count": float(psh_cnt),
                    "ACK Flag Count": float(ack_cnt),
                }
            )
            df_in = pd.DataFrame([row])
            X_test_bg, _, _ = load_test_split_sample(max_rows=100)
            try:
                res_df, preds = engine.predict_batch(
                    df_in,
                    model_choice=active_model,
                    anomaly_threshold_override=anom_thresh,
                    min_actionable_confidence=conf_floor,
                    trigger_response=log_to_response,
                    X_background=X_test_bg if len(X_test_bg) > 0 else None,
                    explain_top_k=10,
                )
                pred = preds[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted Class", pred.predicted_class)
                m2.metric("Confidence Score", f"{pred.confidence_score * 100:.2f}%")
                m3.metric(
                    "Autoencoder Anomaly MSE",
                    f"{pred.anomaly_score:.4f}",
                    delta="ANOMALOUS" if pred.is_anomaly else "Normal",
                    delta_color="inverse" if pred.is_anomaly else "normal",
                )
                with m4:
                    st.markdown("**Threat Tier & Action**")
                    st.markdown(
                        f"{threat_badge_html(pred.threat_level)} &nbsp; `{pred.recommended_response['action']}`",
                        unsafe_allow_html=True,
                    )

                col_p, col_s = st.columns(2)
                with col_p:
                    prob_df = pd.DataFrame(
                        {
                            "Class": list(pred.class_probabilities.keys()),
                            "Probability": list(pred.class_probabilities.values()),
                        }
                    ).sort_values("Probability", ascending=True)
                    fig_p = px.bar(
                        prob_df,
                        x="Probability",
                        y="Class",
                        orientation="h",
                        title=f"Per-Class Probability Distribution ({pred.selected_model})",
                    )
                    st.plotly_chart(apply_soc_plotly_theme(fig_p, height=340), use_container_width=True)

                with col_s:
                    if pred.important_contributing_features:
                        sf_df = pd.DataFrame(pred.important_contributing_features).sort_values(
                            "abs_shap", ascending=True
                        )
                        sf_df["Direction"] = np.where(
                            sf_df["shap_value"] >= 0, "Increases Target Prob", "Decreases Target Prob"
                        )
                        fig_sf = px.bar(
                            sf_df,
                            x="shap_value",
                            y="feature",
                            color="Direction",
                            orientation="h",
                            color_discrete_map={
                                "Increases Target Prob": "#ef4444",
                                "Decreases Target Prob": "#38bdf8",
                            },
                            title="Local SHAP Feature Attribution for This Flow",
                        )
                        st.plotly_chart(apply_soc_plotly_theme(fig_sf, height=340), use_container_width=True)

                with st.expander("🛡️ Recommended Active Response Decision"):
                    st.json(pred.recommended_response)
            except Exception as exc:
                st.error(f"Inference error: {exc}")

    with tab_batch:
        st.markdown(
            "Upload a CSV containing raw CICIDS2017 flow columns (`77` required numeric features; extra IDs or `Label` are safely ignored)."
        )
        batch_file = st.file_uploader("Upload Batch Flow CSV", type=["csv"], key="batch_predict_csv")
        if batch_file is not None:
            try:
                batch_raw = pd.read_csv(batch_file, low_memory=False)
                st.write(f"Loaded `{len(batch_raw):,}` rows x `{batch_raw.shape[1]}` columns.")
                if st.button("⚡ Run Batch Predictions", type="primary"):
                    out_df, _ = engine.predict_batch(
                        batch_raw,
                        model_choice=active_model,
                        anomaly_threshold_override=anom_thresh,
                        min_actionable_confidence=conf_floor,
                        trigger_response=False,
                    )
                    st.success(f"Completed batch prediction on {len(out_df):,} flows.")
                    st.dataframe(out_df.head(100), use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        data=out_df.to_csv(index=False).encode("utf-8"),
                        file_name="smart_ids_batch_predictions.csv",
                        mime="text/csv",
                    )
            except SchemaValidationError as exc:
                st.error(f"Schema Validation Failed: {exc}")
            except Exception as exc:
                st.error(f"Batch inference failed: {exc}")
