"""Tab 6: Active Response — Safe Dry-Run Mitigation Simulator, Audit Trail & Incident Export."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.components.cards import threat_badge_html
from dashboard.data_access import get_response_engine, load_cached_artifacts
from src.config import get_app_config


def render_active_response_page(sidebar_state: dict[str, Any], overview_logs: pd.DataFrame) -> None:
    """Render manual mitigation trigger, Critical alert batching, blocked IP table, and audit log."""
    st.subheader("Active Response & Mitigation Simulation Engine")
    st.info(
        "🛡️ **Academic Safety Mode Active:** All firewall commands execute strictly in **DRY-RUN / SIMULATION** mode. "
        "No host `iptables` or OS network rules are modified."
    )

    resp_engine = get_response_engine()
    artifacts = load_cached_artifacts()
    class_names = (
        [str(c) for c in artifacts.label_encoder.classes_]
        if artifacts.label_encoder is not None
        else [
            "BENIGN",
            "DDoS",
            "DoS GoldenEye",
            "DoS Hulk",
            "FTP-Patator",
            "PortScan",
            "SSH-Patator",
            "Web Attack",
        ]
    )
    conf_floor = sidebar_state["confidence_threshold"]

    col_form, col_ref = st.columns([1.3, 1.0])
    with col_form:
        st.markdown("#### 🎯 Manual Response Trigger")
        with st.form("manual_response_form"):
            r1, r2 = st.columns(2)
            with r1:
                ip_input = st.text_input(
                    "Source IP (try `INVALID_IP` to test validation gate)", value="203.0.113.55"
                )
                cls_input = st.selectbox(
                    "Detected Attack Class", options=class_names, index=min(1, len(class_names) - 1)
                )
            with r2:
                conf_input = st.slider("Detection Confidence", 0.0, 1.0, 0.96, 0.01)
                model_input = st.selectbox(
                    "Detecting Model", options=["Hybrid Ensemble", "XGBoost", "MLP", "CNN", "Autoencoder"]
                )
            anom_flag = st.checkbox("Autoencoder Zero-Day Anomaly Flag", value=False)
            submitted = st.form_submit_button("Execute Simulated Response", type="primary")

        if submitted:
            decision = resp_engine.apply_active_response(
                src_ip=ip_input,
                attack_class=cls_input,
                confidence=conf_input,
                model_used=model_input,
                dry_run=True,
                is_anomaly=anom_flag,
                anomaly_ratio=2.5 if anom_flag else 0.5,
                min_actionable_confidence=conf_floor,
            )
            st.markdown(
                f"**Resolved Threat Tier:** {threat_badge_html(decision['threat_level'])}",
                unsafe_allow_html=True,
            )
            st.json(decision)

        # Batch process top Critical events from active Overview logs
        if not overview_logs.empty:
            crit_rows = overview_logs[overview_logs["threat_level"] == "Critical"].head(5)
            if st.button(f"⚡ Auto-Process Up to 5 Critical Alerts from Active Log ({len(crit_rows)} ready)"):
                if crit_rows.empty:
                    st.info("No Critical-tier flows found in the current Overview log.")
                else:
                    for _, row in crit_rows.iterrows():
                        resp_engine.apply_active_response(
                            src_ip=str(row.get("src_ip", "0.0.0.0")),
                            attack_class=str(row.get("attack_class", "Unknown")),
                            confidence=float(row.get("confidence", 0.95)),
                            model_used=str(row.get("model_used", "Hybrid Ensemble")),
                            dry_run=True,
                            is_anomaly=bool(row.get("is_anomaly", False)),
                            anomaly_score=float(row.get("anomaly_score", 0.0)),
                        )
                    st.success(
                        f"Processed {len(crit_rows)} Critical alert(s) into simulated mitigation audit log."
                    )

    with col_ref:
        st.markdown("#### 📜 Four-Tier Threat Taxonomy")
        taxonomy_df = pd.DataFrame(
            [
                {
                    "Tier": "🟢 Low",
                    "Confidence": "0.00 – 0.40 (or Benign)",
                    "Action": "LOG",
                    "Behavior": "Passive statistical logging",
                },
                {
                    "Tier": "🟡 Medium",
                    "Confidence": "0.40 – 0.70",
                    "Action": "ALERT",
                    "Behavior": "SOC/SIEM analyst review flag",
                },
                {
                    "Tier": "🟠 High",
                    "Confidence": "0.70 – 0.90",
                    "Action": "RATE_LIMIT",
                    "Behavior": "Simulated 100 pps ceiling",
                },
                {
                    "Tier": "🔴 Critical",
                    "Confidence": "0.90 – 1.00",
                    "Action": "BLOCK",
                    "Behavior": "Simulated `iptables -I INPUT 1 -s <ip> -j DROP`",
                },
            ]
        )
        st.dataframe(taxonomy_df, use_container_width=True, hide_index=True)

        st.markdown("#### 🚫 Simulated Blocked IPs Registry")
        if resp_engine.blocked_ips:
            blocked_list = sorted(resp_engine.blocked_ips)
            st.write(blocked_list)
            ip_to_unblock = st.selectbox("Select IP to Simulated-Unblock", options=blocked_list)
            if st.button("Remove Simulated Block"):
                resp_engine.unblock_ip(ip_to_unblock)
                st.rerun()
        else:
            st.caption("No source IPs currently in simulated block registry.")

    st.markdown("---")
    st.markdown("#### 🧾 Incident Response History & Audit Export")
    if resp_engine.history:
        hist_df = pd.DataFrame(list(reversed(resp_engine.history)))
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        ex1, ex2, ex3 = st.columns([1, 1, 2])
        with ex1:
            st.download_button(
                "⬇️ Export Incidents (JSON)",
                data=json.dumps(resp_engine.history, indent=2).encode("utf-8"),
                file_name="smart_ids_incident_report.json",
                mime="application/json",
            )
        with ex2:
            st.download_button(
                "⬇️ Export Incidents (CSV)",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="smart_ids_incident_report.csv",
                mime="text/csv",
            )
        with ex3:
            if st.button("🗑️ Clear Session Incident History"):
                resp_engine.clear_history()
                resp_engine.save_history()
                st.rerun()
    else:
        st.caption("No mitigation events recorded yet in this session.")

    log_path = get_app_config().paths.resolve("response_log_file")
    if log_path.exists():
        with st.expander("📄 Persistent File Audit Trail (`ids_response_log.txt`)"):
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:]
            st.code("\n".join(lines), language="text")
